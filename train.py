import os
from tqdm.auto import tqdm
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from models import ConditionedUNetConfig, DenoiserConfig, SigmaDistributionConfig, Denoiser
from data import SequenceMazeDataset, collate_maze_sequences
import pandas as pd
import argparse
from torch.amp import autocast
from torch.cuda.amp import GradScaler
import wandb

def get_lr_lambda(current_step: int, warmup_steps: int):
    if current_step < warmup_steps:
        return current_step / warmup_steps
    return 1.0


def setup_distributed():
    """Setup distributed training if available and requested."""
    if not torch.cuda.is_available():
        return False, 0, 1
        
    # Check if we're running under torchrun/torch.distributed.launch
    if "LOCAL_RANK" not in os.environ:
        return False, 0, 1

    # Initialize distributed process group
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl")
    
    return True, local_rank, world_size

def get_device(args):
    """Get appropriate device based on arguments and availability."""
    if not torch.cuda.is_available() or args.cpu:
        return "cpu"
    return f"cuda:{args.local_rank}" if args.distributed else "cuda:0"

def train_one_epoch(model, train_loader, optimizer, scheduler, device, 
                   gradient_accumulation_steps, max_grad_norm, is_main_process, scaler,
                   global_step, log_every_n_steps, val_loader):
    model.train()
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(train_loader, desc=f"Training", disable=not is_main_process)
    running_loss = 0.0
    running_path_loss = 0.0  # Track path segmentation loss
    
    for step, batch in enumerate(progress_bar):
        batch = batch.to(device)
        
        with autocast(device_type="cuda"):
            loss, logs = model(batch)  # Get both loss and logs
            loss = loss / gradient_accumulation_steps

        try:
            scaler.scale(loss).backward()
            running_loss += loss.item() * gradient_accumulation_steps
        except Exception as e:
            print(f"Error during backward pass: {e}")
            print(f"Loss: {loss}")
            print(f"Logs: {logs}")
            print(f"Batch: {batch}")
            raise e
        
        # Extract and accumulate path loss if available
        if 'loss_path' in logs:
            running_path_loss += logs['loss_path'] * gradient_accumulation_steps
        
        if (step + 1) % gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            scaler.step(optimizer)
            optimizer.zero_grad()
            scheduler.step()
            scaler.update()
            
            global_step += 1
            
            # Log metrics every n steps
            if is_main_process and global_step % log_every_n_steps == 0:
                val_loss, val_logs = validate(model, val_loader, device)
                
                # Extract individual loss components if available
                train_denoising_loss = logs.get('loss_denoising', 0.0)
                train_path_loss = logs.get('loss_path', 0.0)
                train_total_loss = logs.get('loss_total', running_loss/gradient_accumulation_steps)
                
                val_denoising_loss = val_logs.get('loss_denoising', 0.0)
                val_path_loss = val_logs.get('loss_path', 0.0)
                val_total_loss = val_loss
                
                metrics = {
                    'step': global_step,
                    'train_loss': train_total_loss,
                    'train_denoising_loss': train_denoising_loss,
                    'train_path_loss': train_path_loss,
                    'val_loss': val_total_loss,
                    'val_denoising_loss': val_denoising_loss,
                    'val_path_loss': val_path_loss,
                    'learning_rate': scheduler.get_last_lr()[0]
                }
                
                wandb.log(metrics)
                
                progress_bar.set_postfix({
                    'total': f'{train_total_loss:.4f}',
                    'denoise': f'{train_denoising_loss:.4f}',
                    'path': f'{train_path_loss:.4f}',
                    'val': f'{val_total_loss:.4f}',
                    'val_path': f'{val_path_loss:.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.6f}'
                })
            else:
                # Extract individual loss components if available
                train_denoising_loss = logs.get('loss_denoising', 0.0)
                train_path_loss = logs.get('loss_path', 0.0)
                train_total_loss = logs.get('loss_total', running_loss/gradient_accumulation_steps)
                
                progress_bar.set_postfix({
                    'total': f'{train_total_loss:.4f}',
                    'denoise': f'{train_denoising_loss:.4f}',
                    'path': f'{train_path_loss:.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.6f}'
                })
                
            running_loss = 0.0
            running_path_loss = 0.0  # Reset running path loss
            
        total_loss += loss.item() * gradient_accumulation_steps
        num_batches += 1
    
    progress_bar.close()
    return total_loss / num_batches, global_step

def validate(model, val_loader, device):
    model.eval()
    total_loss = 0
    num_batches = 0
    total_denoising_loss = 0
    total_path_loss = 0
    
    progress_bar = tqdm(val_loader, desc=f"Validating", leave=False)
    
    with torch.no_grad(), autocast(device_type="cuda"):
        for batch in progress_bar:
            batch = batch.to(device)
            loss, logs = model(batch)
            
            # Extract individual loss components
            denoising_loss = logs.get('loss_denoising', 0.0)
            path_loss = logs.get('loss_path', 0.0)
            
            total_loss += loss.item()
            total_denoising_loss += denoising_loss
            total_path_loss += path_loss
            
            num_batches += 1
            
            progress_bar.set_postfix({
                'val_total': f'{total_loss/num_batches:.4f}',
                'val_denoise': f'{total_denoising_loss/num_batches:.4f}',
                'val_path': f'{total_path_loss/num_batches:.4f}'
            })
    
    progress_bar.close()
    
    # Return both the total loss and a dictionary of component losses
    return total_loss / num_batches, {
        'loss_denoising': total_denoising_loss / num_batches,
        'loss_path': total_path_loss / num_batches,
        'loss_total': total_loss / num_batches
    }

def parse_args():
    parser = argparse.ArgumentParser(description='Train denoising model')
    
    # Training infrastructure arguments
    parser.add_argument('--cpu', action='store_true', help='Force CPU training')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size per GPU')
    parser.add_argument('--distributed', action='store_true', help='Enable distributed training (multi-GPU)')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers')
    
    # Training hyperparameters
    parser.add_argument('--num-epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-2, help='Weight decay')
    parser.add_argument('--max-grad-norm', type=float, default=5, help='Maximum gradient norm')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=8, help='Gradient accumulation steps')
    parser.add_argument('--warmup-steps', type=int, default=50, help='Learning rate warmup steps')
    
    # Model configuration
    parser.add_argument('--img-channels', type=int, default=3, help='Number of image channels')
    parser.add_argument('--num-conditioning-steps', type=int, default=4, help='Number of conditioning steps')
    parser.add_argument('--cond-channels', type=int, default=256, help='Number of conditioning channels')
    parser.add_argument('--model-depths', nargs='+', type=int, default=[2, 2, 2, 2], help='Model depths per layer')
    parser.add_argument('--model-channels', nargs='+', type=int, default=[64, 64, 64, 64], help='Model channels per layer')
    parser.add_argument('--attn-depths', nargs='+', type=bool, default=[False, False, False, False], help='Attention at each depth')
    parser.add_argument('--num-actions', type=int, default=4, help='Number of possible actions')
    
    # Denoiser configuration
    parser.add_argument('--sigma-data', type=float, default=0.5, help='Sigma data parameter')
    parser.add_argument('--sigma-offset-noise', type=float, default=0.3, help='Sigma offset noise parameter')
    parser.add_argument('--noise-previous-obs', type=bool, default=True, help='Whether to noise previous observations')
    
    # Sigma distribution configuration
    parser.add_argument('--sigma-loc', type=float, default=-0.3, help='Sigma distribution location parameter')
    parser.add_argument('--sigma-scale', type=float, default=1.0, help='Sigma distribution scale parameter')
    parser.add_argument('--sigma-min', type=float, default=5e-3, help='Minimum sigma value')
    parser.add_argument('--sigma-max', type=float, default=5.0, help='Maximum sigma value')
    
    # Data paths
    parser.add_argument('--train-data', type=str, default="dataset/dit_data/train_dataset.parquet",
                      help='Path to training data')
    parser.add_argument('--val-data', type=str, default="dataset/dit_data/test_dataset.parquet",
                      help='Path to validation data')
    parser.add_argument('--checkpoint-dir', type=str, default=".", help='Directory to save checkpoints')
    
    # Wandb configuration
    parser.add_argument('--wandb-project', type=str, default='denoising-model', help='Weights & Biases project name')
    parser.add_argument('--wandb-entity', type=str, default=None, help='Weights & Biases entity name')
    parser.add_argument('--wandb-name', type=str, default=None, help='Weights & Biases run name')
    
    # Add checkpoint loading argument
    parser.add_argument('--resume-from', type=str, default=None, 
                       help='Path to checkpoint file to resume training from')
    
    args = parser.parse_args()
    
    # Validate arguments
    assert len(args.model_depths) == len(args.model_channels) == len(args.attn_depths), \
        "Model depths, channels, and attention depths must have the same length"
    
    return args

def main():
    args = parse_args()
    
    # Setup distributed training
    is_distributed, local_rank, world_size = setup_distributed()
    args.distributed = is_distributed
    args.local_rank = local_rank
    
    # Set device
    device = get_device(args)
    is_main_process = (not is_distributed) or (local_rank == 0)
    
    if is_main_process:
        print(f"Training on: {device}")
        if is_distributed:
            print(f"Distributed training with {world_size} GPUs")
            
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler()
    
    # Data loading
    train_df = pd.read_parquet(args.train_data)
    val_df = pd.read_parquet(args.val_data)

    train_dataset = SequenceMazeDataset(train_df)
    val_dataset = SequenceMazeDataset(val_df)
    
    # Setup samplers
    train_sampler = DistributedSampler(train_dataset) if is_distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_distributed else None
    
    # Adjust batch size for distributed training
    effective_batch_size = args.batch_size
    if is_distributed:
        effective_batch_size = args.batch_size // world_size
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=effective_batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_maze_sequences,
        num_workers=0 if device == "cpu" else args.num_workers,
        pin_memory=(device != "cpu")
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=effective_batch_size,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=collate_maze_sequences,
        num_workers=0 if device == "cpu" else 4,
        pin_memory=(device != "cpu")
    )

    # Model configuration
    conditioned_unet_cfg = ConditionedUNetConfig(
        img_channels=args.img_channels,
        num_conditioning_steps=args.num_conditioning_steps,
        cond_channels=args.cond_channels,
        depths=args.model_depths,
        channels=args.model_channels,
        attn_depths=args.attn_depths,
        num_actions=args.num_actions,
    )

    denoiser_cfg = DenoiserConfig(
        conditioned_unet=conditioned_unet_cfg,
        sigma_data=args.sigma_data,
        sigma_offset_noise=args.sigma_offset_noise,
        noise_previous_obs=args.noise_previous_obs,
    )

    sigma_distribution_cfg = SigmaDistributionConfig(
        loc=args.sigma_loc,
        scale=args.sigma_scale,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
    )
    
    # Model initialization
    denoiser = Denoiser(denoiser_cfg)
    denoiser.setup_training(sigma_distribution_cfg)
    denoiser = denoiser.to(device)
    
    if is_distributed:
        denoiser = DDP(denoiser, device_ids=[local_rank] if device != "cpu" else None)
    
    optimizer = torch.optim.AdamW(
        denoiser.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    
    scheduler = LambdaLR(
        optimizer, 
        lr_lambda=lambda step: get_lr_lambda(step, args.warmup_steps)
    )

    # Initialize tracking variables
    start_epoch = 0
    best_val_loss = float('inf')
    
    # Load checkpoint if specified
    if args.resume_from is not None and os.path.exists(args.resume_from):
        if is_main_process:
            print(f"Loading checkpoint from {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=device)
        
        # Load model state
        if is_distributed:
            denoiser.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            denoiser.load_state_dict(checkpoint['model_state_dict'])
            
        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load training state
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        
        if is_main_process:
            print(f"Resuming from epoch {start_epoch} with best val loss: {best_val_loss:.4f}")
    
    epoch_progress = tqdm(
        range(start_epoch, args.num_epochs),  # Start from loaded epoch
        desc="Training Progress", 
        disable=not is_main_process
    )
    
    # Initialize wandb if main process
    if is_main_process:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            config=vars(args)
        )
    
    global_step = 0
    log_every_n_steps = 100  
    
    try:
        for epoch in epoch_progress:
            if is_distributed:
                train_sampler.set_epoch(epoch)
            
            train_loss, global_step = train_one_epoch(
                denoiser, train_loader, optimizer, scheduler,
                device, args.gradient_accumulation_steps, args.max_grad_norm, 
                is_main_process, scaler, global_step, log_every_n_steps, val_loader
            )
            
            # Save checkpoint at the end of each epoch
            if is_main_process:
                checkpoint_path = f'checkpoint_epoch_{epoch}.pt'
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': denoiser.module.state_dict() if is_distributed 
                                      else denoiser.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                }, checkpoint_path)
                wandb.save(checkpoint_path)

    except KeyboardInterrupt:
        print("Training interrupted by user")
    
    finally:
        if is_distributed:
            torch.distributed.destroy_process_group()
        if is_main_process:
            wandb.finish()

if __name__ == "__main__":
    main()



#