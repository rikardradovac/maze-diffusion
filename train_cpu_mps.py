import os
import argparse
from tqdm.auto import tqdm
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from models import ConditionedUNetConfig, DenoiserConfig, SigmaDistributionConfig, Denoiser
from data import SequenceMazeDataset, collate_maze_sequences
import pandas as pd
import wandb

def get_lr_lambda(current_step: int, warmup_steps: int):
    if current_step < warmup_steps:
        return current_step / warmup_steps
    return 1.0

def get_device(args):
    """Get appropriate device based on arguments and availability."""
    if args.cpu:
        return "cpu"
    elif args.mps and torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda:0"
    else:
        return "cpu"

def train_one_epoch(model, train_loader, optimizer, scheduler, device, 
                   gradient_accumulation_steps, max_grad_norm, scaler,
                   global_step, log_every_n_steps, val_loader):
    model.train()
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(train_loader, desc=f"Training")
    running_loss = 0.0
    
    for step, batch in enumerate(progress_bar):
        batch = batch.to(device)
        
        # For MPS/CPU, we don't need autocast
        if device == "cuda:0":
            with torch.amp.autocast(device_type="cuda"):
                loss, _ = model(batch)
                loss = loss / gradient_accumulation_steps
            
            scaler.scale(loss).backward()
        else:
            loss, _ = model(batch)
            loss = loss / gradient_accumulation_steps
            loss.backward()
            
        running_loss += loss.item() * gradient_accumulation_steps
        
        if (step + 1) % gradient_accumulation_steps == 0:
            if device == "cuda:0":
                scaler.unscale_(optimizer)
                
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            if device == "cuda:0":
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
                
            optimizer.zero_grad()
            scheduler.step()
            
            global_step += 1
            
            # Log metrics every n steps
            if global_step % log_every_n_steps == 0:
                val_loss = validate(model, val_loader, device)
                
                metrics = {
                    'step': global_step,
                    'train_loss': running_loss/gradient_accumulation_steps,
                    'val_loss': val_loss,
                    'learning_rate': scheduler.get_last_lr()[0]
                }
                
                wandb.log(metrics)
                
                progress_bar.set_postfix({
                    'loss': f'{running_loss/gradient_accumulation_steps:.4f}',
                    'val_loss': f'{val_loss:.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.6f}'
                })
            else:
                progress_bar.set_postfix({
                    'loss': f'{running_loss/gradient_accumulation_steps:.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.6f}'
                })
                
            running_loss = 0.0
            
        total_loss += loss.item() * gradient_accumulation_steps
        num_batches += 1
    
    progress_bar.close()
    return total_loss / num_batches, global_step

def validate(model, val_loader, device):
    model.eval()
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(val_loader, desc=f"Validating", leave=False)
    
    with torch.no_grad():
        for batch in progress_bar:
            batch = batch.to(device)
            loss, _ = model(batch)
            total_loss += loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({'val_loss': f'{total_loss/num_batches:.4f}'})
    
    progress_bar.close()
    return total_loss / num_batches

def parse_args():
    parser = argparse.ArgumentParser(description='Train denoising model on CPU or MPS')
    
    # Device selection
    parser.add_argument('--cpu', action='store_true', help='Force CPU training')
    parser.add_argument('--mps', action='store_true', help='Use MPS (Metal Performance Shaders) for Mac')
    
    # Training infrastructure arguments
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=2, help='Number of data loading workers')
    
    # Training hyperparameters
    parser.add_argument('--num-epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-2, help='Weight decay')
    parser.add_argument('--max-grad-norm', type=float, default=5, help='Maximum gradient norm')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=8, help='Gradient accumulation steps')
    parser.add_argument('--warmup-steps', type=int, default=50, help='Learning rate warmup steps')
    
    # Model configuration
    parser.add_argument('--img-channels', type=int, default=3, help='Number of image channels')
    parser.add_argument('--num-conditioning-steps', type=int, default=3, help='Number of conditioning steps')
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
    parser.add_argument('--train-data', type=str, default="dataset/dit_data/train_dataset2.parquet",
                      help='Path to training data')
    parser.add_argument('--val-data', type=str, default="dataset/dit_data/test_dataset2.parquet",
                      help='Path to validation data')
    parser.add_argument('--checkpoint-dir', type=str, default=".", help='Directory to save checkpoints')
    
    # Wandb configuration
    parser.add_argument('--wandb-project', type=str, default='denoising-model', help='Weights & Biases project name')
    parser.add_argument('--wandb-entity', type=str, default=None, help='Weights & Biases entity name')
    parser.add_argument('--wandb-name', type=str, default=None, help='Weights & Biases run name')
    parser.add_argument('--use-wandb', action='store_true', help='Whether to use Weights & Biases logging')
    
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
    
    # Set device
    device = get_device(args)
    print(f"Training on: {device}")
    
    # Initialize gradient scaler for mixed precision training (only used for CUDA)
    scaler = torch.cuda.amp.GradScaler() if device == "cuda:0" else None
    
    # Data loading
    train_df = pd.read_parquet(args.train_data)
    val_df = pd.read_parquet(args.val_data)

    train_dataset = SequenceMazeDataset(train_df)
    val_dataset = SequenceMazeDataset(val_df)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_maze_sequences,
        num_workers=0 if device == "cpu" else args.num_workers,
        pin_memory=(device != "cpu")
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_maze_sequences,
        num_workers=0 if device == "cpu" else args.num_workers,
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
        print(f"Loading checkpoint from {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=device)
        
        # Load model state
        denoiser.load_state_dict(checkpoint['model_state_dict'])
            
        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load training state
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        
        print(f"Resuming from epoch {start_epoch} with best val loss: {best_val_loss:.4f}")
    
    epoch_progress = tqdm(
        range(start_epoch, args.num_epochs),  # Start from loaded epoch
        desc="Training Progress"
    )
    
    # Initialize wandb if requested
    if args.use_wandb:
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
            train_loss, global_step = train_one_epoch(
                denoiser, train_loader, optimizer, scheduler,
                device, args.gradient_accumulation_steps, args.max_grad_norm, 
                scaler, global_step, log_every_n_steps, val_loader
            )
            
            # Save checkpoint at the end of each epoch
            checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': denoiser.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
            }, checkpoint_path)
            
            if args.use_wandb:
                wandb.save(checkpoint_path)

    except KeyboardInterrupt:
        print("Training interrupted by user")
    
    finally:
        if args.use_wandb:
            wandb.finish()

if __name__ == "__main__":
    main() 