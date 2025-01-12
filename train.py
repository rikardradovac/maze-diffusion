import os
from tqdm.auto import tqdm
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from models import InnerModelConfig, DenoiserConfig, SigmaDistributionConfig, Batch, Denoiser
from data import SequenceMazeDataset
import pandas as pd
# Initialize DDP
torch.distributed.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")


# Define warmup function
def get_lr_lambda(current_step: int, warmup_steps: int):
    if current_step < warmup_steps:
        return current_step / warmup_steps
    return 1.0

# Hyperparameters
num_epochs = 30
max_grad_norm = 5  
gradient_accumulation_steps = 8  
warmup_steps = 50

def train_one_epoch(model, train_loader, optimizer, scheduler, device, gradient_accumulation_steps, max_grad_norm, local_rank):
    model.train()
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(train_loader, desc=f"Training", disable=local_rank != 0)
    running_loss = 0.0
    
    for step, batch in enumerate(progress_bar):
        # Compute loss
        loss, _ = model(Batch(obs=batch["maze_sequence"], act=batch["actions"]).to(device))
        loss = loss / gradient_accumulation_steps
        loss.backward()

        running_loss += loss.item() * gradient_accumulation_steps
        
        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{running_loss/gradient_accumulation_steps:.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.6f}'
            })
            running_loss = 0.0
            
        total_loss += loss.item() * gradient_accumulation_steps
        num_batches += 1
    
    progress_bar.close()
    return total_loss / num_batches

def validate(model, val_loader, device):
    model.eval()
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(val_loader, desc=f"Validating", leave=False)
    
    with torch.no_grad():
        for batch in progress_bar:
            loss, _ = model(Batch(obs=batch["maze_sequence"], act=batch["actions"]).to(device))
            total_loss += loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({'val_loss': f'{total_loss/num_batches:.4f}'})
    
    progress_bar.close()
    return total_loss / num_batches

def main():
    # Data loading
    train_df = pd.read_parquet("dataset/dit_data/train_dataset.parquet")
    val_df = pd.read_parquet("dataset/dit_data/test_dataset.parquet")

    train_dataset = SequenceMazeDataset(train_df)
    val_dataset = SequenceMazeDataset(val_df)
    
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=8)
    val_loader = DataLoader(val_dataset, sampler=val_sampler, batch_size=8)




    inner_model_cfg = InnerModelConfig(
        img_channels=3,
        num_steps_conditioning=4,
        cond_channels=256,  # Increased
        depths=[2, 2, 2, 2],
        channels=[64, 64, 64, 64],
        attn_depths=[False, False, False, False], 
        num_actions=4,
        is_upsampler=False
    )

    denoiser_cfg = DenoiserConfig(
        inner_model=inner_model_cfg,
        sigma_data=0.5,
        sigma_offset_noise=0.3,
        noise_previous_obs=True,
        upsampling_factor=None
    )

    sigma_distribution_cfg = SigmaDistributionConfig(
        loc=-0.3,
        scale=1,
        sigma_min=5e-3,
        sigma_max=3.0,
    )
    # Model initialization
    denoiser = Denoiser(denoiser_cfg)
    denoiser.setup_training(sigma_distribution_cfg)
    denoiser = denoiser.to(device)
    denoiser = DDP(denoiser, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(denoiser.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: get_lr_lambda(step, warmup_steps))

    best_val_loss = float('inf')
    
    epoch_progress = tqdm(range(num_epochs), desc="Training Progress", disable=local_rank != 0)
    
    for epoch in epoch_progress:
        train_sampler.set_epoch(epoch)
        
        # Training phase
        train_loss = train_one_epoch(
            denoiser, train_loader, optimizer, scheduler,
            device, gradient_accumulation_steps, max_grad_norm, local_rank
        )
        
        # Validation phase
        val_loss = validate(denoiser, val_loader, device)
        
        if local_rank == 0:
            epoch_progress.set_postfix({
                'train_loss': f'{train_loss:.4f}',
                'val_loss': f'{val_loss:.4f}',
                'best_val_loss': f'{best_val_loss:.4f}'
            })
            
            # Save checkpoint if validation loss improved
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': denoiser.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, f'checkpoint_best.pt')

    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()



#