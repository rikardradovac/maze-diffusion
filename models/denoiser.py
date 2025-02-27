import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Optional, Tuple

from .types.common import DenoiserConfig, SigmaDistributionConfig, Batch, Conditioners, LossAndLogs
from .conditioned_unet import ConditionedUNet

def add_dims(input: Tensor, n: int) -> Tensor:
    return input.reshape(input.shape + (1,) * (n - input.ndim))


def dice_loss(logits, target, smooth=1.0):
    """
    Compute the Dice loss, given logits and a binary target mask.
    
    Args:
        logits: Raw output from the network (before sigmoid), shape (B, 1, H, W)
        target: Binary ground truth mask, same shape as logits (values 0 or 1)
        smooth: Smoothing constant to avoid division by zero.
        
    Returns:
        Dice loss (1 - Dice coefficient), summed over batch
    """
    # Apply sigmoid to get probabilities
    probs = torch.sigmoid(logits)
    
    # Flatten the tensors
    probs_flat = probs.view(probs.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    
    intersection = (probs_flat * target_flat).sum(dim=1)
    union = probs_flat.sum(dim=1) + target_flat.sum(dim=1)
    
    dice_coeff = (2.0 * intersection + smooth) / (union + smooth)
    return (1 - dice_coeff).sum()

def combined_loss(logits, target, loss_fn, bce_weight=0.5, dice_weight=0.5):
    bce = loss_fn(logits, target)
    dice = dice_loss(logits, target)
    return bce_weight * bce + dice_weight * dice


class Denoiser(nn.Module):
    def __init__(self, cfg: DenoiserConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.conditioned_unet = ConditionedUNet(cfg.conditioned_unet)
        self.sample_sigma_training = None

    @property
    def device(self) -> torch.device:
        return self.conditioned_unet.noise_emb.weight.device

    def setup_training(self, cfg: SigmaDistributionConfig) -> None:
        assert self.sample_sigma_training is None

        def sample_sigma(n: int, device: torch.device):
            s = torch.randn(n, device=device) * cfg.scale + cfg.loc
            return s.exp().clip(cfg.sigma_min, cfg.sigma_max)

        self.sample_sigma_training = sample_sigma

    def apply_noise(self, x: Tensor, sigma: Tensor, sigma_offset_noise: float) -> Tensor:
        b, c, _, _ = x.shape
        offset_noise = sigma_offset_noise * torch.randn(b, c, 1, 1, device=self.device)
        return x + offset_noise + torch.randn_like(x) * add_dims(sigma, x.ndim)

    def compute_conditioners(self, sigma: Tensor, sigma_cond: Optional[Tensor]) -> Conditioners:
        sigma = (sigma**2 + self.cfg.sigma_offset_noise**2).sqrt()
        c_in = 1 / (sigma**2 + self.cfg.sigma_data**2).sqrt()
        c_skip = self.cfg.sigma_data**2 / (sigma**2 + self.cfg.sigma_data**2)
        c_out = sigma * c_skip.sqrt()
        c_noise = sigma.log() / 4
        c_noise_cond = sigma_cond.log() / 4 if sigma_cond is not None else torch.zeros_like(c_noise)
        return Conditioners(*(add_dims(c, n) for c, n in zip((c_in, c_out, c_skip, c_noise, c_noise_cond), (4, 4, 4, 1, 1))))

    def compute_model_output(self, noisy_next_obs: Tensor, obs: Tensor, cs: Conditioners) -> Tuple[Tensor, Tensor]:
        rescaled_obs = obs / self.cfg.sigma_data
        rescaled_noise = noisy_next_obs * cs.c_in
        return self.conditioned_unet(rescaled_noise, cs.c_noise, cs.c_noise_cond, rescaled_obs)

    @torch.no_grad()
    def wrap_model_output(self, noisy_next_obs: Tensor, model_output: Tensor, cs: Conditioners) -> Tensor:
        d = cs.c_skip * noisy_next_obs + cs.c_out * model_output
        d = d.clamp(-1, 1).add(1).div(2).mul(255).byte().div(255).mul(2).sub(1)
        return d

    @torch.no_grad()
    def denoise(self, noisy_next_obs: Tensor, sigma: Tensor, sigma_cond: Optional[Tensor], 
                obs: Tensor) -> Tensor:
        cs = self.compute_conditioners(sigma, sigma_cond)
        model_output, seg_logits = self.compute_model_output(noisy_next_obs, obs, cs)
        denoised = self.wrap_model_output(noisy_next_obs, model_output, cs)
        return denoised

    def forward(self, batch: Batch) -> LossAndLogs:
        b, t, c, H, W = batch.obs.size()
        n = self.cfg.conditioned_unet.num_conditioning_steps
        seq_length = t - n

        # Get mask for valid sequences
        sequence_mask = batch.mask if batch.mask is not None else torch.ones(b, dtype=torch.bool, device=self.device)
        
        # Get path mask if available
        path_mask = batch.path_mask if batch.path_mask is not None else None
        
        all_obs = batch.obs.clone()

        loss = 0
        path_loss = 0
        total_valid = 0
        total_path_pixels = 0
        
        
        pos_weight = torch.tensor(10.0).to(self.device) 
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='sum')
        
        for i in range(seq_length):
            # Only process valid sequences
            valid_mask = sequence_mask[:, n+i]
            if not valid_mask.any():
                continue
            # Index only valid sequences
            valid_obs = all_obs[valid_mask]  # Shape: [valid_size, t, c, H, W]
            valid_size = valid_mask.sum()
            
            # Get path mask for current timestep if available
            current_path_mask = None
            if path_mask is not None:
                # Fix: First select valid batch items, then select the timestep
                current_path_mask = path_mask[valid_mask, n+i]  # Shape: [valid_size, H, W]
                current_path_mask = current_path_mask.unsqueeze(1)

            # Get previous observations for conditioning
            prev_obs = valid_obs[:, i:i+n].reshape(valid_size, n * c, H, W)
            # Get current observation
            obs = valid_obs[:, n + i]  # Shape: [valid_size, c, H, W]

            if self.cfg.noise_previous_obs:
                sigma_cond = self.sample_sigma_training(valid_size, self.device)
                prev_obs = self.apply_noise(prev_obs, sigma_cond, self.cfg.sigma_offset_noise)
            else:
                sigma_cond = None

            sigma = self.sample_sigma_training(valid_size, self.device)
            noisy_obs = self.apply_noise(obs, sigma, self.cfg.sigma_offset_noise)
            
            cs = self.compute_conditioners(sigma, sigma_cond)
            model_output, seg_logits = self.compute_model_output(noisy_obs, prev_obs, cs)
            
            target = (obs - cs.c_skip * noisy_obs) / cs.c_out
            
            # Compute regular MSE loss
            mse_loss = F.mse_loss(model_output, target, reduction='sum')
            loss += mse_loss
            
            # Add path segmentation loss if path mask is available
            if current_path_mask is not None:
                path_seg_loss = combined_loss(seg_logits, current_path_mask, loss_fn)
                path_loss += path_seg_loss
                total_path_pixels += valid_size * H * W
            
            denoised = self.wrap_model_output(noisy_obs, model_output, cs)
            valid_obs[:, n + i] = denoised
            all_obs[valid_mask] = valid_obs
            
            total_valid += valid_size

        # Normalize losses
        loss = loss / (total_valid * c * H * W) if total_valid > 0 else loss
        path_loss = path_loss / total_path_pixels if total_path_pixels > 0 else path_loss
        total_loss = loss + path_loss


        return total_loss, {
            "loss_denoising": loss.item(),
            "loss_path": path_loss.item() if total_path_pixels > 0 else 0.0,
            "loss_total": total_loss.item()
        } 