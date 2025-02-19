import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Optional

from .types.common import DenoiserConfig, SigmaDistributionConfig, Batch, Conditioners, LossAndLogs
from .conditioned_unet import ConditionedUNet

def add_dims(input: Tensor, n: int) -> Tensor:
    return input.reshape(input.shape + (1,) * (n - input.ndim))

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

    def compute_model_output(self, noisy_next_obs: Tensor, obs: Tensor, cs: Conditioners) -> Tensor:
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
        model_output = self.compute_model_output(noisy_next_obs, obs, cs)
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
        
        for i in range(seq_length):
            # Only process valid sequences
            valid_mask = sequence_mask[:, n+i]
            if not valid_mask.any():
                continue
            
            # Index only valid sequences
            valid_obs = all_obs[valid_mask]
            valid_size = valid_mask.sum()
            
            # Get path mask for current timestep if available
            current_path_mask = None
            if path_mask is not None:
                current_path_mask = path_mask[valid_mask, n+i]  # Shape: [valid_size, H, W]
                current_path_mask = current_path_mask.unsqueeze(1).expand(-1, c, -1, -1)  # Shape: [valid_size, c, H, W]

            prev_obs = valid_obs[:, i:i+n].reshape(valid_size, n * c, H, W)
            obs = valid_obs[:, n + i]

            if self.cfg.noise_previous_obs:
                sigma_cond = self.sample_sigma_training(valid_size, self.device)
                prev_obs = self.apply_noise(prev_obs, sigma_cond, self.cfg.sigma_offset_noise)
            else:
                sigma_cond = None

            sigma = self.sample_sigma_training(valid_size, self.device)
            noisy_obs = self.apply_noise(obs, sigma, self.cfg.sigma_offset_noise)
            
            cs = self.compute_conditioners(sigma, sigma_cond)
            model_output = self.compute_model_output(noisy_obs, prev_obs, cs)
            
            target = (obs - cs.c_skip * noisy_obs) / cs.c_out
            
            # Compute regular MSE loss
            mse_loss = F.mse_loss(model_output, target, reduction='none')
            
            if current_path_mask is not None:
                # Compute path-weighted loss
                path_pixels = current_path_mask.sum()
                if path_pixels > 0:
                    path_weighted_loss = (mse_loss * current_path_mask).sum() / path_pixels
                    path_loss += path_weighted_loss * valid_size
                    total_path_pixels += path_pixels * valid_size
                
                # For non-path pixels
                non_path_mask = ~current_path_mask
                non_path_pixels = non_path_mask.sum()
                if non_path_pixels > 0:
                    non_path_loss = (mse_loss * non_path_mask).sum() / non_path_pixels
                    loss += non_path_loss * valid_size
            else:
                # If no path mask, use regular MSE loss
                loss += mse_loss.mean() * valid_size
            
            denoised = self.wrap_model_output(noisy_obs, model_output, cs)
            valid_obs[:, n + i] = denoised
            all_obs[valid_mask] = valid_obs
            
            total_valid += valid_size

        # Normalize losses
        loss = loss / total_valid if total_valid > 0 else loss
        if total_path_pixels > 0:
            path_loss = path_loss / total_valid
            # Increase path loss weight significantly for full path learning
            total_loss = loss + 5.0 * path_loss 
        else:
            total_loss = loss

        return total_loss, {
            "loss_denoising": total_loss.item(),
            "loss_path": path_loss.item() if total_path_pixels > 0 else 0.0,
            "loss_regular": loss.item()
        } 