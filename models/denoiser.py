import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Optional

from .types.common import DenoiserConfig, SigmaDistributionConfig, Batch, Conditioners, LossAndLogs
from .inner_model import InnerModel

def add_dims(input: Tensor, n: int) -> Tensor:
    return input.reshape(input.shape + (1,) * (n - input.ndim))

class Denoiser(nn.Module):
    def __init__(self, cfg: DenoiserConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.inner_model = InnerModel(cfg.inner_model)
        self.sample_sigma_training = None

    @property
    def device(self) -> torch.device:
        return self.inner_model.noise_emb.weight.device

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
        return self.inner_model(rescaled_noise, cs.c_noise, cs.c_noise_cond, rescaled_obs)

    @torch.no_grad()
    def wrap_model_output(self, noisy_next_obs: Tensor, model_output: Tensor, cs: Conditioners) -> Tensor:
        d = cs.c_skip * noisy_next_obs + cs.c_out * model_output
        d = d.clamp(-1, 1).add(1).div(2).mul(255).byte().div(255).mul(2).sub(1)
        return d

    @torch.no_grad()
    def denoise(self, noisy_next_obs: Tensor, sigma: Tensor, sigma_cond: Optional[Tensor], 
                obs: Tensor, act: Optional[Tensor]) -> Tensor:
        cs = self.compute_conditioners(sigma, sigma_cond)
        model_output = self.compute_model_output(noisy_next_obs, obs, act, cs)
        denoised = self.wrap_model_output(noisy_next_obs, model_output, cs)
        return denoised

    def forward(self, batch: Batch) -> LossAndLogs:
        b, t, c, h, w = batch.obs.size()
        H, W = (self.cfg.upsampling_factor * h, self.cfg.upsampling_factor * w) if self.is_upsampler else (h, w)
        n = self.cfg.inner_model.num_steps_conditioning
        seq_length = t - n

        # Get mask for valid sequences
        sequence_mask = batch.mask if batch.mask is not None else torch.ones(b, dtype=torch.bool, device=self.device)

        if self.is_upsampler:
            all_obs = torch.stack([x["full_res"] for x in batch.info]).to(self.device)
            low_res = F.interpolate(
                batch.obs.reshape(b * t, c, h, w), 
                scale_factor=self.cfg.upsampling_factor, 
                mode="bicubic"
            ).reshape(b, t, c, H, W)
            assert all_obs.shape == low_res.shape
        else:
            all_obs = batch.obs.clone()

        loss = 0
        total_valid = 0
        
        for i in range(seq_length):
            # Only process valid sequences
            valid_mask = sequence_mask[:, n+i]
            if not valid_mask.any():
                continue
            
            # Index only valid sequences
            valid_obs = all_obs[valid_mask]
            valid_size = valid_mask.sum()

            prev_obs = valid_obs[:, i:i+n].reshape(valid_size, n * c, H, W)
            obs = valid_obs[:, n + i]

            if self.cfg.noise_previous_obs:
                sigma_cond = self.sample_sigma_training(valid_size, self.device)
                prev_obs = self.apply_noise(prev_obs, sigma_cond, self.cfg.sigma_offset_noise)
            else:
                sigma_cond = None

            if self.is_upsampler:
                valid_low_res = low_res[valid_mask]
                prev_obs = torch.cat((prev_obs, valid_low_res[:, n + i]), dim=1)

            sigma = self.sample_sigma_training(valid_size, self.device)
            noisy_obs = self.apply_noise(obs, sigma, self.cfg.sigma_offset_noise)
            
            cs = self.compute_conditioners(sigma, sigma_cond)
            model_output = self.compute_model_output(noisy_obs, prev_obs, cs)
            
            target = (obs - cs.c_skip * noisy_obs) / cs.c_out
            loss += F.mse_loss(model_output, target) * valid_size
            
            denoised = self.wrap_model_output(noisy_obs, model_output, cs)
            valid_obs[:, n + i] = denoised
            all_obs[valid_mask] = valid_obs
            
            total_valid += valid_size

        # Normalize loss by total number of valid sequences processed
        loss = loss / total_valid if total_valid > 0 else loss
        return loss, {"loss_denoising": loss.item()} 