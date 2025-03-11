import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Tuple

from .types.common import ConditionedUNetConfig
from .layers.fourier import FourierFeatures
from .layers.conv import Conv3x3
from .layers.norm import GroupNorm
from .unet import UNet

class ConditionedUNet(nn.Module):
    def __init__(self, cfg: ConditionedUNetConfig) -> None:
        super().__init__()
        self.noise_emb = FourierFeatures(cfg.cond_channels)
        self.noise_cond_emb = FourierFeatures(cfg.cond_channels)

        self.cond_proj = nn.Sequential(
            nn.Linear(cfg.cond_channels, cfg.cond_channels * 2),
            nn.SiLU(),
            nn.Linear(cfg.cond_channels * 2, cfg.cond_channels * 2),
            nn.SiLU(),
            nn.Linear(cfg.cond_channels * 2, cfg.cond_channels),
        )
        self.conv_in = Conv3x3(
            (cfg.num_conditioning_steps + 1) * cfg.img_channels, 
            cfg.channels[0]
        )

        self.unet = UNet(cfg.cond_channels, cfg.depths, cfg.channels, cfg.attn_depths)

        self.norm_out = GroupNorm(cfg.channels[0])
        self.conv_out = Conv3x3(cfg.channels[0], cfg.img_channels)
        nn.init.zeros_(self.conv_out.weight)

        # Auxiliary segmentation head for predicting the optimal path mask.
        # This head outputs a 1-channel (64x64) map.
        self.seg_head = nn.Conv2d(cfg.channels[0], 1, kernel_size=1)

    def forward(self, noisy_next_obs: Tensor, c_noise: Tensor, c_noise_cond: Tensor, 
                obs: Tensor) -> Tuple[Tensor, Tensor]:
        
        cond = self.cond_proj(
            self.noise_emb(c_noise) + 
            self.noise_cond_emb(c_noise_cond)
        )
        x = self.conv_in(torch.cat((obs, noisy_next_obs), dim=1))
        # Shared feature extraction through the UNet
        x, _, _ = self.unet(x, cond)
        
        # Auxiliary output: segmentation logits for the red optimal path.
        seg_logits = self.seg_head(x)  # Shape: (B, 1, 64, 64)
        
        # Main output: full RGB image reconstruction.
        x = self.conv_out(F.silu(self.norm_out(x)))
        
        # Return both outputs.
        return x, seg_logits
