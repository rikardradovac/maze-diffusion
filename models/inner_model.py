import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Optional

from .types.common import InnerModelConfig
from .layers.fourier import FourierFeatures
from .layers.conv import Conv3x3
from .layers.norm import GroupNorm
from .unet import UNet

class InnerModel(nn.Module):
    def __init__(self, cfg: InnerModelConfig) -> None:
        super().__init__()
        self.noise_emb = FourierFeatures(cfg.cond_channels)
        self.noise_cond_emb = FourierFeatures(cfg.cond_channels)
        self.act_emb = nn.Sequential(
            nn.Embedding(
            cfg.num_actions,  # 4 for UP, DOWN, LEFT, RIGHT
            cfg.cond_channels // cfg.num_steps_conditioning
            ),
            nn.Flatten()
        )

        nn.init.orthogonal_(self.act_emb[0].weight, gain=1.0)

        self.cond_proj = nn.Sequential(
            nn.Linear(cfg.cond_channels, cfg.cond_channels),
            nn.SiLU(),
            nn.Linear(cfg.cond_channels, cfg.cond_channels),
        )
        self.conv_in = Conv3x3(
            (cfg.num_steps_conditioning + int(cfg.is_upsampler) + 1) * cfg.img_channels, 
            cfg.channels[0]
        )

        self.unet = UNet(cfg.cond_channels, cfg.depths, cfg.channels, cfg.attn_depths)

        self.norm_out = GroupNorm(cfg.channels[0])
        self.conv_out = Conv3x3(cfg.channels[0], cfg.img_channels)
        nn.init.zeros_(self.conv_out.weight)

    def forward(self, noisy_next_obs: Tensor, c_noise: Tensor, c_noise_cond: Tensor, 
                obs: Tensor, act: Optional[Tensor]) -> Tensor:
        if self.act_emb is not None and act is not None:
            # Convert one-hot to indices
            act_indices = torch.argmax(act, dim=-1)
            act_emb = self.act_emb(act_indices)
        else:
            assert act is None
            act_emb = 0

        cond = self.cond_proj(
            self.noise_emb(c_noise) + 
            self.noise_cond_emb(c_noise_cond) + 
            act_emb
        )
        x = self.conv_in(torch.cat((obs, noisy_next_obs), dim=1))
        x, _, _ = self.unet(x, cond)
        x = self.conv_out(F.silu(self.norm_out(x)))
        return x 