from torch import nn, Tensor
from .conv import Conv1x1, Conv3x3
from .norm import GroupNorm, AdaGroupNorm
from .attention import SelfAttention2d
import torch.nn.functional as F
from typing import List, Optional
import torch

class SmallResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.f = nn.Sequential(
            GroupNorm(in_channels), 
            nn.SiLU(inplace=True), 
            Conv3x3(in_channels, out_channels)
        )
        self.skip_projection = nn.Identity() if in_channels == out_channels else Conv1x1(in_channels, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.skip_projection(x) + self.f(x)

class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_channels: int, attn: bool) -> None:
        super().__init__()
        should_proj = in_channels != out_channels
        self.proj = Conv1x1(in_channels, out_channels) if should_proj else nn.Identity()
        self.norm1 = AdaGroupNorm(in_channels, cond_channels)
        self.conv1 = Conv3x3(in_channels, out_channels)
        self.norm2 = AdaGroupNorm(out_channels, cond_channels)
        self.conv2 = Conv3x3(out_channels, out_channels)
        self.attn = SelfAttention2d(out_channels) if attn else nn.Identity()
        nn.init.zeros_(self.conv2.weight)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        r = self.proj(x)
        x = self.conv1(F.silu(self.norm1(x, cond)))
        x = self.conv2(F.silu(self.norm2(x, cond)))
        x = x + r
        x = self.attn(x)
        return x

class ResBlocks(nn.Module):
    def __init__(
        self,
        list_in_channels: List[int],
        list_out_channels: List[int],
        cond_channels: int,
        attn: bool,
    ) -> None:
        super().__init__()
        assert len(list_in_channels) == len(list_out_channels)
        self.in_channels = list_in_channels[0]
        self.resblocks = nn.ModuleList([
            ResBlock(in_ch, out_ch, cond_channels, attn)
            for (in_ch, out_ch) in zip(list_in_channels, list_out_channels)
        ])

    def forward(self, x: Tensor, cond: Tensor, to_cat: Optional[List[Tensor]] = None) -> Tensor:
        outputs = []
        for i, resblock in enumerate(self.resblocks):
            x = x if to_cat is None else torch.cat((x, to_cat[i]), dim=1)
            x = resblock(x, cond)
            outputs.append(x)
        return x, outputs 