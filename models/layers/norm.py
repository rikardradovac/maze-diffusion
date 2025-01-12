import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor

GN_GROUP_SIZE = 32
GN_EPS = 1e-5

class GroupNorm(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        num_groups = max(1, in_channels // GN_GROUP_SIZE)
        self.norm = nn.GroupNorm(num_groups, in_channels, eps=GN_EPS)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x)

class AdaGroupNorm(nn.Module):
    def __init__(self, in_channels: int, cond_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_groups = max(1, in_channels // GN_GROUP_SIZE)
        self.linear = nn.Linear(cond_channels, in_channels * 2)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        assert x.size(1) == self.in_channels
        x = F.group_norm(x, self.num_groups, eps=GN_EPS)
        scale, shift = self.linear(cond)[:, :, None, None].chunk(2, dim=1)
        return x * (1 + scale) + shift