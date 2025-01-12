import torch.nn.functional as F
from torch import nn, Tensor
from .conv import Conv3x3

class Downsample(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)
        nn.init.orthogonal_(self.conv.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = Conv3x3(in_channels, in_channels)

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x) 