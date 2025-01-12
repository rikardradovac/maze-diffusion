import math
import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
from .conv import Conv1x1
from .norm import GroupNorm

ATTN_HEAD_DIM = 8

class SelfAttention2d(nn.Module):
    def __init__(self, in_channels: int, head_dim: int = ATTN_HEAD_DIM) -> None:
        super().__init__()
        self.n_head = max(1, in_channels // head_dim)
        assert in_channels % self.n_head == 0
        self.norm = GroupNorm(in_channels)
        self.qkv_proj = Conv1x1(in_channels, in_channels * 3)
        self.out_proj = Conv1x1(in_channels, in_channels)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: Tensor) -> Tensor:
        n, c, h, w = x.shape
        x = self.norm(x)
        qkv = self.qkv_proj(x)
        qkv = qkv.view(n, self.n_head * 3, c // self.n_head, h * w).transpose(2, 3).contiguous()
        q, k, v = [x for x in qkv.chunk(3, dim=1)]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(2, 3).reshape(n, c, h, w)
        return x + self.out_proj(y) 