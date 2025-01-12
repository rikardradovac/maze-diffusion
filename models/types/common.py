from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import torch
from torch import Tensor

LossAndLogs = Tuple[Tensor, Dict[str, Any]]

@dataclass
class SegmentId:
    episode_id: int
    start: int
    stop: int

@dataclass
class Segment:
    obs: torch.FloatTensor
    act: torch.LongTensor
    rew: torch.FloatTensor
    end: torch.ByteTensor
    trunc: torch.ByteTensor
    mask_padding: torch.BoolTensor
    info: Dict[str, Any]
    id: SegmentId

    @property
    def effective_size(self):
        return self.mask_padding.sum().item()

@dataclass
class Batch:
    obs: torch.ByteTensor
    act: torch.LongTensor
    rew: Optional[torch.FloatTensor] = None
    end: Optional[torch.LongTensor] = None
    trunc: Optional[torch.LongTensor] = None
    info: Optional[List[Dict[str, Any]]] = None
    segment_ids: Optional[List[SegmentId]] = None

    def pin_memory(self) -> 'Batch':
        return Batch(
            obs=self.obs.pin_memory() if self.obs is not None else None,
            act=self.act.pin_memory() if self.act is not None else None,
            rew=self.rew.pin_memory() if self.rew is not None else None,
            end=self.end.pin_memory() if self.end is not None else None,
            trunc=self.trunc.pin_memory() if self.trunc is not None else None,
            info=self.info,
            segment_ids=self.segment_ids
        )

    def to(self, device: torch.device) -> 'Batch':
        return Batch(
            obs=self.obs.to(device) if self.obs is not None else None,
            act=self.act.to(device) if self.act is not None else None,
            rew=self.rew.to(device) if self.rew is not None else None,
            end=self.end.to(device) if self.end is not None else None,
            trunc=self.trunc.to(device) if self.trunc is not None else None,
            info=self.info,
            segment_ids=self.segment_ids
        )

@dataclass
class Conditioners:
    c_in: Tensor
    c_out: Tensor
    c_skip: Tensor
    c_noise: Tensor
    c_noise_cond: Tensor

@dataclass
class SigmaDistributionConfig:
    loc: float
    scale: float
    sigma_min: float
    sigma_max: float

@dataclass
class InnerModelConfig:
    img_channels: int
    num_steps_conditioning: int
    cond_channels: int
    depths: List[int]
    channels: List[int]
    attn_depths: List[bool]
    num_actions: Optional[int] = None
    is_upsampler: Optional[bool] = None

@dataclass
class DenoiserConfig:
    inner_model: InnerModelConfig
    sigma_data: float
    sigma_offset_noise: float
    noise_previous_obs: bool
    upsampling_factor: Optional[int] = None