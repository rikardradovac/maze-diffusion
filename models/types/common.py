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
    rew: Optional[torch.FloatTensor] = None
    end: Optional[torch.LongTensor] = None
    trunc: Optional[torch.LongTensor] = None
    info: Optional[List[Dict[str, Any]]] = None
    segment_ids: Optional[List[SegmentId]] = None
    mask: Optional[torch.BoolTensor] = None

    def __post_init__(self):
        # If mask is not provided, assume all elements are valid
        if self.mask is None and self.obs is not None:
            self.mask = torch.ones(self.obs.shape[0], dtype=torch.bool, 
                                 device=self.obs.device)

    def get_batch(self, indices: torch.Tensor) -> 'Batch':
        """
        Returns a new Batch containing only the specified indices.
        Handles masking appropriately.
        """
        return Batch(
            obs=self.obs[indices] if self.obs is not None else None,
            rew=self.rew[indices] if self.rew is not None else None,
            end=self.end[indices] if self.end is not None else None,
            trunc=self.trunc[indices] if self.trunc is not None else None,
            info=[self.info[i] for i in indices] if self.info is not None else None,
            segment_ids=[self.segment_ids[i] for i in indices] if self.segment_ids is not None else None,
            mask=self.mask[indices] if self.mask is not None else None
        )

    def filter_valid(self) -> 'Batch':
        """
        Returns a new Batch containing only the valid (unpadded) elements 
        according to the mask.
        """
        if self.mask is None:
            return self
        
        valid_indices = torch.where(self.mask)[0]
        return self.get_batch(valid_indices)

    def get_masked_elements(self, tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Applies the mask to a tensor if both exist.
        """
        if tensor is None or self.mask is None:
            return tensor
        return tensor[self.mask]

    def pin_memory(self) -> 'Batch':
        return Batch(
            obs=self.obs.pin_memory() if self.obs is not None else None,
            rew=self.rew.pin_memory() if self.rew is not None else None,
            end=self.end.pin_memory() if self.end is not None else None,
            mask=self.mask.pin_memory() if self.mask is not None else None,
            trunc=self.trunc.pin_memory() if self.trunc is not None else None,
            info=self.info,
            segment_ids=self.segment_ids
        )

    def to(self, device: torch.device) -> 'Batch':
        return Batch(
            obs=self.obs.to(device) if self.obs is not None else None,
            rew=self.rew.to(device) if self.rew is not None else None,
            end=self.end.to(device) if self.end is not None else None,
            mask=self.mask.to(device) if self.mask is not None else None,
            trunc=self.trunc.to(device) if self.trunc is not None else None,
            info=self.info,
            segment_ids=self.segment_ids
        )

    @property
    def batch_size(self) -> int:
        """
        Returns the effective batch size (number of valid elements).
        """
        if self.mask is not None:
            return self.mask.sum().item()
        return len(self.obs) if self.obs is not None else 0

    def split(self, batch_size: int) -> List['Batch']:
        """
        Splits the batch into smaller batches of specified size, 
        respecting the mask.
        """
        if self.mask is None:
            total_size = len(self.obs) if self.obs is not None else 0
            indices = torch.arange(total_size)
        else:
            indices = torch.where(self.mask)[0]
            
        splits = torch.split(indices, batch_size)
        return [self.get_batch(s) for s in splits]

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