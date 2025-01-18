


import torch
from typing import List
from dataclasses import dataclass
import matplotlib.pyplot as plt
from IPython.display import clear_output
from models import Denoiser, DenoiserConfig, InnerModelConfig
import numpy as np
import onnxruntime


@dataclass
class DiffusionSamplerConfig:
    num_steps_denoising: int = 3
    sigma_min: float = 2e-3
    sigma_max: float = 5
    rho: int = 7
    s_cond: float = 0.2

class BaseMazeGenerator:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.action_map = {'UP': 0, 'DOWN': 1, 'LEFT': 2, 'RIGHT': 3}

    def _build_sigmas(self, num_steps: int, sigma_min: float, sigma_max: float, rho: int) -> torch.Tensor:
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        l = torch.linspace(0, 1, num_steps, device=self.device)
        sigmas = (max_inv_rho + l * (min_inv_rho - max_inv_rho)) ** rho
        return torch.cat((sigmas, sigmas.new_zeros(1)))

    def generate_sequence(self, initial_frames: torch.Tensor, num_frames: int = 40) -> List[torch.Tensor]:
        """Generate a sequence of maze frames given initial context."""
        obs_buffer = initial_frames.clone().to(self.device)
        generated_frames = []

        for _ in range(num_frames):
            # Generate next frame
            next_frame = self._generate_frame(obs_buffer)
            generated_frames.append(next_frame.cpu())

            # Update observation buffer
            obs_buffer = obs_buffer.roll(-1, dims=1)
            obs_buffer[:, -1] = next_frame

        return generated_frames

    @staticmethod
    def play_movie(frames: list[torch.Tensor], interval: float = 0.1, cmap="gray"):
        if not frames:
            raise ValueError("The list of frames is empty!")

        for i, frame in enumerate(frames):
            frame_data = frame[0].squeeze().cpu().numpy()

            if frame_data.ndim == 3 and frame_data.shape[0] == 3:
                frame_data = frame_data.transpose(1, 2, 0)

            frame_data = (frame_data - frame_data.min()) / (frame_data.max() - frame_data.min())

            clear_output(wait=True)

            plt.figure(figsize=(4, 4))
            plt.imshow(frame_data, cmap=cmap if frame_data.ndim == 2 else None)
            plt.title(f"Frame {i}")
            plt.axis("off")
            plt.show()

            plt.pause(interval)

    def _generate_frame(self, context_frames: torch.Tensor) -> torch.Tensor:
        """Generate a single frame (to be implemented by subclasses)."""
        raise NotImplementedError
    
    
    
    

class MazeGeneratorPyTorch(BaseMazeGenerator):
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        super().__init__(device)
        self.denoiser = self._initialize_model(checkpoint_path)

    def _initialize_model(self, checkpoint_path: str) -> Denoiser:
        """Initialize the denoiser model with configurations."""
        inner_model_cfg = InnerModelConfig(
            img_channels=3,
            num_steps_conditioning=2,
            cond_channels=256,
            depths=[2, 2, 2, 2],
            channels=[64, 64, 64, 64],
            attn_depths=[False, False, False, False],
            num_actions=4,
            is_upsampler=False
        )

        denoiser_cfg = DenoiserConfig(
            inner_model=inner_model_cfg,
            sigma_data=0.5,
            sigma_offset_noise=0.3,
            noise_previous_obs=True,
            upsampling_factor=None
        )

        denoiser = Denoiser(denoiser_cfg)
        if checkpoint_path:
            weights = torch.load(checkpoint_path, map_location=self.device)["model_state_dict"]
            weights = {k.strip("module."): v for k, v in weights.items()}
            denoiser.load_state_dict(weights)
        return denoiser.to(self.device)

    def _generate_frame(self, context_frames: torch.Tensor) -> torch.Tensor:
        """Generate a single frame using PyTorch denoiser."""
        config = DiffusionSamplerConfig()
        b, t, c, h, w = context_frames.size()

        if t > 0:
            context_frames = context_frames.reshape(b, t * c, h, w)
        else:
            context_frames = torch.zeros((b, 0, h, w), device=self.device)

        sigmas = self._build_sigmas(
            config.num_steps_denoising,
            config.sigma_min,
            config.sigma_max,
            config.rho
        )

        with torch.no_grad():
            x = torch.randn(b, c, h, w, device=self.device)

            for sigma, next_sigma in zip(sigmas[:-1], sigmas[1:]):
                # Conditioning
                if config.s_cond > 0 and t > 0:
                    sigma_cond = torch.full((b,), fill_value=config.s_cond, device=self.device)
                    context = self.denoiser.apply_noise(context_frames, sigma_cond, sigma_offset_noise=0)
                else:
                    sigma_cond = None
                    context = context_frames

                # Denoise using the denoise method
                denoised = self.denoiser.denoise(x, sigma, sigma_cond, context)

                # Euler step
                d = (x - denoised) / sigma
                dt = next_sigma - sigma
                x = x + d * dt

        return x
    
    
    
    
class MazeGeneratorONNX(BaseMazeGenerator):
    def __init__(self, onnx_model_path: str, device: str = "cuda"):
        super().__init__(device)
        self.ort_session = onnxruntime.InferenceSession(onnx_model_path)
        print("ONNX model inputs:", [input.name for input in self.ort_session.get_inputs()])

    def _generate_frame(self, context_frames: torch.Tensor) -> torch.Tensor:
        """Generate a single frame using the ONNX model for denoising."""
        config = DiffusionSamplerConfig()
        device = context_frames.device
        b, t, c, h, w = context_frames.size()

        # Reshape context frames or create empty context if t=0
        if t > 0:
            context_frames = context_frames.reshape(b, t * c, h, w)
        else:
            context_frames = torch.zeros((b, 0, h, w), device=device)

        sigmas = self._build_sigmas(
            config.num_steps_denoising,
            config.sigma_min,
            config.sigma_max,
            config.rho
        )

        with torch.no_grad():
            x = torch.randn(b, c, h, w, device=device)

            for sigma, next_sigma in zip(sigmas[:-1], sigmas[1:]):
                # Prepare inputs for ONNX Runtime
                noisy_next_obs_np = x.cpu().numpy()
                sigma_np = np.array([sigma.item()], dtype=np.float32)
                sigma_cond_np = np.array([config.s_cond], dtype=np.float32) if t > 0 else np.array([0.0], dtype=np.float32)
                context_np = context_frames.cpu().numpy()

                ort_inputs = {
                    'noisy_next_obs': noisy_next_obs_np,
                    'sigma': sigma_np,
                    'sigma_cond': sigma_cond_np,
                    'obs': context_np
                }

                # Run inference with ONNX Runtime
                ort_outputs = self.ort_session.run(None, ort_inputs)
                denoised_np = ort_outputs[0]  # Assuming the first output is 'denoised'
                denoised = torch.from_numpy(denoised_np).to(device)

                # Euler step
                d = (x - denoised) / sigma
                dt = next_sigma - sigma
                x = x + d * dt

        return x