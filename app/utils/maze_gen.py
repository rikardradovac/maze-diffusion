from dataclasses import dataclass
import numpy as np
import onnxruntime as ort
from typing import List

@dataclass
class DiffusionSamplerConfig:
    num_steps_denoising: int = 3
    sigma_min: float = 2e-3
    sigma_max: float = 5
    rho: int = 7
    s_cond: float = 0.2

class MazeGeneratorONNX:
    def __init__(self, onnx_model_path: str):
        # Initialize ONNX Runtime session
        self.ort_session = ort.InferenceSession(onnx_model_path)
        
    def _build_sigmas(self, num_steps: int, sigma_min: float, sigma_max: float, rho: int) -> np.ndarray:
        """Build a sequence of noise levels."""
        ramp = np.linspace(0, 1, num_steps)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return sigmas
    
    def generate_sequence(self, initial_frames: np.ndarray, num_frames: int = 40) -> List[np.ndarray]:
        """Generate a sequence of maze frames given initial context."""
        obs_buffer = initial_frames.copy()  # Replace clone() with copy()
        generated_frames = []

        for _ in range(num_frames):
            # Generate next frame
            next_frame = self._generate_frame(obs_buffer)
            generated_frames.append(next_frame)  # No need for .cpu()

            # Update observation buffer
            obs_buffer = np.roll(obs_buffer, -1, axis=1)  # Replace dims with axis
            obs_buffer[:, -1] = next_frame

        return generated_frames
    
    def _generate_frame(self, context_frames: np.ndarray) -> np.ndarray:
        """Generate a single frame using the ONNX model for denoising."""
        config = DiffusionSamplerConfig()
        b, t, c, h, w = context_frames.shape  # Replace size() with shape

        # Reshape context frames or create empty context if t=0
        if t > 0:
            context_frames_reshaped = context_frames.reshape(b, t * c, h, w)
        else:
            context_frames_reshaped = np.zeros((b, 0, h, w), dtype=np.float32)  # Replace torch.zeros

        sigmas = self._build_sigmas(
            config.num_steps_denoising,
            config.sigma_min,
            config.sigma_max,
            config.rho
        )

        # No need for torch.no_grad() context
        x = np.random.randn(b, c, h, w).astype(np.float32)  # Replace torch.randn

        for sigma, next_sigma in zip(sigmas[:-1], sigmas[1:]):
            # Prepare inputs for ONNX Runtime - no need for .cpu().numpy()
            noisy_next_obs_np = x
            sigma_np = np.array([sigma], dtype=np.float32)  # No need for .item()
            sigma_cond_np = np.array([config.s_cond], dtype=np.float32) if t > 0 else np.array([0.0], dtype=np.float32)
            context_np = context_frames_reshaped  # No need for .cpu().numpy()

            ort_inputs = {
                'noisy_next_obs': noisy_next_obs_np,
                'sigma': sigma_np,
                'sigma_cond': sigma_cond_np,
                'obs': context_np
            }

            # Run inference with ONNX Runtime
            ort_outputs = self.ort_session.run(None, ort_inputs)
            denoised_np = ort_outputs[0]  # No need for torch.from_numpy().to(device)

            # Euler step
            d = (x - denoised_np) / sigma
            dt = next_sigma - sigma
            x = x + d * dt

        return x[0]  # Return the first item from batch