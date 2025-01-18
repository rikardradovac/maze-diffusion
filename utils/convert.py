import torch
import torch.nn as nn
import torch.onnx
from maze_gen import MazeGeneratorPyTorch, DiffusionSamplerConfig

class DenoiserWrapper(nn.Module):
    def __init__(self, denoiser):
        super().__init__()
        self.denoiser = denoiser

    def forward(self, noisy_next_obs, sigma, sigma_cond, obs):
        return self.denoiser.denoise(noisy_next_obs, sigma, sigma_cond, obs)

def convert_to_onnx(checkpoint_path: str, onnx_model_path: str):
    """Converts a PyTorch checkpoint to an ONNX model."""

    # 1. Instantiate PyTorch generator and denoiser (move to CPU for simplicity)
    generator_pytorch = MazeGeneratorPyTorch(checkpoint_path=checkpoint_path, device='cpu')
    denoiser = generator_pytorch.denoiser
    denoiser.eval()
    wrapper_model = DenoiserWrapper(denoiser).to('cpu')

    # 2. Prepare dummy inputs for ONNX
    noisy_next_obs = torch.randn(1, 3, 32, 32).to('cpu')  # Example: Single frame
    sigma = torch.tensor([0.5]).float().to('cpu')  # Example: Representative sigma
    sigma_cond = torch.full((1,), DiffusionSamplerConfig().s_cond).to('cpu') # Example: s_cond
    obs = torch.randn(1, 2 * 3, 32, 32).to('cpu')  # Example: Two frames as context

    # 3. Export the ONNX model
    torch.onnx.export(wrapper_model,
                      (noisy_next_obs, sigma, sigma_cond, obs),
                      onnx_model_path,
                      export_params=True,
                      opset_version=13,
                      do_constant_folding=True,
                      input_names=['noisy_next_obs', 'sigma', 'sigma_cond', 'obs'],
                      output_names=['denoised'],
                      dynamic_axes={'noisy_next_obs': {0: 'batch_size'},
                                    'obs': {0: 'batch_size', 1: 'num_context_frames'}})

    print(f"Denoiser exported to ONNX successfully at {onnx_model_path}!")

if __name__ == "__main__":
    CHECKPOINT_PATH = "checkpoint_best.pt"  # Replace with your checkpoint path
    ONNX_MODEL_PATH = "denoiser_denoise.onnx"
    convert_to_onnx(CHECKPOINT_PATH, ONNX_MODEL_PATH)