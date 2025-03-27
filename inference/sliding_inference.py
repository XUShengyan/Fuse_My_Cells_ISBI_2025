import numpy as np
import torch
from typing import Tuple
from inference.inference_utils import pad_to_target, generate_coords, compute_gaussian
import tqdm as tqdm

# Perform sliding window inference for image denoising
def sliding_window_inference(image: np.ndarray,
                             model: torch.nn.Module,
                             tile_size: Tuple[int, int, int],
                             tile_step_size: float,
                             device: torch.device,
                             batch_size: int = 1,
                             use_amp: bool = False) -> torch.Tensor:
    torch.cuda.empty_cache()

    auto_dtype = torch.float if not use_amp else torch.float16

    # Normalize input image
    image_tensor = torch.as_tensor(image, dtype=auto_dtype, device=device)
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)  # Add channel dimension

    C, D, H, W = image_tensor.shape
    spatial_shape = (D, H, W)

    # Generate sliding coordinates with edge handling
    coords = generate_coords(spatial_shape, tile_size, tile_step_size)

    with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=use_amp):
        accumulator = torch.zeros((C, D, H, W), device='cuda', dtype=auto_dtype)
        weight_map = torch.zeros((C, D, H, W), device='cuda', dtype=auto_dtype)

        model.eval()

        for batch_idx in tqdm.trange(0, len(coords), batch_size, desc='Sliding Window'):
            batch_coords = coords[batch_idx:batch_idx + batch_size]

            # Pipeline data loading
            batch_tensor = torch.stack([image_tensor[..., c[0]:c[1], c[2]:c[3], c[4]:c[5]] for c in batch_coords], dim=0)

            preds = model(batch_tensor)

            # Parallel accumulation
            for i, (d_s, d_e, h_s, h_e, w_s, w_e) in enumerate(batch_coords):
                accumulator[..., d_s:d_e, h_s:h_e, w_s:w_e] += preds[i]
                weight_map[..., d_s:d_e, h_s:h_e, w_s:w_e] += 1.0

    # Safe fusion strategy
    final_pred = accumulator / torch.clamp(weight_map, min=1e-7)
    return final_pred.squeeze(0).squeeze(0)  # Remove channel and batch dimensions

# Fast sliding window inference for efficiency

# Sliding window inference for SSIM prediction
def sliding_window_inference_ssim_predict(image: np.ndarray,
                                          model: torch.nn.Module,
                                          tile_size: Tuple[int, int, int],
                                          tile_step_size: float,
                                          device: torch.device,
                                          batch_size: int = 1,
                                          use_amp: bool = False) -> torch.Tensor:
    torch.cuda.empty_cache()

    auto_dtype = torch.float if not use_amp else torch.float16

    # Normalize input image
    image_tensor = torch.as_tensor(image, dtype=auto_dtype, device=device)
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)  # Add channel dimension

    C, D, H, W = image_tensor.shape
    spatial_shape = (D, H, W)

    coords = generate_coords(spatial_shape, tile_size, tile_step_size)

    with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=use_amp):
        accumulator = torch.zeros((C, D, H, W), device='cuda', dtype=auto_dtype)
        weight_map = torch.zeros((C, D, H, W), device='cuda', dtype=auto_dtype)

        model.eval()

        for batch_idx in tqdm.trange(0, len(coords), batch_size, desc='Sliding Window'):
            batch_coords = coords[batch_idx:batch_idx + batch_size]

            # Pipeline data loading
            batch_tensor = torch.stack([image_tensor[..., c[0]:c[1], c[2]:c[3], c[4]:c[5]] for c in batch_coords], dim=0)

            preds = model(batch_tensor)

            for i, (d_s, d_e, h_s, h_e, w_s, w_e) in enumerate(batch_coords):
                accumulator[..., d_s:d_e, h_s:h_e, w_s:w_e] += preds[i]
                weight_map[..., d_s:d_e, h_s:h_e, w_s:w_e] += 1.0

    # Safe fusion strategy
    final_pred = accumulator / torch.clamp(weight_map, min=1e-7)
    return final_pred.squeeze(0).squeeze(0)  # Remove channel and batch dimensions

