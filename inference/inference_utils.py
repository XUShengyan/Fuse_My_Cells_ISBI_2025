import numpy as np
import torch
from typing import Union, Tuple, List
from scipy.ndimage import gaussian_filter
from functools import lru_cache
from inference.sliding_window import compute_steps_for_sliding_window

# Generate coordinates for sliding window with edge handling
def generate_coords(spatial_shape: Tuple[int, int, int],
                    tile_size: Tuple[int, int, int],
                    tile_step_size: float) -> list:
    """Generate coordinates for sliding window with intelligent edge handling."""
    steps = compute_steps_for_sliding_window(spatial_shape, tile_size, tile_step_size)
    coords = []

    for d in steps[0]:
        for h in steps[1]:
            for w in steps[2]:
                d_end = min(d + tile_size[0], spatial_shape[0])
                h_end = min(h + tile_size[1], spatial_shape[1])
                w_end = min(w + tile_size[2], spatial_shape[2])

                if (d_end - d) < tile_size[0]:
                    d = spatial_shape[0] - tile_size[0]
                if (h_end - h) < tile_size[1]:
                    h = spatial_shape[1] - tile_size[1]
                if (w_end - w) < tile_size[2]:
                    w = spatial_shape[2] - tile_size[2]

                coords.append((d, d + tile_size[0],
                               h, h + tile_size[1],
                               w, w + tile_size[2]))
    return coords

# Pad the image to the target size
def pad_to_target(image, target_size):
    """Pad the image to the target size if necessary."""
    # image shape: (C, H, W)
    D, H, W = image.shape
    target_D, target_H, target_W = target_size

    pad_D = max(target_D - D, 0)
    pad_H = max(target_H - H, 0)
    pad_W = max(target_W - W, 0)

    if pad_D == 0 and pad_H == 0 and pad_W == 0:
        return image, (0, 0, 0, 0, 0, 0)

    # 对每个需要填充的维度采用均匀填充策略
    pad_front = pad_D // 2
    pad_back = pad_D - pad_front
    pad_top = pad_H // 2
    pad_bottom = pad_H - pad_top
    pad_left = pad_W // 2
    pad_right = pad_W - pad_left

    padded = np.pad(image, ((pad_front, pad_back),
                            (pad_top, pad_bottom),
                            (pad_left, pad_right)), mode='constant')
    return padded, (pad_front, pad_back, pad_top, pad_bottom, pad_left, pad_right)

# Remove padding from the image
def remove_padding(padded: np.ndarray, pad_info: Tuple[int, int, int, int, int, int], original_size: Tuple[int, int, int]) -> np.ndarray:
    """Remove padding to restore the image to its original size."""
    pad_front, _, pad_top, _, pad_left, _ = pad_info
    orig_D, orig_H, orig_W = original_size
    return padded[pad_front:pad_front + orig_D,
                  pad_top:pad_top + orig_H,
                  pad_left:pad_left + orig_W]

# Compute a Gaussian importance map
@lru_cache(maxsize=2)
def compute_gaussian(tile_size: Union[Tuple[int, ...], List[int]], sigma_scale: float = 1. / 8,
                     value_scaling_factor: float = 1, dtype=torch.float16, device=torch.device('cuda', 0)) \
        -> torch.Tensor:
    """Compute a Gaussian importance map for the tile."""
    tmp = np.zeros(tile_size)
    center_coords = [i // 2 for i in tile_size]
    sigmas = [max(i * sigma_scale, 0.001) for i in tile_size]  # Avoid zero sigma
    tmp[tuple(center_coords)] = 1

    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)

    gaussian_importance_map = torch.from_numpy(gaussian_importance_map)
    gaussian_importance_map /= torch.max(gaussian_importance_map) + 1e-8
    gaussian_importance_map *= value_scaling_factor

    gaussian_importance_map = gaussian_importance_map.to(device=device, dtype=dtype)

    gaussian_importance_map[gaussian_importance_map == 0] = torch.min(
        gaussian_importance_map[gaussian_importance_map > 0]
    )
    return gaussian_importance_map

# Scale and pad the XY plane of the volume
def scale_xy_and_pad(volume, target_size=512):
    """Scale and pad the XY plane of the volume to the target size."""
    d, h, w = volume.shape
    processed = volume.copy()
    meta = {
        'original_shape': (h, w),
        'action': 'none',
        'scale_factor': 1.0,
        'padding': (0, 0, 0, 0),
        'crop': None
    }

    if h > target_size and w > target_size:
        meta['action'] = 'no_processing'
        return processed, meta

    if h > target_size or w > target_size:
        crop_h = h > target_size
        crop_w = w > target_size

        start_h = (h - target_size) // 2 if crop_h else 0
        start_w = (w - target_size) // 2 if crop_w else 0
        end_h = start_h + target_size if crop_h else h
        end_w = start_w + target_size if crop_w else w

        pad_h = target_size - (end_h - start_h) if crop_h else target_size - h
        pad_w = target_size - (end_w - start_w) if crop_w else target_size - w

        cropped = volume[:, start_h:end_h, start_w:end_w]

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        processed = np.pad(cropped,
                           ((0, 0),
                            (pad_top, pad_bottom),
                            (pad_left, pad_right)),
                           mode='constant')

        meta.update({
            'action': 'crop_and_pad',
            'crop': (start_h, end_h, start_w, end_w),
            'padding': (pad_top, pad_bottom, pad_left, pad_right)
        })
        return processed, meta

    scale = target_size / max(h, w)
    new_h = int(np.round(h * scale))
    new_w = int(np.round(w * scale))

    scaled = np.zeros((d, new_h, new_w), dtype=volume.dtype)
    for z in range(d):
        y_ratio = (h - 1) / new_h if new_h != 0 else 0
        x_ratio = (w - 1) / new_w if new_w != 0 else 0
        y_grid, x_grid = np.indices((new_h, new_w))
        y = y_grid * y_ratio
        x = x_grid * x_ratio

        y0 = np.floor(y).astype(int)
        x0 = np.floor(x).astype(int)
        y1 = np.clip(y0 + 1, 0, h - 1)
        x1 = np.clip(x0 + 1, 0, w - 1)

        wy = y - y0
        wx = x - x0

        scaled[z] = (
            (1 - wy) * (1 - wx) * volume[z, y0, x0] +
            (1 - wy) * wx * volume[z, y0, x1] +
            wy * (1 - wx) * volume[z, y1, x0] +
            wy * wx * volume[z, y1, x1]
        )

    pad_h = target_size - new_h
    pad_w = target_size - new_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    processed = np.pad(scaled,
                       ((0, 0),
                        (pad_top, pad_bottom),
                        (pad_left, pad_right)),
                       mode='constant')

    meta.update({
        'action': 'scale_and_pad',
        'scale_factor': scale,
        'padding': (pad_top, pad_bottom, pad_left, pad_right)
    })
    return processed, meta

# Restore the XY plane to original size
def unscale_xy_and_unpad(processed, meta):
    """Restore the XY plane of the processed volume to its original size."""
    original_h, original_w = meta['original_shape']
    restored = np.zeros((processed.shape[0], original_h, original_w), dtype=processed.dtype)

    if meta['action'] == 'no_processing':
        return processed.copy()

    if meta['action'] == 'crop_and_pad':
        pad_top, pad_bottom, pad_left, pad_right = meta['padding']
        cropped = processed[:,
                          pad_top:processed.shape[1] - pad_bottom,
                          pad_left:processed.shape[2] - pad_right]

        start_h, end_h, start_w, end_w = meta['crop']
        restored[:, start_h:end_h, start_w:end_w] = cropped
        return restored

    if meta['action'] == 'scale_and_pad':
        pad_top, pad_bottom, pad_left, pad_right = meta['padding']
        scaled = processed[:,
                         pad_top:processed.shape[1] - pad_bottom,
                         pad_left:processed.shape[2] - pad_right]

        scale = 1.0 / meta['scale_factor']
        new_h = int(np.round(original_h * scale))
        new_w = int(np.round(original_w * scale))

        for z in range(scaled.shape[0]):
            y_ratio = (scaled.shape[1] - 1) / original_h if original_h != 0 else 0
            x_ratio = (scaled.shape[2] - 1) / original_w if original_w != 0 else 0
            y_grid, x_grid = np.indices((original_h, original_w))
            y = y_grid * y_ratio
            x = x_grid * x_ratio

            y0 = np.floor(y).astype(int)
            x0 = np.floor(x).astype(int)
            y1 = np.clip(y0 + 1, 0, scaled.shape[1] - 1)
            x1 = np.clip(x0 + 1, 0, scaled.shape[2] - 1)

            wy = y - y0
            wx = x - x0

            restored[z] = (
                (1 - wy) * (1 - wx) * scaled[z, y0, x0] +
                (1 - wy) * wx * scaled[z, y0, x1] +
                wy * (1 - wx) * scaled[z, y1, x0] +
                wy * wx * scaled[z, y1, x1]
            )

        return restored

    raise ValueError("Unknown processing type")

