# inference_utils.py
import numpy as np
from typing import Tuple
from inference.inference_utils import generate_coords
from utils.kmean_seg import kmeans_segmentation_pipeline


def sliding_window_inference_kmean(image: np.ndarray,
                                   tile_size: Tuple[int, int, int],
                                   tile_step_size: float):
    """
    Compute a foreground ratio map using sliding window inference with KMeans segmentation.

    Args:
        image (np.ndarray): Input image.
        tile_size (Tuple[int, int, int]): Size of the sliding window.
        tile_step_size (float): Step size for the sliding window.

    Returns:
        np.ndarray: Foreground ratio map.
    """
    # Perform KMeans segmentation to generate mask
    mask = kmeans_segmentation_pipeline(image, n_clusters=2, min_volume=10)

    # Generate sliding coordinates with edge handling
    coords = generate_coords(image.shape, tile_size, tile_step_size)

    # Initialize accumulator and weight map
    accumulator = np.zeros(image.shape, dtype=np.float32)
    weight_map = np.zeros(image.shape, dtype=np.float32)

    # Process each sliding window
    for i, (d_s, d_e, h_s, h_e, w_s, w_e) in enumerate(coords):
        # Ensure start indices are non-negative
        d_s = max(0, d_s)
        h_s = max(0, h_s)
        w_s = max(0, w_s)

        # Extract patch from mask
        patch_mask = mask[d_s:d_e, h_s:h_e, w_s:w_e]

        # Compute max projection along depth axis
        max_proj = patch_mask.max(axis=0)

        # Calculate foreground ratio (assuming foreground is 1)
        foreground_ratio = max_proj.mean()

        # Accumulate foreground ratio
        accumulator[d_s:d_e, h_s:h_e, w_s:w_e] += foreground_ratio
        weight_map[d_s:d_e, h_s:h_e, w_s:w_e] += 1.0

    # Compute final foreground ratio map
    foreground_ratio_map = accumulator / weight_map

    return foreground_ratio_map