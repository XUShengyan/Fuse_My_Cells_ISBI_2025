# sliding_window.py
import numpy as np
from typing import Tuple, List

# Compute starting coordinates for sliding windows
def compute_steps_for_sliding_window(image_size: Tuple[int, ...],
                                     tile_size: Tuple[int, ...],
                                     tile_step_size: float) -> List[List[int]]:
    """
    Compute starting coordinates for sliding windows in each dimension.

    Args:
        image_size (Tuple[int, ...]): Image dimensions, e.g., (H, W).
        tile_size (Tuple[int, ...]): Patch size, e.g., (256, 256).
        tile_step_size (float): Relative step size (0 < tile_step_size <= 1).

    Returns:
        List[List[int]]: Starting coordinates for each dimension.
    """
    # Ensure image size is larger than or equal to tile size
    # assert all(i >= j for i, j in zip(image_size, tile_size)), "Image size must be larger than or equal to tile size."
    # Ensure step size is within (0, 1]
    # assert 0 < tile_step_size <= 1, "tile_step_size must be in (0, 1]."

    # Calculate target step sizes in pixels
    target_step_sizes = [int(ts * tile_step_size) for ts in tile_size]
    # Calculate number of steps for each dimension
    num_steps = [int(np.ceil((img - tile) / step)) + 1 for img, step, tile in
                 zip(image_size, target_step_sizes, tile_size)]

    steps = []
    for dim in range(len(tile_size)):
        max_start = image_size[dim] - tile_size[dim]
        if num_steps[dim] > 1:
            actual_step = max_start / (num_steps[dim] - 1)
        else:
            actual_step = 0  # Single step starts at 0
        steps_dim = [int(np.round(actual_step * i)) for i in range(num_steps[dim])]
        steps.append(steps_dim)

    return steps

if __name__ == '__main__':
    from inference_utils import pad_to_target

    image_size = (300, 400, 400)
    tile_size = (3, 512, 512)

    steps = compute_steps_for_sliding_window(image_size=image_size, tile_size=tile_size, tile_step_size=0.5)
    print(steps)