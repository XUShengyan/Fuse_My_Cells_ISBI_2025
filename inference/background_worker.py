# background_worker.py
import torch
import torch.nn.functional as F
import numpy as np
import tifffile
from concurrent.futures import ThreadPoolExecutor
from utils.save_data import save_image
from inference.inference_utils import remove_padding, unscale_xy_and_unpad

# Executor is initialized in init_executor function, not here
executor = None


# Initialize the thread pool executor with a specified number of workers
def init_executor(max_workers: int):
    """Initialize the thread pool executor with a specified number of workers."""
    global executor
    executor = ThreadPoolExecutor(max_workers=max_workers)


# Post-process the prediction for denoising tasks and save the result
def process_and_export(prediction: torch.Tensor,
                       original_shape: tuple,
                       pad_info: tuple,
                       output_path: str,
                       meta):
    """
    Post-process the prediction for denoising tasks:
      - Remove padding to match the original image size if necessary.
      - Save the denoised image directly without further operations like argmax.

    Args:
        prediction (torch.Tensor): Prediction result, shape should match the denoising task, e.g., (C, D, H, W) or (C, H, W).
        original_shape (tuple): Original image dimensions, e.g., (D, H, W) or (H, W).
        pad_info (tuple): Padding information used during preprocessing.
        output_path (str): Path to save the exported file.
        meta: Metadata to be saved with the image.
    """
    # Remove padding to restore the original shape
    prediction = remove_padding(prediction, pad_info, original_shape)

    # Convert to numpy array, handle NaN and Inf, and clip to [0, 1]
    arr = np.clip(np.nan_to_num(prediction, nan=0.0, posinf=1.0, neginf=0.0), 0, 1).astype(np.float32)

    # Scale to uint16 for saving
    arr_uint16 = (arr * 65535).astype(np.uint16)

    # Save the image with metadata
    save_image(location=output_path, array=arr_uint16, metadata=meta)


# Submit the prediction to the background thread pool for post-processing
def submit_prediction_for_postprocessing(prediction: torch.Tensor, output_path: str, original_shape: tuple, meta):
    """Submit the prediction to the background thread pool for post-processing."""
    if executor is None:
        raise RuntimeError(
            "Background thread pool not initialized. Please call init_executor(config.inference.max_workers) first.")
    executor.submit(process_and_export, prediction, original_shape, output_path, meta)


if __name__ == '__main__':
    import torch

    prediction = torch.rand(1, 100, 224, 224).to('cuda')
    meta = []
    process_and_export(prediction, (100, 224, 224), (0, 0, 0, 0, 0, 0), './inference/prediction', meta)