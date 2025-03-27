import os
import torch

from inference.inference_utils import pad_to_target, scale_xy_and_pad, unscale_xy_and_unpad
from inference.sliding_inference import sliding_window_inference
from inference.background_worker import submit_prediction_for_postprocessing, init_executor, process_and_export
from inference.protect_reconstruction import protected_reconstruction, protected_reconstruction_foreground_ratio
from inference.sliding_inference_kMean import sliding_window_inference_kmean

from utils.general import mkdir_x, get_filename_without_extension
from utils.load_data import load_tif_npy_data_with_meta
from utils.normalization import percentile_normalization


# Function to perform the entire prediction pipeline for a list of files
def prediction_pipline(file_list, output_path, model, model_ssim, config, device, ssim_threshold, foreground_threshold):
    """
    Perform the prediction pipeline for a list of files, including normalization, inference, and post-processing.

    Args:
        file_list (list): List of input file paths.
        output_path (str): Directory to save the output files.
        model (torch.nn.Module): The main model for inference.
        model_ssim (torch.nn.Module): The SSIM prediction model.
        config (Config): Configuration object containing inference settings.
        device (str): Device to use for inference (e.g., 'cuda' or 'cpu').
        ssim_threshold (float): Threshold for SSIM-based reconstruction.
        foreground_threshold (float): Threshold for foreground ratio-based reconstruction.
    """
    # Read sliding window parameters from config
    tile_size = config.inference.tile_size  # e.g., [128, 256, 256]
    tile_step_size = config.inference.tile_step_size
    use_amp = config.inference.use_amp

    # Initialize background thread pool based on config
    init_executor(config.inference.max_tif_worker)

    for file_path in file_list:
        # Load image and metadata
        image, meta = load_tif_npy_data_with_meta(file_path)

        # Perform kMean reconstruction to get foreground ratio map
        foreground_ratio_map = sliding_window_inference_kmean(image=image,
                                                              tile_size=tile_size,
                                                              tile_step_size=tile_step_size)

        # Normalize the image
        image = percentile_normalization(image)

        # Record original image shape (excluding channel dimension)
        original_shape = image.shape

        # Pad image and foreground ratio map to target size
        image, pad_info = pad_to_target(image, target_size=tile_size)
        foreground_ratio_map, _ = pad_to_target(foreground_ratio_map, target_size=tile_size)

        # Perform sliding window inference for main model, merging overlapping regions with max probability
        prediction = sliding_window_inference(image, model, tile_size, tile_step_size, device,
                                              batch_size=config.inference.batch_size,
                                              use_amp=use_amp)
        prediction = prediction.squeeze(0).cpu().numpy()
        # Clear cache
        torch.cuda.empty_cache()

        # Perform sliding window inference for SSIM model, merging overlapping regions with max probability
        prediction_ssim = sliding_window_inference(image, model_ssim, tile_size, tile_step_size, device,
                                                   batch_size=config.inference.batch_size,
                                                   use_amp=use_amp)
        prediction_ssim = prediction_ssim.squeeze(0).cpu().numpy()
        # Clear cache
        torch.cuda.empty_cache()

        # Apply protected reconstruction using foreground ratio
        prediction = protected_reconstruction_foreground_ratio(img=image,
                                                               pred=prediction,
                                                               foreground_ratio_map=foreground_ratio_map,
                                                               threshold=foreground_threshold)

        # Apply protected reconstruction using SSIM
        prediction = protected_reconstruction(img=image,
                                              pred=prediction,
                                              pred_ssim=prediction_ssim,
                                              threshold=ssim_threshold)

        # Generate output file path: keep original name but change extension to .tif
        base_name = get_filename_without_extension(file_path)
        output_file = os.path.join(output_path, base_name + '.tif')

        # Submit prediction for post-processing and export (resize, convert segmentation, and save)
        process_and_export(prediction=prediction,
                           original_shape=original_shape,
                           pad_info=pad_info,
                           output_path=output_file,
                           meta=meta)


