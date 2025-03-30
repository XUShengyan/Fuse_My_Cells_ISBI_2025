# Fuse My Cells 2025 ISBI Challenge Codebase


## Overview
This is the codebase for solving the Fuse My Cells 2025 ISBI challenge.

This competition solution proposes an efficient image reconstruction method under limited computational resources (using a 4070 Super GPU).
The approach starts by analyzing the distribution characteristics of the sample data, followed by selecting a small subset for training (<10% Nucleus and ~40% Membrane data patches).
A lightweight model is then developed, integrating the UNet architecture with a Z-axis attention mechanism.
To further enhance performance, a custom loss function inspired by N_SSIM is designed.
Finally, in the post-processing stage, prior knowledge—such as an SSIM-based image quality prediction map and a segmentation foreground ratio map—is incorporated to refine the reconstructed image.


## Inference Process
1. Place the tif data to be inferred (membrane, nucleus, or a mixture of both) directly into the `input\images\fluorescence-lightsheet-3D-microscopy` directory under the project root.
2. Run `inference.py` directly, which will determine the required model type based on the meta information in the tif file.
3. The output results are located in `output\images\fused-fluorescence-lightsheet-3D-microscopy`.


## Algorithm Workflow
1. Original image -> [Pseudo 3D Unet combined with Z-axis attention] -> Preliminary prediction map
2. Original image -> [SSIM predictor] -> Predict SSIM between original and fused images
3. Original image -> [KMeans segmentation] -> Segmentation map
4. Post-processing stage: Combine the predicted SSIM and segmentation map parameters to evaluate the prediction map and finally generate the final prediction map.



