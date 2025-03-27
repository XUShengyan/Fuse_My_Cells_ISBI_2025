"""
This script processes 3D fluorescence microscopy images using a pre-trained model.
It takes input images (membrane and nucleus), applies the model, and saves the results as full black images.
"""

print(" START IMPORT ")

# Import necessary libraries
import os
import sys
from os import listdir, mkdir
from pathlib import Path
import torch

from models.model_factory import create_model
from utils.config_loader import load_config
from utils.load_model import load_pth_model
from utils.load_data import index_tiff_folder
from utils.general import mkdir_x
from inference.inference_from_raw_data import prediction_pipline

print(" END IMPORT ")

# Set input path based on operating system
if sys.platform == "win32":
    INPUT_PATH = Path("input/images/fluorescence-lightsheet-3D-microscopy")
else:
    INPUT_PATH = Path("/input/images/fluorescence-lightsheet-3D-microscopy")
print(f" INPUT_PATH IS   " + str(INPUT_PATH))

# List files in input path
if os.name == 'nt':  # Windows
    command = "dir " + str(INPUT_PATH)
else:  # Unix/Linux
    command = "ls -l " + str(INPUT_PATH)
os.system(command)

# Set output path and create directories
if sys.platform == "win32":
    OUTPUT_PATH = Path("output")
else:
    OUTPUT_PATH = Path("/output")
(OUTPUT_PATH / "images").mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_PATH / "images"
(OUTPUT_PATH / "fused-fluorescence-lightsheet-3D-microscopy").mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_PATH / "fused-fluorescence-lightsheet-3D-microscopy"
print("OUTPUT IS " + str(OUTPUT_PATH))

# Set path for model weights
RESOURCE_PATH = Path("resources")
print(" RESOURCE_PATH IS   " + str(RESOURCE_PATH))

def run():
    """Run the inference process for membrane and nucleus images."""
    print(f" LIST IMAGES IN  {INPUT_PATH} ")

    # Create output directory and clear it if it exists
    mkdir_x(OUTPUT_PATH, emptyFlag=True)

    # Find TIFF files for nucleus and membrane
    nuc_index, mem_index = index_tiff_folder(INPUT_PATH)

    # Define tasks for processing membrane and nucleus images
    tasks = {
        'mem': {
            'files': mem_index,
            'config_yaml': './configs/final_settings_mem.yaml',
            'checkpoint': "mem.pth",
            'ssim_predictor_yaml': './configs/final_settings_mem_ssim_predicitor.yaml',
            'ssim_predictor_pth': 'mem_ssim_predictor.pth',
            'ssim_threshold': 0.9,
            'foregournd_threshold': 0.1,
        },
        'nuc': {
            'files': nuc_index,
            'config_yaml': './configs/final_settings_nuc.yaml',
            'checkpoint': "nuc.pth",
            'ssim_predictor_yaml': './configs/final_settings_mem_ssim_predicitor.yaml',
            'ssim_predictor_pth': 'nuc_ssim_predictor.pth',
            'ssim_threshold': 0.9,
            'foregournd_threshold': 0.3,
        },
    }

    # Process each task (membrane or nucleus)
    for task_name, task_params in tasks.items():
        file_list = task_params['files']
        if not file_list:
            print(f"Task: {task_name} no match data, continue...")
            continue

        # Load task configuration and model
        config = load_config(task_params['config_yaml'], task_name='Final_predict', creat_dir=False)
        checkpoint_path = RESOURCE_PATH / task_params['checkpoint']
        model = create_model(config=config)
        print(" LOAD NETWORK ")
        model = load_pth_model(model, checkpoint_path=checkpoint_path, device='cuda')

        # Load SSIM predictor model
        config = load_config(task_params['ssim_predictor_yaml'], task_name='Final_predict_ssim', creat_dir=False)
        checkpoint_path = RESOURCE_PATH / task_params['ssim_predictor_pth']
        model_ssim = create_model(config=config)
        model_ssim = load_pth_model(model_ssim, checkpoint_path=checkpoint_path, device='cuda')

        # Set thresholds and run prediction
        ssim_threshold = task_params['ssim_threshold']
        foreground_threshold = task_params['foregournd_threshold']
        prediction_pipline(file_list=file_list, output_path=OUTPUT_PATH,
                           model=model, model_ssim=model_ssim,
                           config=config, device='cuda',
                           ssim_threshold=ssim_threshold, foreground_threshold=foreground_threshold)

        # Clear GPU memory
        torch.cuda.empty_cache()

    # List the output images
    print(" --> LIST OUTPUT IMAGES IN " + str(OUTPUT_PATH))
    for output_images in listdir(OUTPUT_PATH):
        print(" --> FOUND " + str(output_images))
    return 0

if __name__ == "__main__":
    raise SystemExit(run())