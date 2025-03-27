import os
import torch
import argparse
import platform
from utils.config_loader import load_config
from data.dataset import get_dataloaders
from models.model_factory import create_model
from train.trainer import Trainer

# Function to run the training process based on configuration
def run_training(config_path, config_name, task_name='Test'):
    """
    Load configuration, initialize model, data loaders, and trainer, then start training.
    """
    # Construct the full path to the configuration file
    config_file = os.path.join(config_path, config_name)
    config = load_config(config_file, task_name=task_name)

    # Set device based on GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get data loaders
    train_loader, val_loader = get_dataloaders(config=config)

    # Initialize model and trainer
    model = create_model(config)

    # Model loading
    model_path = config.paths.trained_model_path

    # Check if the model path is valid
    if model_path:
        if not os.path.isfile(model_path):
            print(f"Warning: Pre-trained model file does not exist at {model_path}, starting training from scratch")
        else:
            try:
                checkpoint = torch.load(model_path, map_location=device)
                # Load model parameters (automatically adapt to current device)
                model.load_state_dict(checkpoint['state_dict'])
                print(f"Successfully loaded pre-trained model: {model_path}")


            except Exception as e:
                print(f"Model loading failed: {str(e)}, starting training from scratch")
    else:
        print("No pre-trained model path provided, starting training from scratch")

    # Move model to device
    model = model.to(device)

    trainer = Trainer(model, train_loader, val_loader, config)

    # Execute training process
    try:
        trainer.train_model()
    except KeyboardInterrupt:
        print("Training interrupted, saving current state...")
    finally:
        trainer.finalize()


if __name__ == '__main__':
    # Determine parameter input method based on current OS
    if platform.system() == 'Windows':
        config_path = r"C:\PythonProjects\25_Fuse_my_cells\configs"
        config_name = r"final_settings_nuc.yaml"
        task_name = 'Unet_att_ssim'

        print("Running on Windows, using preset parameters:")
        print(f"config_path: {config_path}")
        print(f"config_name: {config_name}")
        run_training(config_path, config_name, task_name)

    else:
        # For non-Windows systems (e.g., Ubuntu), use command-line arguments
        parser = argparse.ArgumentParser(description="Train the model")
        parser.add_argument("--config_path", type=str, required=True, help="Directory path where the configuration file is located")
        parser.add_argument("--config_name", type=str, required=True, help="Configuration file name (e.g., train_all_data_valid_mem_4070s.yaml)")
        parser.add_argument("--task_name", type=str, default="unet3d_pseudoAtt_all", help="Task name, default is unet3d_pseudoAtt_all")
        args = parser.parse_args()
        run_training(args.config_path, args.config_name, args.task_name)