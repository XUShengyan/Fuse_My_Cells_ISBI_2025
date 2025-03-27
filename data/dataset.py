# dataset.py
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from glob import glob
import torch
import matplotlib.pyplot as plt
from data.transforms import build_3d_transforms
from utils.normalization import percentile_normalization
import torch.nn.functional as F

class Microscope3DDataset(Dataset):
    def __init__(self,
                 data_root,
                 mode='train',
                 config=None):
        """
        Initialize the dataset.

        Args:
            data_root (str): Root directory of the data (contains train/val subdirectories)
            mode (str): 'train' or 'val'
            config (Config): Configuration object containing data settings
        """
        self.mode = mode
        self.cfg = config.data
        self.transform = self.cfg.transform
        self.target_res = np.array(self.cfg.target_resolution)
        self.crop_size = np.array(self.cfg.crop_size)

        self.data_dir = os.path.join(data_root, mode)

        # Build transformations
        spatial_trans, intensity_trans = build_3d_transforms(config=config,
                                                             mode=self.mode,
                                                             need_int_transform=self.transform)
        self.spatial_transform = spatial_trans
        self.intensity_transform = intensity_trans

        # Get all data file paths
        self.data_files = glob(os.path.join(self.data_dir, "*.npy")) + glob(os.path.join(self.data_dir, "*.npz"))
        assert len(self.data_files) > 0, f"No NPY files found in {self.data_dir}"

    def __len__(self):
        """Return the number of data files."""
        return len(self.data_files)

    def __getitem__(self, idx):
        """Load and preprocess data for a given index."""
        # Load data based on file extension
        file_path = self.data_files[idx]
        if file_path.endswith('.npy'):
            data = np.load(file_path, allow_pickle=True).item()
        elif file_path.endswith('.npz'):
            data = np.load(file_path, allow_pickle=True)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        noisy = data['noisy'].astype(np.float32)  # [D, H, W]
        clean = data['clean'].astype(np.float32)
        meta = data['meta']

        # Apply spatial transformations
        transformed = self.spatial_transform(
            volume=noisy,
            mask3d=clean
        )
        noisy, clean = transformed['volume'], transformed['mask3d']

        # Apply normalization (percentile normalization)
        noisy = percentile_normalization(noisy)
        clean = percentile_normalization(clean)

        # Apply intensity transformations if in training mode and transformations are enabled
        if self.mode == 'train' and self.transform is not False:
            transformed = self.intensity_transform(
                image=noisy,
                mask=clean
            )
            noisy, clean = transformed['image'], transformed['mask']

        # Convert to torch tensors and add channel dimension
        noisy_tensor = torch.from_numpy(noisy).unsqueeze(0)  # [1, D, H, W]
        clean_tensor = torch.from_numpy(clean).unsqueeze(0)

        return noisy_tensor, clean_tensor

    def _resample_volume(self, volume, original_spacing):
        """
        Resample the volume to the target resolution.

        Args:
            volume (np.ndarray): Input volume [D, H, W]
            original_spacing (np.ndarray): Original spacing [z, x, y]
        Returns:
            np.ndarray: Resampled volume
        """
        scale_factors = original_spacing / self.target_res

        # If all scale factors are less than 3, do not resample
        if np.all(scale_factors < 3):
            return volume

        new_shape = np.round(np.array(volume.shape) * scale_factors).astype(int)

        # Convert to torch tensor and add batch and channel dimensions
        tensor_volume = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0)

        # Use trilinear interpolation for resampling
        resampled = F.interpolate(tensor_volume,
                                  size=tuple(new_shape),
                                  mode='trilinear',
                                  align_corners=False)
        return resampled.squeeze().numpy()

def get_dataloaders(config):
    """Create and return DataLoaders for training and validation."""
    data_root = config.paths.root_dir
    batch_size = config.data.batch_size
    batch_size_val = config.data.batch_size_val
    num_workers = config.data.num_workers

    # Training-related settings
    train_batches_per_epoch = config.training.train_batches_per_epoch
    val_batches_per_epoch = config.training.val_batches_per_epoch

    train_dataset = Microscope3DDataset(
        data_root=data_root,
        mode='train',
        config=config)

    val_dataset = Microscope3DDataset(
        data_root=data_root,
        mode='val',
        config=config)

    # Check if datasets are empty
    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty. Please check the data path.")
    if len(val_dataset) == 0:
        raise ValueError("Validation dataset is empty. Please check the data path.")

    # Training set sampler logic
    train_sampler = None
    if train_batches_per_epoch is not None and train_batches_per_epoch > 0:
        num_train_samples = train_batches_per_epoch * batch_size
        replace = num_train_samples > len(train_dataset)
        train_sampler = torch.utils.data.RandomSampler(
            train_dataset,
            replacement=replace,
            num_samples=num_train_samples
        )
        shuffle_train = False  # Sampler handles randomness
    else:
        shuffle_train = True

    # Validation set sampler logic
    val_sampler = None
    if val_batches_per_epoch is not None and val_batches_per_epoch > 0:
        num_val_samples = val_batches_per_epoch * batch_size_val
        replace = num_val_samples > len(val_dataset)
        val_sampler = torch.utils.data.RandomSampler(
            val_dataset,
            replacement=replace,
            num_samples=num_val_samples
        )
        shuffle_val = False  # Sampler handles randomness
    else:
        shuffle_val = True

    # Check batch_size validity when not using sampler
    drop_last_train = True
    drop_last_val = True

    if train_sampler is None:
        check_batch_size_validity(train_dataset, batch_size, drop_last_train, "Training", "train_batches_per_epoch")

    if val_sampler is None:
        check_batch_size_validity(val_dataset, batch_size_val, drop_last_val, "Validation", "val_batches_per_epoch")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True  # Keep worker processes
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size_val,
        sampler=val_sampler,
        shuffle=shuffle_val,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True  # Keep worker processes
    )

    return train_loader, val_loader

def check_batch_size_validity(dataset, batch_size, drop_last, dataset_name, use_batches_per_epoch):
    """Check if batch_size is valid for the dataset."""
    if len(dataset) < batch_size:
        if drop_last:
            raise ValueError(
                f"{dataset_name} dataset has fewer samples ({len(dataset)}) than batch_size ({batch_size}) and drop_last=True. No data will be generated. "
                f"Please reduce batch_size, set drop_last=False, or use {use_batches_per_epoch}."
            )
        else:
            import warnings
            warnings.warn(f"{dataset_name} dataset has fewer samples ({len(dataset)}) than batch_size ({batch_size}). The last batch will contain {len(dataset) % batch_size} samples.")

def visualize_slice(noisy_tensor, clean_tensor, slice_idx=None):
    """
    Visualize a slice of noisy and clean data for comparison.

    Args:
        noisy_tensor (Tensor): Shape [1, D, H, W]
        clean_tensor (Tensor): Shape [1, D, H, W]
        slice_idx (int): Index of the slice to visualize; if None, select the middle slice
    """
    assert noisy_tensor.ndim == 4 and clean_tensor.ndim == 4, "Input tensors must be 4D: [1, D, H, W]"

    _, D, H, W = noisy_tensor.shape

    # If slice_idx is not specified, use the middle slice
    if slice_idx is None:
        slice_idx = D // 2  # Select middle slice

    noisy_slice = noisy_tensor[0, slice_idx, :, :].cpu().numpy()
    clean_slice = clean_tensor[0, slice_idx, :, :].cpu().numpy()

    # Create subplots: 1 row, 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Left: noisy
    axes[0].imshow(noisy_slice, cmap='gray',
                   vmin=noisy_slice.min(), vmax=noisy_slice.max())
    axes[0].set_title(f"Noisy (slice={slice_idx})")
    axes[0].axis('off')

    # Right: clean
    axes[1].imshow(clean_slice, cmap='gray',
                   vmin=clean_slice.min(), vmax=clean_slice.max())
    axes[1].set_title(f"Clean (slice={slice_idx})")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    from utils.config_loader import load_config
    config = load_config(r"..\configs\final_settings_mem.yaml")

    # Get data loaders (without applying data augmentation for now)
    train_loader, val_loader = get_dataloaders(config=config)

    # Test training set
    print("\nTesting training set:")
    for i, (noisy, clean) in enumerate(train_loader):
        print(f"Batch {i + 1}:")
        print(f"Noisy shape: {noisy.shape} | dtype: {noisy.dtype}")
        print(f"Clean shape: {clean.shape} | dtype: {clean.dtype}")
        print(f"Noisy range: [{noisy.min():.2f}, {noisy.max():.2f}]")
        print(f"Clean range: [{clean.min():.2f}, {clean.max():.2f}]\n")

        if i == 1:  # Check only the first two batches
            break

    # Test validation set
    print("\nTesting validation set:")
    sample = next(iter(val_loader))
    print(f"Val sample shape: {sample[0].shape}")