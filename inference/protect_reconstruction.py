import numpy as np

def protected_reconstruction(img, pred, pred_ssim, threshold=None):
    """Protect high SSIM regions using pred_ssim."""
    # Ensure inputs are numpy arrays
    img = np.asarray(img)
    pred = np.asarray(pred)
    pred_ssim = np.asarray(pred_ssim)

    # Check if shapes match
    if img.shape != pred.shape or img.shape != pred_ssim.shape:
        raise ValueError("img, pred, and pred_ssim must have the same shape")

    # Calculate weight
    if threshold is None:
        weight = pred_ssim  # Linear weight
    else:
        weight = (pred_ssim > threshold).astype(float)  # Hard threshold

    # Blend to create new image
    new_img = (1 - weight) * pred + weight * img

    return new_img

def protected_reconstruction_foreground_ratio(img, pred, foreground_ratio_map, threshold=0.3):
    """Protect regions based on foreground ratio."""
    # Ensure inputs are numpy arrays
    img = np.asarray(img)
    pred = np.asarray(pred)
    foreground_ratio_map = np.asarray(foreground_ratio_map)

    # Hard threshold for weight
    weight = (foreground_ratio_map < threshold).astype(float)

    # Blend to create new image
    new_img = (1 - weight) * pred + weight * img

    return new_img