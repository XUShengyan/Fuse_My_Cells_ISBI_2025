import torch
import numpy as np

def percentile_normalization_tensor(image: torch.Tensor, pmin: float = 2, pmax: float = 99.8, dim=None) -> torch.Tensor:
    """
    Perform percentile normalization on a PyTorch tensor.

    Args:
        image (torch.Tensor): Input image tensor of any dimension.
        pmin (float): Lower percentile (default: 2).
        pmax (float): Upper percentile (default: 99.8).
        dim: Dimension(s) to compute percentiles over; if None, uses flattened tensor.

    Returns:
        torch.Tensor: Normalized tensor with values approximately in [0, 1].
    """
    if not image.is_floating_point():
        image = image.float()

    if not (isinstance(pmin, (int, float)) and isinstance(pmax, (int, float)) and 0 <= pmin < pmax <= 100):
        raise ValueError("pmin and pmax must be between 0 and 100 with pmin < pmax")

    qmin = pmin / 100.0
    qmax = pmax / 100.0

    low = torch.quantile(image, qmin, dim=dim, keepdim=True)
    high = torch.quantile(image, qmax, dim=dim, keepdim=True)

    if torch.isclose(low, high).all():
        print(f"Warning: Low percentile {low} and high percentile {high} are the same. Image may be empty.")
        return image

    return (image - low) / (high - low)


def percentile_normalization(image, pmin=2, pmax=99.8, axis=None):
    """
    Compute percentile normalization for a NumPy array.

    Args:
        image (array): 2D or 3D NumPy array of the image.
        pmin (int or float): Lower percentile (0 to 100, default: 2).
        pmax (int or float): Upper percentile (0 to 100, default: 99.8).
        axis: Axis or axes to compute percentiles over; if None, uses flattened array.

    Returns:
        np.ndarray: Normalized image with values clipped to [0, 1].
    """
    if not (np.isscalar(pmin) and np.isscalar(pmax) and 0 <= pmin < pmax <= 100):
        raise ValueError("Invalid values for pmin and pmax")

    low_percentile = np.percentile(image, pmin, axis=axis, keepdims=True)
    high_percentile = np.percentile(image, pmax, axis=axis, keepdims=True)

    if low_percentile == high_percentile:
        print(f"Same min {low_percentile} and high {high_percentile}, image may be empty")
        return image

    norm_image = (image - low_percentile) / (high_percentile - low_percentile)
    norm_image = np.clip(norm_image, 0, 1)
    return norm_image