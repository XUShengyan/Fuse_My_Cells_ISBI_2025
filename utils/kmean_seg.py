#!/usr/bin/env python
import numpy as np
from sklearn.cluster import KMeans
from scipy.ndimage import label
import tifffile

def kmeans_segmentation(image, n_clusters=3):
    """
    Perform KMeans segmentation on a 3D grayscale image.

    Args:
        image: 3D numpy array (D, H, W)
        n_clusters: Number of clusters, default is 3

    Returns:
        labels: Segmentation result, same shape as image, each pixel assigned a class label
    """
    flat_image = image.reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(flat_image)
    labels = kmeans.labels_.reshape(image.shape)
    return labels

def merge_classes(labels, image):
    """
    Remove the class with the lowest mean intensity as background and merge remaining classes into a binary mask.

    Args:
        labels: KMeans segmentation result, same shape as image
        image: Original 3D image, used to compute mean intensity of each class

    Returns:
        mask: Binary mask, non-background areas as 1, background as 0
    """
    unique_labels = np.unique(labels)
    means = {label_val: image[labels == label_val].mean() for label_val in unique_labels}
    background_label = min(means, key=means.get)
    mask = (labels != background_label).astype(np.uint8)
    return mask

def remove_small_regions(mask, min_volume):
    """
    Remove connected components smaller than min_volume from the binary mask using connected component analysis.

    Args:
        mask: Binary mask (0 or 1), can be 2D or 3D array
        min_volume: Minimum volume (pixels or voxels) required to retain a component

    Returns:
        cleaned_mask: Mask with small connected components removed
    """
    labeled_array, num_features = label(mask)
    sizes = np.bincount(labeled_array.ravel())
    keep = sizes >= min_volume
    keep[0] = False
    cleaned_mask = keep[labeled_array]
    return cleaned_mask.astype(mask.dtype)

def save_mask_as_tif(mask, filepath):
    """
    Save the binary mask as a TIFF file.

    Args:
        mask: Binary mask array (2D or 3D), values 0 or 1
        filepath: Path to save the file, e.g., "cleaned_mask.tif"

    Notes:
        Converts mask to uint8 type and sets foreground to 255 for better visualization.
    """
    mask_uint8 = (mask.astype(np.uint8)) * 255
    tifffile.imwrite(filepath, mask_uint8)
    print(f"Mask saved as {filepath}")

def kmeans_segmentation_pipeline(image, n_clusters=3, min_volume=10):
    """
    Execute the full KMeans segmentation pipeline.

    Args:
        image: Input 3D image
        n_clusters: Number of clusters for KMeans, default is 3
        min_volume: Minimum volume for retaining connected components, default is 10

    Returns:
        mask: Final binary mask after segmentation and cleanup
    """
    labels = kmeans_segmentation(image, n_clusters=n_clusters)
    mask = merge_classes(labels, image)
    mask = remove_small_regions(mask, min_volume)
    return mask