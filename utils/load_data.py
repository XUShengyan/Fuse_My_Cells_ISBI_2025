import numpy as np
import os
import tifffile


def index_tiff_folder(folder_path):
    """
    Index TIFF files in a folder and categorize them based on metadata.

    Args:
        folder_path (str): Path to the folder containing TIFF files.

    Returns:
        tuple: (nuc_files, mem_files)
    """
    nuc_files = []
    mem_files = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.tif', '.tiff')):
            file_path = os.path.join(folder_path, filename)
            try:
                with tifffile.TiffFile(file_path) as tif:
                    meta = None
                    if tif.imagej_metadata is not None:
                        meta = tif.imagej_metadata
                    elif tif.shaped_metadata is not None:
                        meta = tif.shaped_metadata[0]
                    else:
                        desc_tag = tif.pages[0].tags.get("ImageDescription")
                        if desc_tag is not None:
                            meta = desc_tag.value

                    meta_str = str(meta).lower() if meta is not None else ""

                    if "mem" in meta_str:
                        mem_files.append(file_path)
                    else:
                        nuc_files.append(file_path)

            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
                nuc_files.append(file_path)

    return nuc_files, mem_files


def load_tif_npy_data_with_meta(file_path: str) -> np.ndarray:
    """
    Load image data and metadata from TIFF or NPY files.

    Args:
        file_path (str): Path to the file.

    Returns:
        tuple: (image, meta)
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.tif', '.tiff']:
        image, meta = read_tiff_with_meta(file_path)

    elif ext == '.npy':
        npy_file = np.load(file_path, allow_pickle=True).item()
        image = npy_file['noisy']
        meta = npy_file['meta']
    else:
        raise ValueError("Unsupported file format, only .tif and .npz are supported")

    if image.ndim == 2:
        image = np.expand_dims(image, axis=0)
    elif image.ndim == 3 and image.shape[-1] < 10:
        image = image.transpose(2, 0, 1)

    return image, meta


def smart_convert(val):
    """
    Convert a string to a number (float or int) if possible, otherwise return the string.

    Args:
        val (str): Value to convert.

    Returns:
        int, float, or str: Converted value or original string.
    """
    try:
        num = float(val)
        if num.is_integer():
            return int(num)
        return num
    except ValueError:
        return val


def read_tiff_with_meta(filepath):
    """
    Read TIFF image and extract standardized metadata.

    Args:
        filepath (str): Path to the TIFF file.

    Returns:
        tuple: (image, meta_info)
    """
    with tifffile.TiffFile(filepath) as tif:
        image = tif.asarray()

        if tif.shaped_metadata is not None:
            shp_metadata = tif.shaped_metadata[0]
            meta_info = standardize_metadata(shp_metadata)

            return image, meta_info
        else:
            if tif.imagej_metadata is not None:
                shape = list(image.shape)
                imgj_metadata = tif.imagej_metadata
                imgj_metadata['shape'] = shape
                meta_info = standardize_metadata(imgj_metadata)

                return image, meta_info
            else:
                meta_info = tif.pages[0].tags['ImageDescription'].value
                print(f"Error loading metadata: {meta_info}, type of object: {type(meta_info)}")

    return image, meta_info


def standardize_metadata(metadata: dict):
    """
    Standardize metadata keys for consistency.

    Args:
        metadata (dict): Metadata dictionary.

    Returns:
        dict: Standardized metadata.
    """
    key_map = {
        "spacing": ["spacing"],
        "PhysicalSizeX": ["PhysicalSizeX", "physicalsizex", "physical_size_x"],
        "PhysicalSizeY": ["PhysicalSizeY", "physicalsizey", "physical_size_y"],
        "PhysicalSizeZ": ["PhysicalSizeZ", "physicalsizez", "physical_size_z"],
        "unit": ["unit"],
        "axes": ["axes"],
        "channel": ["channel"],
        "shape": ["shape"],
        "study": ["study"],
    }

    standardized_metadata = {}
    for standard_key, possible_keys in key_map.items():
        for key in possible_keys:
            if key in metadata:
                standardized_metadata[standard_key] = metadata[key]
                break

    return standardized_metadata