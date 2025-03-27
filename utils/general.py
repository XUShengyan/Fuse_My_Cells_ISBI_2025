import os
import shutil
import numpy as np
import stat
def remove_readonly(func, path, excinfo):
    os.chmod(path, stat.S_IWRITE)
    func(path)

def mkdir_x(input_path, emptyFlag = False):
    folder = os.path.exists(input_path)
    if not folder:
        os.makedirs(input_path)

    else:
        if emptyFlag:
            shutil.rmtree(input_path, onerror=remove_readonly)
            os.makedirs(input_path)

def normalize_volume(vol):
    vol_min = vol.min()
    vol_max = vol.max()
    return (vol - vol_min) / (vol_max - vol_min + 1e-8)

def get_filename_without_extension(file_path):
    base_name = os.path.basename(file_path)
    name, _ = os.path.splitext(base_name)
    return name

