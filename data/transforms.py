import albumentations as A
from albumentations.core.composition import Compose

def build_3d_transforms(config, mode='train', need_int_transform=False):
    if mode != 'train':
        need_int_transform = False

    crop_size = config.data.crop_size
    p_spatial = 0.5
    p_intensity = 0.5


    base_transforms = [
        A.PadIfNeeded3D(min_zyx=crop_size, p=1.0)
    ]


    spatial_transforms = Compose([
        *base_transforms,
    ], additional_targets={'mask': 'image'})

    if need_int_transform:
        intensity_transforms = Compose([
            A.RandomGamma(p=p_intensity),
            A.GaussianBlur(p=p_intensity),
            A.RandomBrightnessContrast(p=p_intensity)
        ])
    else:

        intensity_transforms = None


    return spatial_transforms, intensity_transforms

