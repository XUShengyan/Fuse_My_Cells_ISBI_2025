import tifffile

def save_image(*, location, array, metadata):

    PhysicalSizeX = metadata['PhysicalSizeX']
    PhysicalSizeY = metadata['PhysicalSizeY']
    tifffile.imwrite(
        location,
        array,
        bigtiff=True, #Keep it for 3D images
        resolution=(1. / PhysicalSizeX, 1. / PhysicalSizeY),
        metadata=metadata,
        tile=(128, 128),
        )