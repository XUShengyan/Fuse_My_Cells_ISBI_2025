from models import unet_pseudo3d_addZatt
from models import ssim_predictor

MODELS = {
    'unet3d_pseudo_zAtt': unet_pseudo3d_addZatt.initialize_network,
    'ssim_predictor': ssim_predictor.initialize_network,
}
def create_model(config):
    model_type = config.model.name
    model_params = config.model.params
    return MODELS[model_type](model_params)

