from monai.metrics.regression import SSIMMetric


def calculate_n_ssim_3d(inputs, outputs, targets):
    """
    Calculate N-SSIM directly on 3D data with shape [B, C, D, H, W].
    Uses SSIMMetric with spatial_dims=3.
    """
    ssim_metric = SSIMMetric(spatial_dims=3, data_range=1.0)
    reference_ssim = ssim_metric(inputs, targets)
    prediction_ssim = ssim_metric(outputs, targets)
    n_ssim = (prediction_ssim - reference_ssim) / (1 - reference_ssim + 1e-8)
    return n_ssim.mean()


def calculate_n_ssim_2d(inputs, outputs, targets):
    """
    Calculate N-SSIM on 2D slices of 3D data.
    Converts [B, C, D, H, W] to [B*D, C, H, W] and uses SSIMMetric with spatial_dims=2.
    """
    ssim_metric = SSIMMetric(spatial_dims=2, data_range=1.0)
    inputs_2d = inputs.permute(0, 2, 1, 3, 4).reshape(-1, inputs.size(1), inputs.size(3), inputs.size(4))
    outputs_2d = outputs.permute(0, 2, 1, 3, 4).reshape(-1, outputs.size(1), outputs.size(3), outputs.size(4))
    targets_2d = targets.permute(0, 2, 1, 3, 4).reshape(-1, targets.size(1), targets.size(3), targets.size(4))
    reference_ssim = ssim_metric(inputs_2d, targets_2d)
    prediction_ssim = ssim_metric(outputs_2d, targets_2d)
    n_ssim = (prediction_ssim - reference_ssim) / (1 - reference_ssim + 1e-8)
    return n_ssim.mean(), prediction_ssim.mean()


def calculate_n_ssim(inputs, outputs, targets, method='2d'):
    """
    Calculate N-SSIM using either 2D or 3D method based on 'method' parameter.

    Args:
        inputs: Model input, shape [B, C, D, H, W]
        outputs: Model predictions, shape [B, C, D, H, W]
        targets: Ground truth, shape [B, C, D, H, W]
        method: '2d' or '3d' (default '2d')

    Returns:
        Mean N-SSIM value
    """
    if method == '2d':
        return calculate_n_ssim_2d(inputs, outputs, targets)
    elif method == '3d':
        return calculate_n_ssim_3d(inputs, outputs, targets)
    else:
        raise ValueError("method must be '2d' or '3d'")