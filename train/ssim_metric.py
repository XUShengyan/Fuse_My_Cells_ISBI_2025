from monai.metrics.regression import SSIMMetric

def calculate_ssim_2d(inputs, targets):
    """
    Calculate SSIM on 2D slices of 3D data by converting [B, C, D, H, W] to [B*D, C, H, W].
    Uses SSIMMetric with spatial_dims=2.
    """
    ssim_metric = SSIMMetric(spatial_dims=2, data_range=1.0)

    inputs_2d = inputs.permute(0, 2, 1, 3, 4).reshape(-1, inputs.size(1), inputs.size(3), inputs.size(4))
    targets_2d = targets.permute(0, 2, 1, 3, 4).reshape(-1, targets.size(1), targets.size(3), targets.size(4))

    reference_ssim = ssim_metric(inputs_2d, targets_2d)

    B = inputs.size(0)
    D = inputs.size(2)

    reference_ssim = reference_ssim.view(B, D).mean(dim=1)

    return reference_ssim