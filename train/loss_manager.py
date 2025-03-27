import torch
import torch.nn as nn
from monai.utils import ensure_tuple_rep
from torch.nn.modules.loss import _Loss
from typing import Dict, Callable, Union, Sequence
from monai.losses import SSIMLoss
from monai.metrics.regression import KernelType

class LossManager:
    """Manage multiple loss functions with dynamic combination and weighting."""

    def __init__(self, loss_configs: list, device: str = 'cuda'):
        """
        Initialize LossManager with a list of loss configurations.

        Args:
            loss_configs: List of dictionaries with loss settings (name, weight, impl, params).
            device: Device to use for computations.
        """
        self.loss_funcs: Dict[str, dict] = {}
        self.device = device

        for config in loss_configs:
            self.add_loss(**config)

    def add_loss(self, name: str, weight: float, impl: Callable, params: dict = None):
        """
        Add a loss function to the manager.

        Args:
            name: Name of the loss function.
            weight: Weighting factor for the loss.
            impl: Loss function implementation or class.
            params: Optional parameters for the loss function.
        """
        if params is None:
            params = {}
        loss_instance = impl(**params) if isinstance(impl, type) else impl
        self.loss_funcs[name] = {
            'weight': weight,
            'impl': loss_instance.to(self.device),
            'params': {}
        }

    def compute_loss_with_details(self, pred: torch.Tensor, target: torch.Tensor, input: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        Compute total loss and detailed loss values.

        Args:
            pred: Predicted tensor.
            target: Target tensor.
            input: Input tensor for specific losses like NSSIM.

        Returns:
            Tuple of total loss and dictionary of individual loss values.
        """
        total_loss = 0.0
        details = {}

        for name, cfg in self.loss_funcs.items():
            if name == 'nssim':
                loss_val = cfg['impl'](pred, target, input)
            else:
                loss_val = cfg['impl'](pred, target)
            details[name] = loss_val.item()
            total_loss += cfg['weight'] * loss_val

        return total_loss, details


class SSIM3D(nn.Module):
    """SSIM loss for 3D data, computed slice-wise in 2D."""

    def __init__(self, spatial_dims: int = 2, data_range: float = 1.0, **kwargs):
        """
        Initialize SSIM3D loss.

        Args:
            spatial_dims: Spatial dimensions for SSIM computation (default: 2).
            data_range: Range of input data (e.g., 1.0 for [0,1]).
            **kwargs: Additional arguments for SSIMLoss.
        """
        super().__init__()
        self.ssim_loss = SSIMLoss(spatial_dims=spatial_dims, data_range=data_range, **kwargs)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute SSIM loss for 3D data.

        Args:
            inputs: Predicted tensor (B, C, D, H, W).
            targets: Target tensor (B, C, D, H, W).

        Returns:
            Loss value.
        """
        B, C, D, H, W = inputs.shape
        inputs_2d = inputs.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
        targets_2d = targets.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
        loss = self.ssim_loss(inputs_2d, targets_2d)
        return loss


class NSSIMLoss(_Loss):
    """Normalized SSIM loss for regression tasks."""

    def __init__(
        self,
        spatial_dims: int = 2,
        data_range: float = 1.0,
        kernel_type: KernelType | str = KernelType.GAUSSIAN,
        win_size: int | Sequence[int] = 11,
        kernel_sigma: float | Sequence[float] = 1.5,
        k1: float = 0.01,
        k2: float = 0.03,
        eps: float = 1e-8,
        reduction: str = "mean",
    ):
        """
        Initialize NSSIMLoss.

        Args:
            spatial_dims: Spatial dimensions for SSIM (default: 2).
            data_range: Range of input data.
            kernel_type: Kernel type for SSIM computation.
            win_size: Window size for SSIM.
            kernel_sigma: Sigma for kernel.
            k1: SSIM stability constant.
            k2: SSIM stability constant.
            eps: Small value to avoid division by zero.
            reduction: Reduction method ("mean", "sum", or "none").
        """
        super().__init__(reduction=reduction)
        self.eps = eps
        self.ssim_loss = SSIMLoss(
            spatial_dims=spatial_dims,
            data_range=data_range,
            kernel_type=kernel_type,
            win_size=win_size,
            kernel_sigma=kernel_sigma,
            k1=k1,
            k2=k2,
            reduction="none"
        )

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        input: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute normalized SSIM loss.

        Args:
            pred: Predicted tensor (B, C, D, H, W).
            target: Target tensor (B, C, D, H, W).
            input: Input tensor (B, C, D, H, W).

        Returns:
            Loss value.
        """
        assert input.dim() == 5 and pred.dim() == 5 and target.dim() == 5, "Inputs must be 5D tensors (B, C, D, H, W)"

        def flatten_3d_to_2d(x):
            return x.permute(0, 2, 1, 3, 4).reshape(-1, x.shape[1], x.shape[3], x.shape[4])

        input_2d = flatten_3d_to_2d(input)
        pred_2d = flatten_3d_to_2d(pred)
        target_2d = flatten_3d_to_2d(target)

        with torch.no_grad():
            ref_ssim = 1 - self.ssim_loss(input_2d, target_2d)
        pred_ssim = 1 - self.ssim_loss(pred_2d, target_2d)

        loss = -(pred_ssim - ref_ssim) * ref_ssim

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


if __name__ == '__main__':
    class TestModel(nn.Module):
        """Simple 3D convolution model for testing."""
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv3d(1, 1, kernel_size=3, padding=1)

        def forward(self, x):
            return self.conv(x)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    model = TestModel().to(device)
    model.train()

    torch.manual_seed(42)
    patch_size = (2, 1, 3, 64, 64)
    input = torch.rand(patch_size).to(device)
    target = torch.rand(patch_size).to(device)

    pred = model(input)
    pred.retain_grad()

    loss_configs = [
        {"name": "mse", "weight": 0.5, "impl": nn.MSELoss(), "params": {}},
        {"name": "l1", "weight": 0.3, "impl": nn.L1Loss(), "params": {}},
        {"name": "ssim3d", "weight": 0.5, "impl": SSIM3D},
        {"name": "nssim", "weight": 0.5, "impl": NSSIMLoss}
    ]

    loss_mgr = LossManager(loss_configs, device=device)

    total_loss, loss_details = loss_mgr.compute_loss_with_details(pred, target, input)

    print("\nLoss details:")
    for name, value in loss_details.items():
        print(f"{name}: {value:.4f}")
    print(f"Total loss: {total_loss.item():.4f}")

    try:
        total_loss.backward()
        grad_mean = pred.grad.abs().mean().item()
        print(f"\nGradient check passed! Mean gradient: {grad_mean:.2e}")
    except Exception as e:
        print(f"\nGradient computation failed: {str(e)}")

    inputs = (torch.ones(1, 1, 3, 128, 128) * 0.1).to('cuda')
    targets = (torch.ones(1, 1, 3, 128, 128) * 0.5).to('cuda')
    preds = (targets.clone().requires_grad_(True)).to('cuda')

    loss = NSSIMLoss()(preds, targets, inputs)
    assert abs(loss.item() - 0.0) < 1e-4, "Perfect prediction loss should be 0"
    loss.backward()
    assert not torch.isnan(preds.grad).any(), "Gradient contains NaN"

    preds = inputs.clone().requires_grad_(True)
    loss = NSSIMLoss()(preds, targets, inputs)
    assert abs(loss.item() - 1.0) < 1e-4, "Worst prediction loss should be 1"
    loss.backward()
    assert not torch.isnan(preds.grad).any(), "Gradient contains NaN"