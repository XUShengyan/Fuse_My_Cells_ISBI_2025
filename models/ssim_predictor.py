import torch
import torch.nn as nn


class LightweightSSIMPredictor(nn.Module):
    # Initialize encoder, global pooling, and fully connected layer
    def __init__(self, in_c=1, features=[16, 32, 64]):
        super().__init__()

        self.encoder = nn.ModuleList()
        current_c = in_c
        for f in features:
            self.encoder.append(
                nn.Sequential(
                    nn.Conv2d(current_c, f, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2)
                )
            )
            current_c = f

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(features[-1], 1)

        # Reshape 3D input to 2D for processing

    def _reshape_3d_to_2d(self, x):
        B, C, D, H, W = x.shape
        return x.permute(0, 2, 1, 3, 4).contiguous().view(B * D, C, H, W)

        # Define forward pass for SSIM prediction

    def forward(self, x):
        x_2d = self._reshape_3d_to_2d(x)

        for block in self.encoder:
            x_2d = block(x_2d)

        x_2d = self.global_pool(x_2d)
        x_2d = x_2d.view(x_2d.size(0), -1)  # [B*3, C]

        ssim_pred = self.fc(x_2d)  # [B*3, 1]
        ssim_pred = torch.sigmoid(ssim_pred)  # Constrain output to [0, 1]

        B = x.shape[0]
        ssim_pred = ssim_pred.view(B, 3)  # [B, 3], SSIM predictions for 3 slices
        ssim_pred = ssim_pred.mean(dim=1)  # [B], average SSIM over 3 slices

        return ssim_pred  # Output [B]


# Initialize network with given parameters
def initialize_network(params):
    return LightweightSSIMPredictor(
        in_c=params.get('in_c', 1),
        features=params.get('features', [16, 32, 64])
    )


