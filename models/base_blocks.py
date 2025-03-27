import torch.nn as nn

class BaseConv2D(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Downsample3D(nn.Module):
    def __init__(self, scale=(1, 2, 2)):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=scale)

    def forward(self, x):
        return self.pool(x)

class Upsample3D(nn.Module):

    def __init__(self, in_c, out_c, scale=(1, 2, 2), mode='transpose'):
        super().__init__()
        if mode == 'transpose':
            self.up = nn.ConvTranspose3d(
                in_c, out_c, kernel_size=scale,
                stride=scale
            )
        else:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=scale, mode='trilinear'),
                nn.Conv3d(in_c, out_c, kernel_size=1)
            )

    def forward(self, x):
        return self.up(x)


