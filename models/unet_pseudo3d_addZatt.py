import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_blocks import BaseConv2D

class ZAxisAttention(nn.Module):
    """Lightweight z-axis attention module: performs self-attention across z-layers for each (h, w) position"""
    def __init__(self, channels, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)

    # Apply self-attention along the z-axis for each (h, w) position
    def forward(self, x):
        B, C, D, H, W = x.shape
        x_ = x.permute(0, 3, 4, 2, 1).reshape(B * H * W, D, C)
        qkv = self.qkv(x_)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B * H * W, D, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B * H * W, D, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B * H * W, D, self.num_heads, self.head_dim).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B * H * W, D, C)
        out = self.proj(out)
        out = out + x_
        out = out.view(B, H, W, D, C).permute(0, 4, 3, 1, 2)
        return out

class Pseudo3DUNet(nn.Module):
    # Initialize Pseudo3DUNet with encoder, decoder, and attention modules
    def __init__(self, in_c=1, out_c=1, features=[32, 64, 128, 256]):
        super().__init__()
        self.encoder = nn.ModuleList()
        current_c = in_c
        for idx, f in enumerate(features):
            self.encoder.append(
                nn.Sequential(
                    BaseConv2D(current_c, f),
                    BaseConv2D(f, f)
                )
            )
            current_c = f
        self.decoder = nn.ModuleList()
        for idx, f in enumerate(reversed(features[:-1])):
            self.decoder.append(
                nn.Sequential(
                    nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2),
                    BaseConv2D(f + features[-(idx + 2)], f),
                    BaseConv2D(f, f)
                )
            )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.final_conv_3d = nn.Conv3d(features[0], out_c, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.z_attn = ZAxisAttention(channels=features[0], num_heads=1)

    # Reshape 3D input to 2D for 2D convolution processing
    def _reshape_3d_to_2d(self, x):
        B, C, D, H, W = x.shape
        return x.permute(0, 2, 1, 3, 4).contiguous().view(B * D, C, H, W)

    # Reshape 2D output back to 3D format
    def _reshape_2d_to_3d(self, x, original_shape):
        B, C, D, H, W = original_shape
        return x.view(B, D, -1, H, W).permute(0, 2, 1, 3, 4)

    # Define forward pass through encoder, decoder, attention, and final convolution
    def forward(self, x):
        original_shape = x.shape
        x_2d = self._reshape_3d_to_2d(x)
        skips = []
        for block in self.encoder:
            x_2d = block(x_2d)
            skips.append(x_2d)
            x_2d = self.pool(x_2d)
        for i, block in enumerate(self.decoder):
            x_2d = block[0](x_2d)
            skip = skips[-(i + 2)]
            if x_2d.shape[-2:] != skip.shape[-2:]:
                x_2d = F.interpolate(x_2d, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x_2d = torch.cat([x_2d, skip], dim=1)
            x_2d = block[1](x_2d)
            x_2d = block[2](x_2d)
        x_3d = self._reshape_2d_to_3d(x_2d, original_shape)
        x_3d = self.z_attn(x_3d)
        x_3d = self.final_conv_3d(x_3d)
        return self.sigmoid(x_3d)

# Initialize network with given parameters
def initialize_network(params):
    return Pseudo3DUNet(
        in_c=params.get('in_c', 1),
        out_c=params.get('out_c', 1),
        features=params.get('features', [32, 64, 128, 256, 512])
    )