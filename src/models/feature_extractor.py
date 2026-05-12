from __future__ import annotations
import torch.nn as nn


class FeatureExtractor(nn.Module):
    """Paper Table 1(a): 3 conv layers, 3x3 stride 1, channels 4,8,1."""
    def __init__(self, in_ch: int = 1, out_ch: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)
