from __future__ import annotations
import torch
import torch.nn as nn


class HomographyUncertaintyHead(nn.Module):
    """Small uncertainty/confidence head for dominant homography.

    It predicts 8 log-variances for four-corner offsets plus a global confidence.
    The head is intentionally lightweight and does not use pretrained models.
    """
    def __init__(self, in_ch: int = 2, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, 2, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(hidden * 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.logvar = nn.Linear(hidden * 2, 8)
        self.confidence = nn.Linear(hidden * 2, 1)

    def forward(self, pair_features: torch.Tensor) -> dict:
        z = self.net(pair_features)
        # Clamp range in the model output to avoid early numerical explosion.
        logvar = self.logvar(z).clamp(min=-6.0, max=6.0)
        confidence = torch.sigmoid(self.confidence(z))
        return {'offset_logvar': logvar, 'confidence': confidence}
