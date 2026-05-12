from __future__ import annotations
import torch
import torch.nn as nn


class NonHomographicResidualHead(nn.Module):
    """Predicts regions that should not be explained by one global homography.

    This head is intentionally lightweight. It gives V4 an explicit decomposition:
    dominant homography support versus non-homographic residual regions caused by
    moving objects, parallax, occlusion or low-texture ambiguity.
    """
    def __init__(self, in_ch: int = 4, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 3, 1, 1),
        )

    def forward(self, evidence: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(evidence))
