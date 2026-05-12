from __future__ import annotations
import torch
import torch.nn as nn


def normalize_mask(mask: torch.Tensor, strength: float = 0.5) -> torch.Tensor:
    mask = mask.float()
    b = mask.size(0)
    max_value = mask.reshape(b, -1).max(dim=1).values.reshape(b, 1, 1, 1)
    return (mask / (max_value * strength).clamp_min(torch.finfo(mask.dtype).eps)).clamp(0.0, 1.0)


class MaskPredictor(nn.Module):
    """Paper Table 1(b): 5 conv layers, 3x3 stride 1, channels 4,8,16,32,1.

    The official Oneline code applies ``normMask`` outside ``genMask`` after
    extracting the current patch, so ``forward`` returns the raw sigmoid mask.
    We keep this branch in FP32 even during AMP training; otherwise sigmoid can
    underflow to exact zeros and make the stage-2 weighted loss collapse.
    """
    def __init__(self, in_ch: int = 1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        if x.device.type in {'cuda', 'cpu'}:
            with torch.amp.autocast(x.device.type, enabled=False):
                return self.body(x.float())
        return self.body(x.float())
