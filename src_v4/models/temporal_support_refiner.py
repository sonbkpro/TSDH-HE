from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalSupportRefiner(nn.Module):
    """Residual/cycle-driven support predictor for dominant homography.

    This is the main V4 novelty module: unlike the V1 mask, it does not rely only
    on image appearance. It consumes the initial pairwise support and geometric
    evidence maps: pair residuals, long-range residuals and pixel-level temporal
    cycle residuals. The output is a soft support map indicating pixels that are
    temporally reliable for the single dominant global homography.
    """
    def __init__(self, in_ch: int = 5, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 3, 1, 1),
        )

    def forward(self, evidence: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(evidence))


def normalize_residual_map(r: torch.Tensor, detach_scale: bool = True) -> torch.Tensor:
    """Normalize residual maps per sample so alpha thresholds are stable."""
    r = r.float()
    scale = r.flatten(1).mean(dim=1).view(-1, 1, 1, 1).clamp_min(1e-6)
    if detach_scale:
        scale = scale.detach()
    return (r / scale).clamp(0.0, 10.0)


def support_from_cycle_target(cycle_residual: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    c = normalize_residual_map(cycle_residual.detach())
    return torch.exp(-float(alpha) * c).clamp(0.0, 1.0)
