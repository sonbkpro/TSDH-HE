from __future__ import annotations
import torch
from .dlt import transform_points


def inverse_consistency_loss(Hab: torch.Tensor, Hba: torch.Tensor) -> torch.Tensor:
    eye = torch.eye(3, device=Hab.device, dtype=Hab.dtype).unsqueeze(0)
    prod = Hab @ Hba
    prod = prod / prod[:, 2:3, 2:3].clamp_min(1e-8)
    return ((prod - eye) ** 2).mean()


def point_l2_error(points_a: torch.Tensor, points_b: torch.Tensor, Hab: torch.Tensor) -> torch.Tensor:
    pred = transform_points(points_a, Hab)
    return torch.linalg.norm(pred - points_b, dim=-1).mean()
