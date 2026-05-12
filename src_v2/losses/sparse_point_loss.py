from __future__ import annotations
import torch
import torch.nn as nn
from src.geometry.dlt import transform_points


class SparsePointBestOfKLoss(nn.Module):
    """Optional sparse point loss for labeled .npy pairs.

    V1 validation labels use source/target point pairs. V2 applies the loss to
    the best homography among K, so secondary hypotheses are not incorrectly
    penalized when a point lies on a non-dominant plane.
    """
    def __init__(self, weight: float = 0.5):
        super().__init__()
        self.weight = float(weight)

    def forward(self, Hs_dst_to_src: torch.Tensor, pts_a: torch.Tensor, pts_b: torch.Tensor):
        # Convert Oneline matrix convention to source -> target point mapping.
        Hs_ab = torch.linalg.inv(Hs_dst_to_src)
        errs = []
        for k in range(Hs_ab.size(1)):
            pred = transform_points(pts_a, Hs_ab[:, k])
            errs.append(torch.linalg.norm(pred - pts_b, dim=-1).mean(dim=1))
        err = torch.stack(errs, dim=1)  # [B,K]
        return self.weight * err.min(dim=1).values.mean()
