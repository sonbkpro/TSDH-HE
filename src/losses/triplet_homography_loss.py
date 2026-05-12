from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContentAwareTripletLoss(nn.Module):
    """Triplet loss used by the official released Oneline implementation.

    The released code computes a per-pixel p=1 triplet margin loss:

        TripletMarginLoss(anchor=F_b, positive=F'_a, negative=F_a)

    and normalizes it by the learned RANSAC-style mask ``mask_ap``.
    """
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = float(margin)

    def loss_map(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        positive_dist = (positive.float() - anchor.float()).abs().sum(dim=1, keepdim=True)
        negative_dist = (negative.float() - anchor.float()).abs().sum(dim=1, keepdim=True)
        return F.relu(positive_dist - negative_dist + self.margin)

    def forward(self, model_out: dict) -> dict:
        loss_map = self.loss_map(model_out['Fb'], model_out['pred_ib_feature'], model_out['Fa'])
        mask = model_out['mask_ap'].float()
        denom = mask.sum().clamp_min(torch.finfo(mask.dtype).tiny)
        loss = (loss_map * mask).sum() / denom
        return {
            'loss': loss,
            'feature_loss': loss.detach(),
            'loss_map': loss_map.detach(),
            'mask_sum': mask.sum().detach(),
            'pred_ib': model_out['pred_ib'].detach(),
        }
