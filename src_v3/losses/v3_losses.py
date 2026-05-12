from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.losses.triplet_homography_loss import ContentAwareTripletLoss
from src.geometry.dlt import transform_points


def _normalize_H(H: torch.Tensor) -> torch.Tensor:
    denom = H[:, 2:3, 2:3].clamp_min(1e-8)
    return H / denom


class TemporalCycleLoss(nn.Module):
    """Composition loss for official H orientation.

    In the released V1 branch, H_01 is used as target->source sampling (frame1->frame0).
    Therefore H_02 should be close to H_01 @ H_12.
    """
    def forward(self, H01: torch.Tensor, H12: torch.Tensor, H02: torch.Tensor) -> torch.Tensor:
        comp = _normalize_H(H01.float().bmm(H12.float()))
        tgt = _normalize_H(H02.float())
        return F.smooth_l1_loss(comp, tgt)


class StaticSupportRegularizer(nn.Module):
    def __init__(self, min_support: float = 0.15, tv_weight: float = 0.05):
        super().__init__()
        self.min_support = float(min_support)
        self.tv_weight = float(tv_weight)

    def forward(self, support: torch.Tensor) -> torch.Tensor:
        s = support.float().clamp(0, 1)
        mean_penalty = F.relu(self.min_support - s.mean()).pow(2)
        tv_h = (s[..., 1:, :] - s[..., :-1, :]).abs().mean()
        tv_w = (s[..., :, 1:] - s[..., :, :-1]).abs().mean()
        return mean_penalty + self.tv_weight * (tv_h + tv_w)


class SparsePointCalibrationLoss(nn.Module):
    def __init__(self, robust: bool = True):
        super().__init__()
        self.robust = bool(robust)

    def forward(self, H_official: torch.Tensor, pts_a: torch.Tensor, pts_b: torch.Tensor,
                logvar: torch.Tensor | None = None) -> torch.Tensor:
        # V1 evaluator uses inv(H) for A->B, so keep this convention.
        H_ab = torch.linalg.inv(H_official.float())
        pred_b = transform_points(pts_a.float(), H_ab)
        err = torch.linalg.norm(pred_b - pts_b.float(), dim=-1)  # [B,N]
        if logvar is None:
            return torch.sqrt(err.pow(2) + 1e-6).mean() if self.robust else err.mean()
        # Use average corner logvar as global geometric uncertainty.
        s = logvar.float().mean(dim=1, keepdim=True).clamp(-6, 6)
        nll = 0.5 * torch.exp(-s) * err.pow(2) + 0.5 * s
        return nll.mean()


class V3DominantHomographyLoss(nn.Module):
    def __init__(self, margin: float = 1.0, weights: dict | None = None,
                 min_support: float = 0.15):
        super().__init__()
        self.triplet = ContentAwareTripletLoss(margin=margin)
        self.support_reg = StaticSupportRegularizer(min_support=min_support)
        self.point_loss = SparsePointCalibrationLoss()
        self.weights = {
            'triplet': 1.0,
            'init_triplet': 0.25,
            'support': 0.01,
            'uncertainty': 0.01,
            'consensus': 0.05,
            'point': 0.0,
            'temporal_cycle': 0.1,
        }
        if weights:
            self.weights.update({k: float(v) for k, v in weights.items()})

    def _feature_l1(self, a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        diff = (a.float() - b.float()).abs().mean(dim=1, keepdim=True)
        m = mask.float()
        return (diff * m).sum() / m.sum().clamp_min(1e-6)

    def pair_losses(self, out: dict, pts_a: torch.Tensor | None = None, pts_b: torch.Tensor | None = None) -> dict:
        final_triplet = self.triplet(out)['loss']
        init_triplet = self.triplet({
            'Fb': out['Fb'],
            'Fa': out['Fa'],
            'pred_ib': out['init_pred_ib'],
            'pred_ib_feature': out['init_pred_ib_feature'],
            'mask_ap': out['support_init'],
        })['loss']
        support_loss = self.support_reg(out['support_ap'])
        align_final = self._feature_l1(out['pred_ib_feature'], out['Fb'], out['support_ap'])
        align_init = self._feature_l1(out['init_pred_ib_feature'], out['Fb'], out['support_init'])
        consensus_loss = F.relu(align_final - align_init.detach())
        # Feature-space NLL: confidence/uncertainty should reflect residual magnitude.
        s = out['offset_logvar'].float().mean(dim=1).clamp(-6, 6)
        per_b = ((out['pred_ib_feature'].float() - out['Fb'].float()).abs().mean(dim=(1, 2, 3)))
        uncertainty_loss = (0.5 * torch.exp(-s) * per_b.pow(2) + 0.5 * s).mean()
        losses = {
            'triplet': final_triplet,
            'init_triplet': init_triplet,
            'support': support_loss,
            'uncertainty': uncertainty_loss,
            'consensus': consensus_loss,
        }
        if pts_a is not None and pts_b is not None:
            losses['point'] = self.point_loss(out['H'], pts_a, pts_b, out.get('offset_logvar'))
        total = sum(self.weights.get(k, 0.0) * v for k, v in losses.items())
        losses['loss'] = total
        return losses

    def forward(self, out: dict, temporal: dict | None = None,
                pts_a: torch.Tensor | None = None, pts_b: torch.Tensor | None = None) -> dict:
        losses = self.pair_losses(out, pts_a=pts_a, pts_b=pts_b)
        if temporal is not None:
            cycle = TemporalCycleLoss()(temporal['out01']['H'], temporal['out12']['H'], temporal['out02']['H'])
            losses['temporal_cycle'] = cycle
            losses['loss'] = losses['loss'] + self.weights.get('temporal_cycle', 0.0) * cycle
        return losses
