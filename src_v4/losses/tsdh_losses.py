from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.losses.triplet_homography_loss import ContentAwareTripletLoss
from src.geometry.dlt import transform_points
from src_v4.models.temporal_support_refiner import normalize_residual_map, support_from_cycle_target


def _normalize_H(H: torch.Tensor) -> torch.Tensor:
    denom = H[:, 2:3, 2:3]
    denom = torch.where(denom.abs() < 1e-8, denom + 1e-8, denom)
    return H / denom


class HomographyCycleLoss(nn.Module):
    def forward(self, H01: torch.Tensor, H12: torch.Tensor, H02: torch.Tensor) -> torch.Tensor:
        comp = _normalize_H(H01.float().bmm(H12.float()))
        tgt = _normalize_H(H02.float())
        return F.smooth_l1_loss(comp, tgt)


class SupportRegularizer(nn.Module):
    def __init__(self, min_support: float = 0.15, tv_weight: float = 0.03):
        super().__init__()
        self.min_support = float(min_support)
        self.tv_weight = float(tv_weight)

    def forward(self, support: torch.Tensor) -> torch.Tensor:
        s = support.float().clamp(0, 1)
        area = F.relu(self.min_support - s.mean()).pow(2)
        tv_h = (s[..., 1:, :] - s[..., :-1, :]).abs().mean()
        tv_w = (s[..., :, 1:] - s[..., :, :-1]).abs().mean()
        return area + self.tv_weight * (tv_h + tv_w)


class SparsePointCalibrationLoss(nn.Module):
    def forward(self, H_official: torch.Tensor, pts_a: torch.Tensor, pts_b: torch.Tensor) -> torch.Tensor:
        H_ab = torch.linalg.inv(H_official.float())
        pred_b = transform_points(pts_a.float(), H_ab)
        err = torch.linalg.norm(pred_b - pts_b.float(), dim=-1)
        return torch.sqrt(err.pow(2) + 1e-6).mean()


class TSDHLoss(nn.Module):
    """Losses for V4 / TSDH-Net.

    The important distinction from V3 is the pixel-level temporal support loss:
    temporal cycle residuals supervise the support map, rather than using only a
    global matrix-composition regularizer.
    """
    def __init__(self, margin: float = 1.0, weights: dict | None = None,
                 min_support: float = 0.15, cycle_support_alpha: float = 1.0):
        super().__init__()
        self.triplet = ContentAwareTripletLoss(margin=margin)
        self.support_reg = SupportRegularizer(min_support=min_support)
        self.point_loss = SparsePointCalibrationLoss()
        self.h_cycle = HomographyCycleLoss()
        self.cycle_support_alpha = float(cycle_support_alpha)
        self.weights = {
            'triplet': 1.0,
            'init_triplet': 0.05,
            'support_reg': 0.005,
            'pixel_cycle_support': 0.05,
            'homography_cycle': 0.005,
            'nonh': 0.02,
            'decomposition': 0.02,
            'point': 0.0,
        }
        if weights:
            self.weights.update({k: float(v) for k, v in weights.items()})

    def _weighted_feature_l1(self, a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        d = (a.float() - b.float()).abs().mean(dim=1, keepdim=True)
        m = mask.float().clamp(0, 1)
        return (d * m).sum() / m.sum().clamp_min(1e-6)

    def _init_triplet(self, out: dict) -> torch.Tensor:
        return self.triplet({
            'Fb': out['Fb'],
            'Fa': out['Fa'],
            'pred_ib': out['init_pred_ib'],
            'pred_ib_feature': out['init_pred_ib_feature'],
            'mask_ap': out['support_init'],
        })['loss']

    def pair_losses(self, out: dict, pts_a: torch.Tensor | None = None, pts_b: torch.Tensor | None = None) -> dict:
        triplet = self.triplet(out)['loss']
        init_triplet = self._init_triplet(out)
        support_reg = self.support_reg(out['support_ap'])
        nonh_target = (1.0 - out['support_temporal'].detach()).clamp(0, 1)
        nonh = F.binary_cross_entropy(out['nonh_map'].float().clamp(1e-5, 1 - 1e-5), nonh_target.float())
        # Decomposition: high support should have low final residual; non-support
        # can be explained by nonH instead of corrupting H.
        res = normalize_residual_map(out['residual_final'])
        s = out['support_temporal'].float().clamp(0, 1)
        n = out['nonh_map'].float().clamp(0, 1)
        decomposition = (s * res + 0.2 * (1.0 - s) * (1.0 - n)).mean()
        losses = {
            'triplet': triplet,
            'init_triplet': init_triplet,
            'support_reg': support_reg,
            'nonh': nonh,
            'decomposition': decomposition,
        }
        if pts_a is not None and pts_b is not None:
            losses['point'] = self.point_loss(out['H'], pts_a, pts_b)
        losses['loss'] = sum(self.weights.get(k, 0.0) * v for k, v in losses.items())
        return losses

    def triplet_losses(self, triplet_out: dict) -> dict:
        out01, out12, out02 = triplet_out['out01'], triplet_out['out12'], triplet_out['out02']
        l01 = self.pair_losses(out01)
        l12 = self.pair_losses(out12)
        l02 = self.pair_losses(out02)
        losses = {f'01_{k}': v for k, v in l01.items() if k != 'loss'}
        losses.update({f'12_{k}': v for k, v in l12.items() if k != 'loss'})
        losses.update({f'02_{k}': v for k, v in l02.items() if k != 'loss'})

        base_pair = (l01['loss'] + l12['loss'] + l02['loss']) / 3.0
        h_cycle = self.h_cycle(out01['H'], out12['H'], out02['H'])
        target = support_from_cycle_target(triplet_out['cycle_residual'], alpha=self.cycle_support_alpha)
        pix01 = F.l1_loss(out01['support_temporal'].float(), target)
        pix12 = F.l1_loss(out12['support_temporal'].float(), target)
        pix02 = F.l1_loss(out02['support_temporal'].float(), target)
        pixel_cycle_support = (pix01 + pix12 + pix02) / 3.0
        total = base_pair
        total = total + self.weights.get('homography_cycle', 0.0) * h_cycle
        total = total + self.weights.get('pixel_cycle_support', 0.0) * pixel_cycle_support
        losses.update({
            'triplet': (l01['triplet'] + l12['triplet'] + l02['triplet']) / 3.0,
            'init_triplet': (l01['init_triplet'] + l12['init_triplet'] + l02['init_triplet']) / 3.0,
            'support_reg': (l01['support_reg'] + l12['support_reg'] + l02['support_reg']) / 3.0,
            'nonh': (l01['nonh'] + l12['nonh'] + l02['nonh']) / 3.0,
            'decomposition': (l01['decomposition'] + l12['decomposition'] + l02['decomposition']) / 3.0,
            'homography_cycle': h_cycle,
            'pixel_cycle_support': pixel_cycle_support,
            'cycle_support_target_mean': target.mean(),
            'loss': total,
        })
        return losses

    def forward(self, out: dict, mode: str = 'pair', pts_a: torch.Tensor | None = None,
                pts_b: torch.Tensor | None = None) -> dict:
        if mode == 'triplet':
            return self.triplet_losses(out)
        return self.pair_losses(out, pts_a=pts_a, pts_b=pts_b)
