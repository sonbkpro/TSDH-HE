from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def _normalize_H(H: torch.Tensor) -> torch.Tensor:
    return H / H[..., 2:3, 2:3].clamp_min(1e-8)


class V2MultiHomographyLoss(nn.Module):
    """Pairwise V2 objective.

    Main alignment term is a K-hypothesis extension of V1's released triplet
    loss: anchor=F_b, positive=F'_a(H_k), negative=F_a. Each hypothesis has its
    own assignment/reliability mask.

    The diversity term prevents all K hypotheses from collapsing to the same H.
    The entropy term encourages sharper region ownership.
    """
    def __init__(self, margin: float = 1.0, lambda_diversity: float = 0.02,
                 lambda_entropy: float = 0.01, lambda_balance: float = 0.001,
                 diversity_scale: float = 16.0):
        super().__init__()
        self.margin = float(margin)
        self.lambda_diversity = float(lambda_diversity)
        self.lambda_entropy = float(lambda_entropy)
        self.lambda_balance = float(lambda_balance)
        self.diversity_scale = float(diversity_scale)

    def forward(self, out: dict) -> dict:
        # pred: [B,K,C,H,W]; Fb/Fa: [B,C,H,W]; mask: [B,K,1,H,W]
        pred = out['pred_ib_feature'].float()
        fb = out['Fb'].float().unsqueeze(1)
        fa = out['Fa'].float().unsqueeze(1)
        mask = out['mask_ap'].float()

        pos = (pred - fb).abs().sum(dim=2, keepdim=True)
        neg = (fa - fb).abs().sum(dim=2, keepdim=True)
        triplet = F.relu(pos - neg + self.margin)

        denom = mask.sum(dim=(2, 3, 4)).clamp_min(1e-6)
        per_hyp = (triplet * mask).sum(dim=(2, 3, 4)) / denom
        loss_align = per_hyp.mean()

        offsets = out['offsets'].float()  # [B,K,8]
        K = offsets.size(1)
        div_terms = []
        for i in range(K):
            for j in range(i + 1, K):
                dist = (offsets[:, i] - offsets[:, j]).abs().mean(dim=1)
                div_terms.append(torch.exp(-dist / self.diversity_scale).mean())
        loss_div = torch.stack(div_terms).mean() if div_terms else offsets.new_tensor(0.0)

        assign = out['assignments'].float().clamp_min(1e-8)  # [B,K,H,W]
        entropy = -(assign * assign.log()).sum(dim=1).mean()

        # Prevent a degenerate solution where one assignment owns the entire image
        area = assign.mean(dim=(2, 3))  # [B,K]
        target = torch.full_like(area, 1.0 / K)
        loss_balance = (area - target).abs().mean()

        loss = (
            loss_align
            + self.lambda_diversity * loss_div
            + self.lambda_entropy * entropy
            + self.lambda_balance * loss_balance
        )
        return {
            'loss': loss,
            'loss_align': loss_align.detach(),
            'loss_diversity': loss_div.detach(),
            'loss_entropy': entropy.detach(),
            'loss_balance': loss_balance.detach(),
            'per_hypothesis_loss': per_hyp.detach(),
            'mask_sum': mask.sum().detach(),
        }


class V2TemporalCycleLoss(nn.Module):
    """Composition consistency for Oneline matrices.

    H01 maps frame1/output coords -> frame0/source coords.
    H12 maps frame2/output coords -> frame1/source coords.
    Therefore H01 @ H12 maps frame2 -> frame0 and should match H02.
    """
    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = float(weight)

    def forward(self, out01: dict, out12: dict, out02: dict) -> torch.Tensor:
        H01 = _normalize_H(out01['Hs'].float())
        H12 = _normalize_H(out12['Hs'].float())
        H02 = _normalize_H(out02['Hs'].float())
        comp = torch.matmul(H01, H12)
        comp = _normalize_H(comp)
        loss = (comp - H02).abs().mean()
        return self.weight * loss
