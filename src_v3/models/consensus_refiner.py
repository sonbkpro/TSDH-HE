from __future__ import annotations
import itertools
import torch
import torch.nn as nn
from src.geometry.dlt import solve_homography_dlt
from src.geometry.warp import transform_official_patch


class LocalConsensusRefiner(nn.Module):
    """Differentiable local consensus around one global homography.

    This is *not* a K-plane model. It samples local perturbations around the
    single regressed offset vector and softly selects the candidate with the best
    support-weighted feature residual.
    """
    def __init__(self, radius: float = 2.0, temperature: float = 0.15, max_candidates: int = 9):
        super().__init__()
        self.radius = float(radius)
        self.temperature = float(temperature)
        self.max_candidates = int(max_candidates)

    def _candidate_offsets(self, offsets: torch.Tensor) -> torch.Tensor:
        b, d = offsets.shape
        if d != 8:
            raise ValueError('offsets must be [B,8]')
        # Candidate set: zero, global x/y translations, scale-like corner spreads.
        # Kept small for speed and stable early training.
        device, dtype = offsets.device, offsets.dtype
        r = torch.tensor(self.radius, device=device, dtype=dtype)
        patterns = []
        patterns.append(torch.zeros(8, device=device, dtype=dtype))
        for sx, sy in [(1,0), (-1,0), (0,1), (0,-1)]:
            p = torch.tensor([sx, sy, sx, sy, sx, sy, sx, sy], device=device, dtype=dtype) * r
            patterns.append(p)
        # Mild perspective/scale-like perturbations.
        patterns.append(torch.tensor([-1,-1, 1,-1, 1,1, -1,1], device=device, dtype=dtype) * r)
        patterns.append(torch.tensor([ 1, 1,-1, 1,-1,-1,  1,-1], device=device, dtype=dtype) * r)
        patterns.append(torch.tensor([-1, 0, 1, 0, 1,0, -1,0], device=device, dtype=dtype) * r)
        patterns.append(torch.tensor([ 0,-1, 0,-1, 0,1,  0,1], device=device, dtype=dtype) * r)
        P = torch.stack(patterns[:self.max_candidates], dim=0)  # [C,8]
        return offsets[:, None, :] + P[None, :, :]

    def forward(self, *, offsets: torch.Tensor, h4p: torch.Tensor, ia_full: torch.Tensor,
                patch_indices: torch.Tensor, patch_h: int, patch_w: int,
                feature_fn, target_feature: torch.Tensor, support: torch.Tensor) -> dict:
        cand_offsets = self._candidate_offsets(offsets)  # [B,C,8]
        b, c, _ = cand_offsets.shape
        src = h4p.reshape(b, 4, 2).to(device=offsets.device, dtype=offsets.dtype)
        src_rep = src[:, None].expand(-1, c, -1, -1).reshape(b * c, 4, 2)
        dst = src_rep + cand_offsets.reshape(b * c, 4, 2)
        H_cand = solve_homography_dlt(src_rep, dst).reshape(b, c, 3, 3)

        ia_rep = ia_full[:, None].expand(-1, c, -1, -1, -1).reshape(b * c, *ia_full.shape[1:])
        idx_rep = patch_indices[:, None].expand(-1, c, -1).reshape(b * c, -1)
        H_rep = H_cand.reshape(b * c, 3, 3)
        pred = transform_official_patch(ia_rep, H_rep, idx_rep, patch_h, patch_w)
        feat = feature_fn(pred).reshape(b, c, *target_feature.shape[1:])
        target = target_feature[:, None].expand_as(feat)
        residual = (feat.float() - target.float()).abs().mean(dim=2, keepdim=True)  # [B,C,1,H,W]
        sup = support[:, None].float().clamp(0, 1)
        energy = (residual * sup).flatten(2).sum(dim=2) / sup.flatten(2).sum(dim=2).clamp_min(1e-6)
        weights = torch.softmax(-energy / max(self.temperature, 1e-6), dim=1)
        refined_offsets = (weights[..., None] * cand_offsets).sum(dim=1)
        dst_refined = src + refined_offsets.reshape(b, 4, 2)
        H_refined = solve_homography_dlt(src, dst_refined)
        return {
            'candidate_H': H_cand,
            'candidate_offsets': cand_offsets,
            'candidate_energy': energy,
            'candidate_weights': weights,
            'refined_offsets': refined_offsets,
            'H_refined': H_refined,
        }
