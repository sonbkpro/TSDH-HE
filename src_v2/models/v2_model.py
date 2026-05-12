from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.feature_extractor import FeatureExtractor
from src.models.mask_predictor import MaskPredictor, normalize_mask
from src.geometry.warp import gather_patch_from_full, transform_official_patch
from .multi_estimator import ResNet34MultiHomographyEstimator


class AssignmentMaskHead(nn.Module):
    """Predicts K soft region-assignment masks from both image features.

    The softmax over K forces every pixel to be explained by one hypothesis.
    This is different from V1's single content-aware mask, which only rejects
    unreliable content.
    """
    def __init__(self, in_ch: int, num_hypotheses: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_hypotheses, 3, 1, 1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(x.float()), dim=1)


class V2MultiHypothesisHomographyNet(nn.Module):
    """V2: K homographies + K assignment masks + dominant selection.

    V1 is intentionally untouched. This model reuses only low-level V1 building
    blocks so V1/V2 comparisons stay clean.
    """
    def __init__(self, feature_channels: int = 1, num_hypotheses: int = 4):
        super().__init__()
        self.num_hypotheses = int(num_hypotheses)
        self.feature = FeatureExtractor(1, feature_channels)
        self.reliability = MaskPredictor(1)
        pair_ch = 4 * feature_channels + 2  # Fa, Fb, |Fa-Fb|, Fa*Fb, Ia, Ib
        self.assignment = AssignmentMaskHead(pair_ch, self.num_hypotheses)
        self.estimator = ResNet34MultiHomographyEstimator(2 * feature_channels, self.num_hypotheses)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.001)
                nn.init.zeros_(module.bias)

    def _encode_patch(self, ia_patch: torch.Tensor, ib_patch: torch.Tensor):
        fa = self.feature(ia_patch)
        fb = self.feature(ib_patch)
        pair = torch.cat([fa, fb, (fa - fb).abs(), fa * fb, ia_patch, ib_patch], dim=1)
        assignments = self.assignment(pair)
        return fa, fb, assignments

    def forward_oneline(self, org_images, input_tensors, h4p, patch_indices,
                        use_attention: bool = True, use_mask_weighting: bool = True):
        if org_images.ndim != 4 or input_tensors.ndim != 4:
            raise ValueError('org_images and input_tensors must be BCHW tensors')
        b, _, patch_h, patch_w = input_tensors.shape
        K = self.num_hypotheses

        ia_full = org_images[:, :1]
        ib_full = org_images[:, 1:]
        ia_patch = input_tensors[:, :1]
        ib_patch = input_tensors[:, 1:]

        ma_full = self.reliability(ia_full)
        mb_full = self.reliability(ib_full)
        ma_patch = normalize_mask(gather_patch_from_full(ma_full, patch_indices, patch_h, patch_w))
        mb_patch = normalize_mask(gather_patch_from_full(mb_full, patch_indices, patch_h, patch_w))

        fa_patch, fb_patch, assignments = self._encode_patch(ia_patch, ib_patch)
        ga_patch = fa_patch * ma_patch if use_attention else fa_patch
        gb_patch = fb_patch * mb_patch if use_attention else fb_patch

        Hs, offsets = self.estimator(torch.cat([ga_patch, gb_patch], dim=1), h4p=h4p)

        pred_patches = []
        pred_mask_patches = []
        pred_features = []
        for k in range(K):
            pred_ib_k = transform_official_patch(ia_full, Hs[:, k], patch_indices, patch_h, patch_w)
            pred_ma_k = transform_official_patch(ma_full, Hs[:, k], patch_indices, patch_h, patch_w)
            pred_ma_k = normalize_mask(pred_ma_k)
            pred_patches.append(pred_ib_k)
            pred_mask_patches.append(pred_ma_k)
            pred_features.append(self.feature(pred_ib_k))

        pred_ib = torch.stack(pred_patches, dim=1)           # [B,K,1,H,W]
        pred_ma = torch.stack(pred_mask_patches, dim=1)     # [B,K,1,H,W]
        pred_feat = torch.stack(pred_features, dim=1)       # [B,K,C,H,W]

        # Assignment masks explain different planar/dynamic regions; V1-style
        # reliability masks reject low-confidence pixels inside each assignment.
        assign_5d = assignments.unsqueeze(2)                # [B,K,1,H,W]
        mask_ap = assign_5d * mb_patch.unsqueeze(1) * pred_ma
        if not use_mask_weighting:
            mask_ap = assign_5d

        # Differentiable consensus scores from current residuals.
        with torch.no_grad():
            fb_expand = fb_patch.unsqueeze(1)
            residual = (pred_feat - fb_expand).abs().sum(dim=2, keepdim=True)
            denom = mask_ap.sum(dim=(2, 3, 4)).clamp_min(1e-6)
            score = (mask_ap * torch.exp(-residual)).sum(dim=(2, 3, 4)) / denom
            dominant_index = torch.argmax(score, dim=1)

        H_dominant = Hs[torch.arange(b, device=Hs.device), dominant_index]
        offsets_dominant = offsets[torch.arange(b, device=offsets.device), dominant_index]

        return {
            'Hs': Hs,
            'offsets': offsets,
            'scores': score,
            'dominant_index': dominant_index,
            'H_dominant': H_dominant,
            'offsets_dominant': offsets_dominant,
            'Fa': fa_patch,
            'Fb': fb_patch,
            'Ma': ma_patch,
            'Mb': mb_patch,
            'Ga': ga_patch,
            'Gb': gb_patch,
            'assignments': assignments,
            'pred_ib': pred_ib,
            'pred_ib_feature': pred_feat,
            'mask_ap': mask_ap,
            'ia_patch': ia_patch,
            'ib_patch': ib_patch,
            'ma_full': ma_full,
            'mb_full': mb_full,
        }

    def forward_pair(self, ia_patch: torch.Tensor, ib_patch: torch.Tensor):
        fa, fb, assignments = self._encode_patch(ia_patch, ib_patch)
        ma = normalize_mask(self.reliability(ia_patch))
        mb = normalize_mask(self.reliability(ib_patch))
        Hs, offsets = self.estimator(torch.cat([fa * ma, fb * mb], dim=1), h4p=None)
        return {'Hs': Hs, 'offsets': offsets, 'assignments': assignments, 'Fa': fa, 'Fb': fb, 'Ma': ma, 'Mb': mb}
