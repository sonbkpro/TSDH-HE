from __future__ import annotations
from typing import Dict
import torch
import torch.nn as nn
from src.models.feature_extractor import FeatureExtractor
from src.models.mask_predictor import MaskPredictor, normalize_mask
from src.models.homography_estimator import ResNet34HomographyEstimator
from src.geometry.warp import gather_patch_from_full, transform_official_patch
from .temporal_support_refiner import TemporalSupportRefiner, normalize_residual_map
from .nonhomographic_residual_head import NonHomographicResidualHead


def _compose_official(H01: torch.Tensor, H12: torch.Tensor) -> torch.Tensor:
    # Official matrices are sampling transforms target->source. The existing V3
    # convention uses H02 ~= H01 @ H12.
    return H01.float().bmm(H12.float())


class TSDHNet(nn.Module):
    """V4 / TSDH-Net: Temporal Support-Decomposed Homography Network.

    V1 code is left untouched. V4 reuses the same core V1 modules and adds only:
    - residual/cycle-driven TemporalSupportRefiner;
    - NonHomographicResidualHead;
    - support-decomposed second-pass single-H estimator.

    The method still outputs one dominant global homography. It does not use K
    global homographies and does not claim to solve all planes exactly.
    """
    def __init__(self, feature_channels: int = 1, pretrained_backbone: bool = True,
                 support_hidden: int = 16, nonh_hidden: int = 16,
                 support_floor: float = 0.15):
        super().__init__()
        self.feature_channels = int(feature_channels)
        self.support_floor = float(support_floor)
        self.feature = FeatureExtractor(1, feature_channels)
        self.mask = MaskPredictor(1)  # V1 pairwise content-aware prior
        self.estimator = ResNet34HomographyEstimator(2 * feature_channels, pretrained_backbone=False)
        self.temporal_support = TemporalSupportRefiner(in_ch=5, hidden=support_hidden)
        self.nonh_head = NonHomographicResidualHead(in_ch=4, hidden=nonh_hidden)
        self._init_added_weights()
        if pretrained_backbone:
            self.estimator.load_imagenet_backbone()

    def _init_added_weights(self):
        # Keep V1-style initialization for all conv/bn modules. ImageNet weights
        # are loaded for the ResNet estimator afterwards if requested.
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, org_images, input_tensors, h4p, patch_indices,
                use_attention: bool = True, use_mask_weighting: bool = True,
                use_temporal_support: bool = True, use_final_estimator: bool = True,
                safe_gate: bool = True):
        return self.forward_pair(
            org_images=org_images, input_tensors=input_tensors, h4p=h4p, patch_indices=patch_indices,
            use_attention=use_attention, use_mask_weighting=use_mask_weighting,
            use_temporal_support=use_temporal_support, use_final_estimator=use_final_estimator,
            safe_gate=safe_gate,
        )

    def _unpack_pair(self, org_images, input_tensors):
        if org_images.ndim != 4 or input_tensors.ndim != 4:
            raise ValueError('org_images and input_tensors must be BCHW tensors')
        ia_full = org_images[:, :1]
        ib_full = org_images[:, 1:]
        ia_patch = input_tensors[:, :1]
        ib_patch = input_tensors[:, 1:]
        return ia_full, ib_full, ia_patch, ib_patch

    def _initial_pair(self, org_images, input_tensors, h4p, patch_indices,
                      use_attention: bool = True, use_mask_weighting: bool = True) -> Dict[str, torch.Tensor]:
        _, _, patch_h, patch_w = input_tensors.shape
        ia_full, ib_full, ia_patch, ib_patch = self._unpack_pair(org_images, input_tensors)

        ma_full = self.mask(ia_full)
        mb_full = self.mask(ib_full)
        ma_patch = normalize_mask(gather_patch_from_full(ma_full, patch_indices, patch_h, patch_w))
        mb_patch = normalize_mask(gather_patch_from_full(mb_full, patch_indices, patch_h, patch_w))

        fa_patch = self.feature(ia_patch)
        fb_patch = self.feature(ib_patch)
        ga_patch = fa_patch * ma_patch if use_attention else fa_patch
        gb_patch = fb_patch * mb_patch if use_attention else fb_patch
        H_init, offsets_init = self.estimator(torch.cat([ga_patch, gb_patch], dim=1), h4p=h4p)

        init_pred_ib = transform_official_patch(ia_full, H_init, patch_indices, patch_h, patch_w)
        init_pred_feature = self.feature(init_pred_ib)
        pred_ma = normalize_mask(transform_official_patch(ma_full, H_init, patch_indices, patch_h, patch_w))
        support_init = mb_patch * pred_ma
        if not use_mask_weighting:
            support_init = torch.ones_like(support_init)
        residual_init = (fb_patch.float() - init_pred_feature.float()).abs().mean(dim=1, keepdim=True)
        return {
            'H_init': H_init, 'offsets_init': offsets_init,
            'Fa': fa_patch, 'Fb': fb_patch, 'Ma': ma_patch, 'Mb': mb_patch,
            'Ga_init': ga_patch, 'Gb_init': gb_patch,
            'init_pred_ib': init_pred_ib, 'init_pred_ib_feature': init_pred_feature,
            'support_init': support_init, 'residual_init': residual_init,
            'ia_full': ia_full, 'ib_full': ib_full, 'ia_patch': ia_patch, 'ib_patch': ib_patch,
            'ma_full': ma_full, 'mb_full': mb_full,
            'patch_indices': patch_indices, 'h4p': h4p,
        }

    def _support_refine_pair(self, init: Dict[str, torch.Tensor]) -> torch.Tensor:
        r = normalize_residual_map(init['residual_init'])
        zeros = torch.zeros_like(r)
        # Pair fallback: no real temporal evidence is available, so repeated pair
        # residuals are used and cycle residual is zero. The same refiner is still
        # used at validation on two-image labels.
        evidence = torch.cat([init['support_init'].float(), r, r, r, zeros], dim=1)
        return self.temporal_support(evidence)

    def _effective_support(self, support: torch.Tensor) -> torch.Tensor:
        """Map raw support to a non-zero effective support used by the estimator.

        Raw support is kept for visualization and support-specific losses.  The
        effective support is used for feature attention and mask weighting so the
        homography estimator never receives near-empty feature maps.
        """
        s_raw = support.float().clamp(0, 1)
        floor = max(0.0, min(float(self.support_floor), 0.95))
        return floor + (1.0 - floor) * s_raw

    def _build_nonh_map(self, residual_final: torch.Tensor, residual_init: torch.Tensor,
                        support_raw: torch.Tensor, support_eff: torch.Tensor) -> torch.Tensor:
        # NonH evidence is residual-driven.  It intentionally does not use
        # 1-support as a training target, avoiding the positive feedback loop
        # support->0, nonH->1, support->0.
        nonh_evidence = torch.cat([
            normalize_residual_map(residual_final),
            normalize_residual_map(residual_init),
            support_raw.float().clamp(0, 1),
            support_eff.float().clamp(0, 1),
        ], dim=1)
        return self.nonh_head(nonh_evidence)

    def _return_init_with_support_analysis(self, init: Dict[str, torch.Tensor], support: torch.Tensor,
                                           use_mask_weighting: bool = True) -> Dict[str, torch.Tensor]:
        """Use H_init as the output while still training/visualizing support.

        Stage 2 uses this path: support and nonH are learned from residual
        evidence, but the final support-decomposed estimator is not allowed to
        update/corrupt the shared V1 estimator.
        """
        s_raw = support.float().clamp(0, 1)
        s_eff = self._effective_support(s_raw)
        # Detach the triplet mask so triplet loss cannot minimize itself by
        # shrinking support. Support is learned by support-specific losses.
        mask = s_eff.detach() if use_mask_weighting else torch.ones_like(s_eff)
        nonh = self._build_nonh_map(init['residual_init'], init['residual_init'], s_raw, s_eff)
        out = dict(init)
        out.update({
            'H': init['H_init'],
            'offsets': init['offsets_init'],
            'Ga': init['Ga_init'],
            'Gb': init['Gb_init'],
            'pred_ib': init['init_pred_ib'],
            'pred_ib_feature': init['init_pred_ib_feature'],
            'mask_ap': mask,
            'support_temporal': s_raw,
            'support_effective': s_eff,
            'support_ap': mask,
            'support_final': mask,
            'residual_final': init['residual_init'],
            'nonh_map': nonh,
            'H_final_raw': init['H_init'],
            'offsets_final_raw': init['offsets_init'],
            'used_final_gate_mean': torch.zeros((), device=s_raw.device, dtype=s_raw.dtype),
            'ma_full': init['ma_full'],
            'mb_full': init['mb_full'],
        })
        return out

    def _final_from_support(self, init: Dict[str, torch.Tensor], support: torch.Tensor,
                            use_attention: bool = True, use_mask_weighting: bool = True,
                            safe_gate: bool = True) -> Dict[str, torch.Tensor]:
        patch_h, patch_w = init['ia_patch'].shape[-2:]
        s_raw = support.float().clamp(0, 1)
        s_eff = self._effective_support(s_raw)
        if use_attention:
            ga = init['Fa'] * s_eff
            gb = init['Fb'] * s_eff
        else:
            ga, gb = init['Fa'], init['Fb']
        H_final, offsets_final = self.estimator(torch.cat([ga, gb], dim=1), h4p=init['h4p'])

        pred_final = transform_official_patch(init['ia_full'], H_final, init['patch_indices'], patch_h, patch_w)
        feat_final = self.feature(pred_final)
        residual_final_raw = (init['Fb'].float() - feat_final.float()).abs().mean(dim=1, keepdim=True)

        H_out, offsets_out = H_final, offsets_final
        pred_ib, pred_feature = pred_final, feat_final
        residual_final = residual_final_raw
        gate = torch.ones(H_final.shape[0], 1, 1, 1, device=H_final.device, dtype=H_final.dtype)
        if safe_gate:
            # Per-sample residual gate: never prefer the refined branch if it is
            # worse than the stable V1-style initial branch under the same
            # effective support.  This is intentionally hard/detached for safety.
            with torch.no_grad():
                dims = (1, 2, 3)
                score_final = (residual_final_raw * s_eff).flatten(1).sum(dim=1) / s_eff.flatten(1).sum(dim=1).clamp_min(1e-6)
                score_init = (init['residual_init'].float() * s_eff).flatten(1).sum(dim=1) / s_eff.flatten(1).sum(dim=1).clamp_min(1e-6)
                use_final = (score_final <= score_init).view(-1, 1, 1, 1)
                gate = use_final.to(dtype=H_final.dtype)
            H_out = torch.where(gate.view(-1, 1, 1).bool(), H_final, init['H_init'])
            offsets_out = torch.where(gate.view(-1, 1).bool(), offsets_final, init['offsets_init'])
            pred_ib = transform_official_patch(init['ia_full'], H_out, init['patch_indices'], patch_h, patch_w)
            pred_feature = self.feature(pred_ib)
            residual_final = (init['Fb'].float() - pred_feature.float()).abs().mean(dim=1, keepdim=True)

        # Detach the mask used by triplet loss: support must not learn the
        # degenerate all-zero shortcut through the triplet denominator.
        support_final = s_eff.detach() if use_mask_weighting else torch.ones_like(s_eff)
        nonh = self._build_nonh_map(residual_final, init['residual_init'], s_raw, s_eff)
        out = dict(init)
        out.update({
            'H': H_out, 'offsets': offsets_out,
            'H_final_raw': H_final, 'offsets_final_raw': offsets_final,
            'Ga': ga, 'Gb': gb,
            'pred_ib': pred_ib, 'pred_ib_feature': pred_feature,
            'mask_ap': support_final,
            'support_temporal': s_raw,
            'support_effective': s_eff,
            'support_ap': support_final,
            'support_final': support_final,
            'residual_final': residual_final,
            'residual_final_raw': residual_final_raw,
            'nonh_map': nonh,
            'used_final_gate_mean': gate.mean().detach(),
            # V1 aliases for tooling.
            'ma_full': init['ma_full'], 'mb_full': init['mb_full'],
        })
        return out

    def _return_init_as_final(self, init: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Return the V1-style initial branch as the final output.

        This is intentionally used during Stage 1 warm-up. In V1, the mask is
        used as attention for estimating H, while mask weighting in the loss can
        be disabled. Earlier V4 versions still ran a second support-decomposed
        estimator when temporal support was disabled, which made Stage 1 *not*
        equivalent to V1 and caused unstable validation: H_init was good but H
        was worse.
        """
        out = dict(init)
        z = torch.zeros_like(init['support_init'])
        out.update({
            'H': init['H_init'],
            'offsets': init['offsets_init'],
            'Ga': init['Ga_init'],
            'Gb': init['Gb_init'],
            'pred_ib': init['init_pred_ib'],
            'pred_ib_feature': init['init_pred_ib_feature'],
            'mask_ap': init['support_init'],
            'support_temporal': init['support_init'],
            'support_ap': init['support_init'],
            'support_final': init['support_init'],
            'residual_final': init['residual_init'],
            # Non-homographic branch is inactive in true V1 warm-up.
            'nonh_map': z,
            'ma_full': init['ma_full'],
            'mb_full': init['mb_full'],
        })
        return out

    def forward_pair(self, org_images, input_tensors, h4p, patch_indices,
                     use_attention: bool = True, use_mask_weighting: bool = True,
                     use_temporal_support: bool = True, use_final_estimator: bool = True,
                     safe_gate: bool = True) -> Dict[str, torch.Tensor]:
        init = self._initial_pair(org_images, input_tensors, h4p, patch_indices, use_attention, use_mask_weighting)
        # Critical stability fix: Stage 1 must be a true V1-style warm-up.
        # When temporal support is disabled, do not train/evaluate the second
        # support-decomposed estimator. Use H_init as H.
        if not use_temporal_support:
            return self._return_init_as_final(init)
        support = self._support_refine_pair(init)
        if not use_final_estimator:
            return self._return_init_with_support_analysis(init, support, use_mask_weighting=use_mask_weighting)
        return self._final_from_support(init, support, use_attention=use_attention,
                                        use_mask_weighting=use_mask_weighting, safe_gate=safe_gate)

    def _pair_from_dict(self, p: dict, use_attention: bool, use_mask_weighting: bool) -> Dict[str, torch.Tensor]:
        return self._initial_pair(p['org_images'], p['input_tensors'], p['h4p'], p['patch_indices'],
                                  use_attention=use_attention, use_mask_weighting=use_mask_weighting)

    def forward_triplet(self, p01: dict, p12: dict, p02: dict,
                        use_attention: bool = True, use_mask_weighting: bool = True,
                        use_temporal_support: bool = True, use_final_estimator: bool = True,
                        safe_gate: bool = True) -> dict:
        init01 = self._pair_from_dict(p01, use_attention, use_mask_weighting)
        init12 = self._pair_from_dict(p12, use_attention, use_mask_weighting)
        init02 = self._pair_from_dict(p02, use_attention, use_mask_weighting)

        # Pixel-level temporal cycle evidence on the same crop.
        _, _, patch_h, patch_w = p02['input_tensors'].shape
        H_comp = _compose_official(init01['H_init'], init12['H_init']).to(dtype=init01['H_init'].dtype)
        comp_pred = transform_official_patch(init01['ia_full'], H_comp, p02['patch_indices'], patch_h, patch_w)
        comp_feature = self.feature(comp_pred)
        cycle_residual = (comp_feature.float() - init02['init_pred_ib_feature'].float()).abs().mean(dim=1, keepdim=True)

        r01 = normalize_residual_map(init01['residual_init'])
        r12 = normalize_residual_map(init12['residual_init'])
        r02 = normalize_residual_map(init02['residual_init'])
        c = normalize_residual_map(cycle_residual)

        if use_temporal_support:
            s01 = self.temporal_support(torch.cat([init01['support_init'].float(), r01, r02, c, r12], dim=1))
            s12 = self.temporal_support(torch.cat([init12['support_init'].float(), r12, r02, c, r01], dim=1))
            s02 = self.temporal_support(torch.cat([init02['support_init'].float(), r02, r01, c, r12], dim=1))
        else:
            s01, s12, s02 = init01['support_init'], init12['support_init'], init02['support_init']

        if use_temporal_support and not use_final_estimator:
            out01 = self._return_init_with_support_analysis(init01, s01, use_mask_weighting=use_mask_weighting)
            out12 = self._return_init_with_support_analysis(init12, s12, use_mask_weighting=use_mask_weighting)
            out02 = self._return_init_with_support_analysis(init02, s02, use_mask_weighting=use_mask_weighting)
        else:
            out01 = self._final_from_support(init01, s01, use_attention, use_mask_weighting, safe_gate=safe_gate)
            out12 = self._final_from_support(init12, s12, use_attention, use_mask_weighting, safe_gate=safe_gate)
            out02 = self._final_from_support(init02, s02, use_attention, use_mask_weighting, safe_gate=safe_gate)
        return {
            'out01': out01, 'out12': out12, 'out02': out02,
            'cycle_residual': cycle_residual,
            'comp_pred_ib': comp_pred,
            'comp_pred_feature': comp_feature,
            'H_comp_init': H_comp,
        }
