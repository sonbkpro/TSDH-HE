from __future__ import annotations
import torch
import torch.nn as nn
from src.models.feature_extractor import FeatureExtractor
from src.models.mask_predictor import MaskPredictor, normalize_mask
from src.models.homography_estimator import ResNet34HomographyEstimator
from src.geometry.warp import gather_patch_from_full, transform_official_patch
from .uncertainty_head import HomographyUncertaintyHead
from .consensus_refiner import LocalConsensusRefiner


class DominantHomographyV3Net(nn.Module):
    """V3: one dominant global homography + temporal static support + consensus + uncertainty.

    V1 modules are reused exactly: FeatureExtractor, MaskPredictor, ResNet34 estimator.
    V3 adds lightweight heads around the single-H formulation instead of K homographies.
    """
    def __init__(self, feature_channels: int = 1, pretrained_backbone: bool = False,
                 consensus_radius: float = 2.0, consensus_temperature: float = 0.15, consensus_max_candidates: int = 5):
        super().__init__()
        self.feature = FeatureExtractor(1, feature_channels)
        self.support = MaskPredictor(1)  # temporal/static support starts from V1 mask architecture
        self.estimator = ResNet34HomographyEstimator(2 * feature_channels, pretrained_backbone=False)
        self.uncertainty = HomographyUncertaintyHead(2 * feature_channels)
        self.consensus = LocalConsensusRefiner(radius=consensus_radius, temperature=consensus_temperature, max_candidates=consensus_max_candidates)
        self._init_weights()
        if pretrained_backbone:
            self.estimator.load_imagenet_backbone()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


    def forward(self, org_images, input_tensors, h4p, patch_indices,
                use_attention: bool = True, use_mask_weighting: bool = True,
                use_consensus: bool = True):
        """PyTorch/DataParallel entry point.

        TrainerV3 calls the model as self.model(...). Without this method,
        nn.Module.forward is not implemented and DataParallel raises:
        _forward_unimplemented() got an unexpected keyword argument 'org_images'.
        """
        return self.forward_oneline(
            org_images=org_images,
            input_tensors=input_tensors,
            h4p=h4p,
            patch_indices=patch_indices,
            use_attention=use_attention,
            use_mask_weighting=use_mask_weighting,
            use_consensus=use_consensus,
        )

    def forward_oneline(self, org_images, input_tensors, h4p, patch_indices,
                        use_attention: bool = True, use_mask_weighting: bool = True,
                        use_consensus: bool = True):
        if org_images.ndim != 4 or input_tensors.ndim != 4:
            raise ValueError('org_images and input_tensors must be BCHW tensors')
        _, _, patch_h, patch_w = input_tensors.shape
        ia_full = org_images[:, :1]
        ib_full = org_images[:, 1:]
        ia_patch = input_tensors[:, :1]
        ib_patch = input_tensors[:, 1:]

        sa_full = self.support(ia_full)
        sb_full = self.support(ib_full)
        sa_patch = normalize_mask(gather_patch_from_full(sa_full, patch_indices, patch_h, patch_w))
        sb_patch = normalize_mask(gather_patch_from_full(sb_full, patch_indices, patch_h, patch_w))

        fa_patch = self.feature(ia_patch)
        fb_patch = self.feature(ib_patch)
        ga_patch = fa_patch * sa_patch if use_attention else fa_patch
        gb_patch = fb_patch * sb_patch if use_attention else fb_patch
        pair_features = torch.cat([ga_patch, gb_patch], dim=1)
        H_init, offsets_init = self.estimator(pair_features, h4p=h4p)
        unc = self.uncertainty(pair_features)

        init_pred_ib = transform_official_patch(ia_full, H_init, patch_indices, patch_h, patch_w)
        init_pred_support = normalize_mask(transform_official_patch(sa_full, H_init, patch_indices, patch_h, patch_w))
        support_ap = sb_patch * init_pred_support
        if not use_mask_weighting:
            support_ap = torch.ones_like(support_ap)

        if use_consensus:
            ref = self.consensus(
                offsets=offsets_init, h4p=h4p, ia_full=ia_full, patch_indices=patch_indices,
                patch_h=patch_h, patch_w=patch_w, feature_fn=self.feature,
                target_feature=fb_patch, support=support_ap,
            )
            H = ref['H_refined']
            offsets = ref['refined_offsets']
        else:
            ref = {}
            H = H_init
            offsets = offsets_init

        pred_ib = transform_official_patch(ia_full, H, patch_indices, patch_h, patch_w)
        pred_support = normalize_mask(transform_official_patch(sa_full, H, patch_indices, patch_h, patch_w))
        support_final = sb_patch * pred_support
        if not use_mask_weighting:
            support_final = torch.ones_like(support_final)
        pred_ib_feature = self.feature(pred_ib)
        init_pred_ib_feature = self.feature(init_pred_ib)

        out = {
            'H': H,
            'offsets': offsets,
            'H_init': H_init,
            'offsets_init': offsets_init,
            'Fa': fa_patch,
            'Fb': fb_patch,
            'Sa': sa_patch,
            'Sb': sb_patch,
            'Ma': sa_patch,  # aliases for V1-style tools
            'Mb': sb_patch,
            'Ga': ga_patch,
            'Gb': gb_patch,
            'pred_ib': pred_ib,
            'pred_ib_feature': pred_ib_feature,
            'init_pred_ib': init_pred_ib,
            'init_pred_ib_feature': init_pred_ib_feature,
            'mask_ap': support_final,
            'support_ap': support_final,
            'support_init': support_ap,
            'ia_patch': ia_patch,
            'ib_patch': ib_patch,
            'sa_full': sa_full,
            'sb_full': sb_full,
            'ma_full': sa_full,
            'mb_full': sb_full,
            'offset_logvar': unc['offset_logvar'],
            'confidence': unc['confidence'],
        }
        out.update(ref)
        return out
