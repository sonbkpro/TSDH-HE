from __future__ import annotations
import torch
import torch.nn as nn
from .feature_extractor import FeatureExtractor
from .mask_predictor import MaskPredictor, normalize_mask
from .homography_estimator import ResNet34HomographyEstimator
from src.geometry.warp import gather_patch_from_full, transform_official_patch


class ContentAwareHomographyNet(nn.Module):
    def __init__(self, feature_channels: int = 1, pretrained_backbone: bool = False):
        super().__init__()
        self.feature = FeatureExtractor(1, feature_channels)
        self.mask = MaskPredictor(1)
        self.estimator = ResNet34HomographyEstimator(2 * feature_channels, pretrained_backbone=False)
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

    def encode(self, x):
        F = self.feature(x)
        M = normalize_mask(self.mask(x))
        return F, M

    def forward_oneline(self, org_images, input_tensors, h4p, patch_indices,
                        use_attention: bool = True, use_mask_weighting: bool = True):
        """Official released Oneline forward path.

        ``org_images`` is [B,2,H,W], ``input_tensors`` is the cropped
        [B,2,patch_h,patch_w] pair, and ``h4p``/``patch_indices`` are full-image
        patch geometry from the dataset.
        """
        if org_images.ndim != 4 or input_tensors.ndim != 4:
            raise ValueError('org_images and input_tensors must be BCHW tensors')
        _, _, patch_h, patch_w = input_tensors.shape
        ia_full = org_images[:, :1]
        ib_full = org_images[:, 1:]
        ia_patch = input_tensors[:, :1]
        ib_patch = input_tensors[:, 1:]

        ma_full = self.mask(ia_full)
        mb_full = self.mask(ib_full)
        ma_patch = normalize_mask(gather_patch_from_full(ma_full, patch_indices, patch_h, patch_w))
        mb_patch = normalize_mask(gather_patch_from_full(mb_full, patch_indices, patch_h, patch_w))

        fa_patch = self.feature(ia_patch)
        fb_patch = self.feature(ib_patch)
        ga_patch = fa_patch * ma_patch if use_attention else fa_patch
        gb_patch = fb_patch * mb_patch if use_attention else fb_patch
        H_dst_to_src, offsets = self.estimator(torch.cat([ga_patch, gb_patch], dim=1), h4p=h4p)

        pred_ib_patch = transform_official_patch(ia_full, H_dst_to_src, patch_indices, patch_h, patch_w)
        pred_ma_patch = transform_official_patch(ma_full, H_dst_to_src, patch_indices, patch_h, patch_w)
        pred_ma_patch = normalize_mask(pred_ma_patch)
        mask_ap = mb_patch * pred_ma_patch
        if not use_mask_weighting:
            mask_ap = torch.ones_like(mask_ap)

        pred_ib_feature = self.feature(pred_ib_patch)
        return {
            'H': H_dst_to_src,
            'offsets': offsets,
            'Fa': fa_patch,
            'Fb': fb_patch,
            'Ma': ma_patch,
            'Mb': mb_patch,
            'Ga': ga_patch,
            'Gb': gb_patch,
            'pred_ib': pred_ib_patch,
            'pred_ib_feature': pred_ib_feature,
            'mask_ap': mask_ap,
            'ia_patch': ia_patch,
            'ib_patch': ib_patch,
            'ma_full': ma_full,
            'mb_full': mb_full,
        }

    def estimate(self, ia, ib, use_attention: bool = True):
        Fa, Ma = self.encode(ia)
        Fb, Mb = self.encode(ib)
        Ga = Fa * Ma if use_attention else Fa
        Gb = Fb * Mb if use_attention else Fb
        Hab, off = self.estimator(torch.cat([Ga, Gb], dim=1))
        return {'H': Hab, 'offsets': off, 'Fa': Fa, 'Fb': Fb, 'Ma': Ma, 'Mb': Mb, 'Ga': Ga, 'Gb': Gb}

    def forward(self, ia, ib, use_attention: bool = True, bidirectional: bool = True):
        ab = self.estimate(ia, ib, use_attention=use_attention)
        out = {'ab': ab}
        if bidirectional:
            out['ba'] = self.estimate(ib, ia, use_attention=use_attention)
        return out
