#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import sys

import cv2
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.transforms import OFFICIAL_IMG_H, OFFICIAL_IMG_W, OFFICIAL_PATCH_H, OFFICIAL_PATCH_W
from src.data.video_pair_dataset import build_oneline_sample
from src.geometry.warp import warp_official_full
from src.utils.checkpoint import load_checkpoint
from src.utils.visualization import make_alignment_overlay, save_image, tensor_gray_to_uint8
from src_v3.models.dominant_homography_v3 import DominantHomographyV3Net


MODEL_MAP_OUTPUTS = (
    ('Sa', 'support_a_patch'),
    ('Sb', 'support_b_patch'),
    ('support_init', 'support_init'),
    ('support_ap', 'support_final'),
    ('sa_full', 'support_a_full'),
    ('sb_full', 'support_b_full'),
)


def _checkpoint_model_cfg(path: str) -> dict:
    ckpt = torch.load(path, map_location='cpu')
    if not isinstance(ckpt, dict):
        return {}
    cfg = ckpt.get('config', {})
    if not isinstance(cfg, dict):
        return {}
    model_cfg = cfg.get('model', {})
    return model_cfg if isinstance(model_cfg, dict) else {}


def _pick(value, cfg: dict, key: str, default, cast):
    if value is None:
        value = cfg.get(key, default)
    return cast(value)


def _save_model_maps(out_prefix: str, pred: dict) -> None:
    for key, suffix in MODEL_MAP_OUTPUTS:
        if key in pred:
            save_image(f'{out_prefix}_{suffix}.png', tensor_gray_to_uint8(pred[key]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--image_a', required=True)
    ap.add_argument('--image_b', required=True)
    ap.add_argument('--out_prefix', default='vis')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--img_h', type=int, default=OFFICIAL_IMG_H)
    ap.add_argument('--img_w', type=int, default=OFFICIAL_IMG_W)
    ap.add_argument('--patch_h', type=int, default=OFFICIAL_PATCH_H)
    ap.add_argument('--patch_w', type=int, default=OFFICIAL_PATCH_W)
    ap.add_argument('--crop_x', type=int, default=40)
    ap.add_argument('--crop_y', type=int, default=23)
    ap.add_argument('--feature_channels', type=int, default=None)
    ap.add_argument('--consensus_radius', type=float, default=None)
    ap.add_argument('--consensus_temperature', type=float, default=None)
    ap.add_argument('--consensus_max_candidates', type=int, default=None)
    ap.add_argument('--no_consensus', action='store_true')
    args = ap.parse_args()

    device = torch.device(args.device if args.device == 'cpu' or torch.cuda.is_available() else 'cpu')
    model_cfg = _checkpoint_model_cfg(args.ckpt)
    model = DominantHomographyV3Net(
        feature_channels=_pick(args.feature_channels, model_cfg, 'feature_channels', 1, int),
        pretrained_backbone=False,
        consensus_radius=_pick(args.consensus_radius, model_cfg, 'consensus_radius', 2.0, float),
        consensus_temperature=_pick(args.consensus_temperature, model_cfg, 'consensus_temperature', 0.15, float),
        consensus_max_candidates=_pick(args.consensus_max_candidates, model_cfg, 'consensus_max_candidates', 5, int),
    ).to(device).eval()
    load_checkpoint(args.ckpt, model, map_location=device)

    img_a, img_b = cv2.imread(args.image_a), cv2.imread(args.image_b)
    if img_a is None or img_b is None:
        raise FileNotFoundError(f'Could not read {args.image_a} or {args.image_b}')
    sample = build_oneline_sample(
        img_a, img_b, args.patch_h, args.patch_w, None, args.img_h, args.img_w,
        crop_xy=(args.crop_x, args.crop_y),
    )
    org_images = sample['org_images'].unsqueeze(0).to(device).float()
    input_tensors = sample['input_tensors'].unsqueeze(0).to(device).float()
    h4p = sample['h4p'].unsqueeze(0).to(device).float()
    patch_indices = sample['patch_indices'].unsqueeze(0).to(device).float()

    with torch.no_grad():
        pred = model.forward_oneline(
            org_images, input_tensors, h4p, patch_indices,
            use_attention=True, use_mask_weighting=True, use_consensus=not args.no_consensus,
        )
        warped = warp_official_full(org_images[:, :1], pred['H'])

    save_image(args.out_prefix + '_overlay.png', make_alignment_overlay(warped, org_images[:, 1:]))
    save_image(args.out_prefix + '_mask_a.png', tensor_gray_to_uint8(pred['Ma']))
    save_image(args.out_prefix + '_mask_b.png', tensor_gray_to_uint8(pred['Mb']))
    save_image(args.out_prefix + '_warped_a.png', tensor_gray_to_uint8(warped))
    _save_model_maps(args.out_prefix, pred)


if __name__ == '__main__':
    main()
