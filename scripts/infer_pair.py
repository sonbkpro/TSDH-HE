#!/usr/bin/env python
from __future__ import annotations
import argparse, sys
from pathlib import Path
import cv2, torch
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.models.content_aware_homography import ContentAwareHomographyNet
from src.utils.checkpoint import load_checkpoint
from src.data.video_pair_dataset import build_oneline_sample
from src.data.transforms import OFFICIAL_IMG_H, OFFICIAL_IMG_W, OFFICIAL_PATCH_H, OFFICIAL_PATCH_W
from src.geometry.warp import warp_official_full
from src.utils.visualization import make_alignment_overlay, save_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--image_a', required=True)
    ap.add_argument('--image_b', required=True)
    ap.add_argument('--out', default='alignment_overlay.png')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--img_h', type=int, default=OFFICIAL_IMG_H)
    ap.add_argument('--img_w', type=int, default=OFFICIAL_IMG_W)
    ap.add_argument('--patch_h', type=int, default=OFFICIAL_PATCH_H)
    ap.add_argument('--patch_w', type=int, default=OFFICIAL_PATCH_W)
    ap.add_argument('--crop_x', type=int, default=40)
    ap.add_argument('--crop_y', type=int, default=23)
    args = ap.parse_args()
    device = torch.device(args.device if args.device == 'cpu' or torch.cuda.is_available() else 'cpu')
    model = ContentAwareHomographyNet().to(device).eval()
    load_checkpoint(args.ckpt, model, map_location=device)
    img_a, img_b = cv2.imread(args.image_a), cv2.imread(args.image_b)
    if img_a is None or img_b is None:
        raise FileNotFoundError(f'Could not read {args.image_a} or {args.image_b}')
    sample = build_oneline_sample(
        img_a, img_b, args.patch_h, args.patch_w, None, args.img_h, args.img_w,
        crop_xy=(args.crop_x, args.crop_y),
    )
    org_images = sample['org_images'].unsqueeze(0).to(device)
    input_tensors = sample['input_tensors'].unsqueeze(0).to(device)
    h4p = sample['h4p'].unsqueeze(0).to(device)
    patch_indices = sample['patch_indices'].unsqueeze(0).to(device)
    with torch.no_grad():
        out = model.forward_oneline(org_images, input_tensors, h4p, patch_indices, use_attention=True, use_mask_weighting=True)
        wa = warp_official_full(org_images[:, :1], out['H'])
    save_image(args.out, make_alignment_overlay(wa, org_images[:, 1:]))
    print('H_dst_to_src_sampling=')
    print(out['H'][0].detach().cpu().numpy())
    print('H_ab_point_transform=')
    print(torch.linalg.inv(out['H'])[0].detach().cpu().numpy())
    print(f'saved {args.out}')

if __name__ == '__main__': main()
