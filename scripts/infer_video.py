#!/usr/bin/env python
from __future__ import annotations
import argparse, sys, cv2, torch
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.models.content_aware_homography import ContentAwareHomographyNet
from src.utils.checkpoint import load_checkpoint
from src.data.video_pair_dataset import build_oneline_sample
from src.data.transforms import OFFICIAL_IMG_H, OFFICIAL_IMG_W, OFFICIAL_PATCH_H, OFFICIAL_PATCH_W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True); ap.add_argument('--video', required=True)
    ap.add_argument('--gap', type=int, default=1); ap.add_argument('--max_pairs', type=int, default=100)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--img_h', type=int, default=OFFICIAL_IMG_H)
    ap.add_argument('--img_w', type=int, default=OFFICIAL_IMG_W)
    ap.add_argument('--patch_h', type=int, default=OFFICIAL_PATCH_H)
    ap.add_argument('--patch_w', type=int, default=OFFICIAL_PATCH_W)
    ap.add_argument('--crop_x', type=int, default=40)
    ap.add_argument('--crop_y', type=int, default=23)
    args = ap.parse_args()
    device = torch.device(args.device if args.device == 'cpu' or torch.cuda.is_available() else 'cpu')
    model = ContentAwareHomographyNet().to(device).eval(); load_checkpoint(args.ckpt, model, map_location=device)
    cap = cv2.VideoCapture(args.video)
    frames = []
    while len(frames) < args.max_pairs + args.gap:
        ok, f = cap.read()
        if not ok: break
        frames.append(f)
    cap.release()
    for i in range(min(args.max_pairs, len(frames)-args.gap)):
        sample = build_oneline_sample(
            frames[i], frames[i + args.gap], args.patch_h, args.patch_w, None, args.img_h, args.img_w,
            crop_xy=(args.crop_x, args.crop_y),
        )
        org_images = sample['org_images'].unsqueeze(0).to(device)
        input_tensors = sample['input_tensors'].unsqueeze(0).to(device)
        h4p = sample['h4p'].unsqueeze(0).to(device)
        patch_indices = sample['patch_indices'].unsqueeze(0).to(device)
        with torch.no_grad():
            H = model.forward_oneline(org_images, input_tensors, h4p, patch_indices, use_attention=True, use_mask_weighting=True)['H']
            H_ab = torch.linalg.inv(H)[0].cpu().numpy()
        print(i, '->', i+args.gap, H_ab.reshape(-1).tolist())

if __name__ == '__main__': main()
