#!/usr/bin/env python
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.video_pair_dataset import LabeledPointPairsDataset
from src.data.transforms import OFFICIAL_IMG_H, OFFICIAL_IMG_W, OFFICIAL_PATCH_H, OFFICIAL_PATCH_W
from src.utils.checkpoint import load_checkpoint
from src_v2.models.v2_model import V2MultiHypothesisHomographyNet
from src_v2.engine.evaluator_v2 import evaluate_labeled_points_v2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--npy_dir', required=True)
    ap.add_argument('--image_root', required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--num_hypotheses', type=int, default=4)
    ap.add_argument('--img_h', type=int, default=OFFICIAL_IMG_H)
    ap.add_argument('--img_w', type=int, default=OFFICIAL_IMG_W)
    ap.add_argument('--patch_h', type=int, default=OFFICIAL_PATCH_H)
    ap.add_argument('--patch_w', type=int, default=OFFICIAL_PATCH_W)
    ap.add_argument('--crop_x', type=int, default=40)
    ap.add_argument('--crop_y', type=int, default=23)
    ap.add_argument('--max_points', type=int, default=6)
    args = ap.parse_args()

    device = torch.device(args.device if args.device == 'cpu' or torch.cuda.is_available() else 'cpu')
    model = V2MultiHypothesisHomographyNet(num_hypotheses=args.num_hypotheses).to(device)
    load_checkpoint(args.ckpt, model, map_location=device)
    ds = LabeledPointPairsDataset(
        args.npy_dir, args.image_root, args.patch_h, args.patch_w,
        args.img_h, args.img_w, args.crop_x, args.crop_y,
    )
    print(evaluate_labeled_points_v2(model, ds, device, max_points=args.max_points))


if __name__ == '__main__':
    main()
