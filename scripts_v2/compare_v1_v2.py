#!/usr/bin/env python
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models.content_aware_homography import ContentAwareHomographyNet
from src.engine.evaluator import evaluate_labeled_points
from src.utils.checkpoint import load_checkpoint
from src.data.video_pair_dataset import LabeledPointPairsDataset
from src_v2.models.v2_model import V2MultiHypothesisHomographyNet
from src_v2.engine.evaluator_v2 import evaluate_labeled_points_v2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--v1_ckpt', required=True)
    ap.add_argument('--v2_ckpt', required=True)
    ap.add_argument('--npy_dir', default='dataset/val_labels')
    ap.add_argument('--image_root', default='dataset/val_images')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--num_hypotheses', type=int, default=4)
    args = ap.parse_args()

    device = torch.device(args.device if args.device == 'cpu' or torch.cuda.is_available() else 'cpu')
    ds = LabeledPointPairsDataset(args.npy_dir, args.image_root)

    v1 = ContentAwareHomographyNet().to(device)
    load_checkpoint(args.v1_ckpt, v1, map_location=device)
    print('V1:', evaluate_labeled_points(v1, ds, device))

    v2 = V2MultiHypothesisHomographyNet(num_hypotheses=args.num_hypotheses).to(device)
    load_checkpoint(args.v2_ckpt, v2, map_location=device)
    print('V2:', evaluate_labeled_points_v2(v2, ds, device))


if __name__ == '__main__':
    main()
