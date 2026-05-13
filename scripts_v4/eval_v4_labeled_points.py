from __future__ import annotations
import argparse
from pathlib import Path
import sys
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.video_pair_dataset import LabeledPointPairsDataset
from src.utils.checkpoint import load_checkpoint
from src_v4.models.tsdh_net import TSDHNet
from src_v4.engine.evaluator_v4 import evaluate_labeled_points_v4


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--npy_dir', default='dataset/val_labels')
    p.add_argument('--image_root', default='dataset/val_images')
    p.add_argument('--crop_h', type=int, default=315)
    p.add_argument('--crop_w', type=int, default=560)
    p.add_argument('--img_h', type=int, default=360)
    p.add_argument('--img_w', type=int, default=640)
    p.add_argument('--eval_crop_x', type=int, default=40)
    p.add_argument('--eval_crop_y', type=int, default=23)
    p.add_argument('--max_points', type=int, default=6)
    p.add_argument('--device', default='cuda')
    p.add_argument('--no_temporal_support', action='store_true')
    p.add_argument('--no_final_estimator', action='store_true',
                   help='Evaluate H_init with support/nonH analysis, matching safe Stage 2 behavior.')
    p.add_argument('--disable_safe_gate', action='store_true')
    args = p.parse_args()
    device = torch.device(args.device if args.device == 'cpu' or torch.cuda.is_available() else 'cpu')
    model = TSDHNet(pretrained_backbone=False).to(device)
    load_checkpoint(args.ckpt, model, map_location=device)
    ds = LabeledPointPairsDataset(args.npy_dir, args.image_root, args.crop_h, args.crop_w,
                                  args.img_h, args.img_w, args.eval_crop_x, args.eval_crop_y)
    print(evaluate_labeled_points_v4(
        model, ds, device, max_points=args.max_points,
        use_temporal_support=not args.no_temporal_support,
        use_final_estimator=not args.no_final_estimator,
        safe_gate=not args.disable_safe_gate,
    ))


if __name__ == '__main__':
    main()
