from __future__ import annotations
import argparse
from pathlib import Path
import sys
import cv2
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.video_pair_dataset import build_oneline_sample
from src.utils.checkpoint import load_checkpoint
from src_v3.models.dominant_homography_v3 import DominantHomographyV3Net


def to_img(t):
    a = t.detach().cpu().squeeze().numpy()
    a = np.clip(a * 255, 0, 255).astype(np.uint8)
    return a


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--image_a', required=True)
    p.add_argument('--image_b', required=True)
    p.add_argument('--out', default='v3_static_support.png')
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    device = torch.device(args.device if args.device == 'cpu' or torch.cuda.is_available() else 'cpu')
    img1, img2 = cv2.imread(args.image_a), cv2.imread(args.image_b)
    sample = build_oneline_sample(img1, img2, 315, 560, None, crop_xy=(40, 23))
    model = DominantHomographyV3Net().to(device)
    load_checkpoint(args.ckpt, model, map_location=device)
    model.eval()
    with torch.no_grad():
        out = model.forward_oneline(
            sample['org_images'].unsqueeze(0).to(device).float(),
            sample['input_tensors'].unsqueeze(0).to(device).float(),
            sample['h4p'].unsqueeze(0).to(device).float(),
            sample['patch_indices'].unsqueeze(0).to(device).float(),
        )
    ia = to_img(out['ia_patch'][0])
    ib = to_img(out['ib_patch'][0])
    pred = to_img(out['pred_ib'][0])
    sup = to_img(out['support_ap'][0])
    canvas = np.concatenate([ia, ib, pred, sup], axis=1)
    cv2.imwrite(args.out, canvas)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
