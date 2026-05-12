from __future__ import annotations
from pathlib import Path
import sys
import torch
import cv2
torch.set_num_threads(2)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.video_pair_dataset import build_oneline_sample
from src_v4.models.tsdh_net import TSDHNet
from src_v4.losses.tsdh_losses import TSDHLoss


def _sample(a, b):
    return build_oneline_sample(a, b, 32, 48, None, img_h=64, img_w=96, rho=4, crop_xy=(8, 8))


def _batch(s):
    return {k: v.unsqueeze(0) for k, v in s.items() if torch.is_tensor(v)}


def main():
    root = Path(__file__).resolve().parents[1]
    imgs = [cv2.imread(str(root / f'dataset/val_images/0000011_1000{i}.jpg')) for i in (1, 3, 5)]
    if any(x is None for x in imgs):
        # fallback: reuse the available pair as a degenerate triplet
        imgs = [cv2.imread(str(root / 'dataset/val_images/0000011_10001.jpg')),
                cv2.imread(str(root / 'dataset/val_images/0000011_10005.jpg')),
                cv2.imread(str(root / 'dataset/val_images/0000011_10005.jpg'))]
    p01, p12, p02 = _batch(_sample(imgs[0], imgs[1])), _batch(_sample(imgs[1], imgs[2])), _batch(_sample(imgs[0], imgs[2]))
    model = TSDHNet(pretrained_backbone=False)
    crit = TSDHLoss(weights={'pixel_cycle_support': 0.02, 'homography_cycle': 0.002})
    model.eval()
    with torch.no_grad():
        out = model.forward_triplet(p01, p12, p02, use_attention=True, use_mask_weighting=True, use_temporal_support=True)
        losses = crit(out, mode='triplet')
    print({
        'ok': True,
        'H01_shape': tuple(out['out01']['H'].shape),
        'cycle_residual_shape': tuple(out['cycle_residual'].shape),
        'loss': float(losses['loss']),
        'pixel_cycle_support': float(losses['pixel_cycle_support']),
    })


if __name__ == '__main__':
    main()
