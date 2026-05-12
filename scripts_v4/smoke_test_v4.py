from __future__ import annotations
from pathlib import Path
import sys
import torch
import cv2
torch.set_num_threads(2)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.video_pair_dataset import build_oneline_sample
from src_v4.models.tsdh_net import TSDHNet


def main():
    root = Path(__file__).resolve().parents[1]
    img1 = cv2.imread(str(root / 'dataset/val_images/0000011_10001.jpg'))
    img2 = cv2.imread(str(root / 'dataset/val_images/0000011_10005.jpg'))
    if img1 is None or img2 is None:
        raise FileNotFoundError('Smoke-test images are missing from dataset/val_images')
    sample = build_oneline_sample(img1, img2, 32, 48, None, img_h=64, img_w=96, rho=4, crop_xy=(8, 8))
    model = TSDHNet(pretrained_backbone=False)
    model.eval()
    with torch.no_grad():
        out = model.forward_pair(
            sample['org_images'].unsqueeze(0), sample['input_tensors'].unsqueeze(0),
            sample['h4p'].unsqueeze(0), sample['patch_indices'].unsqueeze(0),
            use_attention=True, use_mask_weighting=True, use_temporal_support=True,
        )
    print({
        'ok': True,
        'H_shape': tuple(out['H'].shape),
        'H_init_shape': tuple(out['H_init'].shape),
        'support_shape': tuple(out['support_temporal'].shape),
        'nonh_shape': tuple(out['nonh_map'].shape),
        'residual_shape': tuple(out['residual_final'].shape),
    })


if __name__ == '__main__':
    main()
