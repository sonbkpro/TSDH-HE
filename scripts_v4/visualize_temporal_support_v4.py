from __future__ import annotations
import argparse
from pathlib import Path
import sys
import cv2
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.video_pair_dataset import build_oneline_sample
from src.utils.checkpoint import load_checkpoint
from src.utils.visualization import save_image, tensor_gray_to_uint8, make_alignment_overlay
from src_v4.models.tsdh_net import TSDHNet


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--image_a', required=True)
    p.add_argument('--image_b', required=True)
    p.add_argument('--out_prefix', default='v4_vis')
    p.add_argument('--device', default='cuda')
    p.add_argument('--img_h', type=int, default=360)
    p.add_argument('--img_w', type=int, default=640)
    p.add_argument('--patch_h', type=int, default=315)
    p.add_argument('--patch_w', type=int, default=560)
    p.add_argument('--crop_x', type=int, default=40)
    p.add_argument('--crop_y', type=int, default=23)
    args = p.parse_args()
    device = torch.device(args.device if args.device == 'cpu' or torch.cuda.is_available() else 'cpu')
    img_a, img_b = cv2.imread(args.image_a), cv2.imread(args.image_b)
    if img_a is None or img_b is None:
        raise FileNotFoundError('Could not read input image_a or image_b')
    sample = build_oneline_sample(img_a, img_b, args.patch_h, args.patch_w, None,
                                  args.img_h, args.img_w, crop_xy=(args.crop_x, args.crop_y))
    model = TSDHNet(pretrained_backbone=False).to(device)
    load_checkpoint(args.ckpt, model, map_location=device)
    model.eval()
    with torch.no_grad():
        out = model.forward_pair(
            sample['org_images'].unsqueeze(0).to(device).float(),
            sample['input_tensors'].unsqueeze(0).to(device).float(),
            sample['h4p'].unsqueeze(0).to(device).float(),
            sample['patch_indices'].unsqueeze(0).to(device).float(),
            use_attention=True, use_mask_weighting=True, use_temporal_support=True,
        )
    prefix = args.out_prefix
    save_image(f'{prefix}_support_temporal.png', tensor_gray_to_uint8(out['support_temporal'].cpu()))
    save_image(f'{prefix}_nonhomographic.png', tensor_gray_to_uint8(out['nonh_map'].cpu()))
    save_image(f'{prefix}_residual_final.png', tensor_gray_to_uint8(out['residual_final'].cpu()))
    overlay = make_alignment_overlay(out['pred_ib'].cpu(), out['ib_patch'].cpu())
    save_image(f'{prefix}_alignment_overlay.png', overlay)
    print({'saved_prefix': prefix})


if __name__ == '__main__':
    main()
