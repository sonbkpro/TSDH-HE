from __future__ import annotations
from pathlib import Path
import sys
import torch
torch.set_num_threads(2)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src_v3.models.dominant_homography_v3 import DominantHomographyV3Net
from src.data.video_pair_dataset import build_oneline_sample
import cv2


def main():
    root = Path(__file__).resolve().parents[1]
    img1 = cv2.imread(str(root / 'dataset/val_images/0000011_10001.jpg'))
    img2 = cv2.imread(str(root / 'dataset/val_images/0000011_10005.jpg'))
    sample = build_oneline_sample(img1, img2, 32, 48, None, img_h=64, img_w=96, rho=4, crop_xy=(8, 8))
    model = DominantHomographyV3Net(consensus_radius=0.5)
    model.eval()
    with torch.no_grad():
        out = model.forward_oneline(
            sample['org_images'].unsqueeze(0), sample['input_tensors'].unsqueeze(0),
            sample['h4p'].unsqueeze(0), sample['patch_indices'].unsqueeze(0),
            use_attention=True, use_mask_weighting=True, use_consensus=True,
        )
    print({
        'ok': True,
        'H_shape': tuple(out['H'].shape),
        'H_init_shape': tuple(out['H_init'].shape),
        'offsets_shape': tuple(out['offsets'].shape),
        'support_shape': tuple(out['support_ap'].shape),
        'confidence_shape': tuple(out['confidence'].shape),
        'candidate_weights_shape': tuple(out.get('candidate_weights', torch.empty(0)).shape),
    })


if __name__ == '__main__':
    main()
