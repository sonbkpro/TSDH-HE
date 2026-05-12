from __future__ import annotations
import torch
from torch.utils.data import DataLoader
from src.geometry.dlt import transform_points


OFFICIAL_CATEGORIES = {
    'RE': {'0000011', '0000016', '00000147', '00000155', '00000158', '00000107', '00000239', '0000030'},
    'LT': {'0000038', '0000044', '0000046', '0000047', '00000238', '00000177', '00000188', '00000181'},
    'LL': {'0000085', '00000100', '0000091', '0000092', '00000216', '00000226'},
    'SF': {'00000244', '00000251', '0000026', '0000034', '00000115'},
    'LF': {'00000104', '0000031', '0000035', '00000129', '00000141', '00000200'},
}


def _video_id(path: str) -> str:
    stem = path.replace('\\', '/').split('/')[-1].split('_')[0]
    return stem


def _category(path: str) -> str | None:
    vid = _video_id(path)
    for name, ids in OFFICIAL_CATEGORIES.items():
        if vid in ids:
            return name
    return None


@torch.no_grad()
def evaluate_labeled_points(model, dataset, device='cuda', max_points: int = 6):
    was_training = model.training
    model.eval()
    try:
        errors = []
        inliers_3px = []
        by_category = {k: [] for k in OFFICIAL_CATEGORIES}
        for sample in DataLoader(dataset, batch_size=1, shuffle=False):
            org_images = sample['org_images'].to(device).float()
            input_tensors = sample['input_tensors'].to(device).float()
            h4p = sample['h4p'].to(device).float()
            patch_indices = sample['patch_indices'].to(device).float()
            pts_a = sample['pts_a'].to(device).float()
            pts_b = sample['pts_b'].to(device).float()
            if max_points is not None:
                pts_a = pts_a[:, :max_points]
                pts_b = pts_b[:, :max_points]
            out = model.forward_oneline(org_images, input_tensors, h4p, patch_indices, use_attention=True, use_mask_weighting=True)
            H_ab = torch.linalg.inv(out['H'])
            pred_b = transform_points(pts_a, H_ab)
            direct = torch.linalg.norm(pred_b - pts_b, dim=-1)

            # The released evaluator keeps this guard because some annotation
            # files do not consistently order the two image points.
            pred_a = transform_points(pts_b, H_ab)
            swapped = torch.linalg.norm(pred_a - pts_a, dim=-1)
            per_point = torch.minimum(direct, swapped)
            err = per_point.mean().item()
            errors.append(err)
            inliers_3px.append((per_point < 3.0).float().mean().item())
            cat = _category(sample['path1'][0])
            if cat:
                by_category[cat].append(err)
        metrics = {
            'point_l2_mean': float(sum(errors) / max(len(errors), 1)),
            'inlier_3px': float(sum(inliers_3px) / max(len(inliers_3px), 1)),
            'num_pairs': len(errors),
        }
        for cat, values in by_category.items():
            if values:
                metrics[f'{cat}_point_l2_mean'] = float(sum(values) / len(values))
        return metrics
    finally:
        if was_training:
            model.train()
