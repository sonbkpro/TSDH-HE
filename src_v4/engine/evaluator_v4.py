from __future__ import annotations
import torch
from torch.utils.data import DataLoader
from src.geometry.dlt import transform_points


@torch.no_grad()
def evaluate_labeled_points_v4(model, dataset, device='cuda', max_points: int | None = 6,
                               use_temporal_support: bool = True):
    was_training = model.training
    model.eval()
    try:
        errors, init_errors, inliers3, supports, nonhs = [], [], [], [], []
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
            out = model.forward_pair(org_images, input_tensors, h4p, patch_indices,
                                     use_attention=True, use_mask_weighting=True,
                                     use_temporal_support=use_temporal_support)
            for key, acc in [('H', errors), ('H_init', init_errors)]:
                H_ab = torch.linalg.inv(out[key].float())
                pred_b = transform_points(pts_a, H_ab)
                direct = torch.linalg.norm(pred_b - pts_b, dim=-1)

                # Match the V1 evaluator exactly: some validation annotation
                # files may not order points consistently, so keep the same
                # direct/swapped guard used by src.engine.evaluator.
                pred_a = transform_points(pts_b, H_ab)
                swapped = torch.linalg.norm(pred_a - pts_a, dim=-1)
                err = torch.minimum(direct, swapped)

                acc.append(err.mean().item())
                if key == 'H':
                    inliers3.append((err < 3.0).float().mean().item())
            supports.append(out['support_temporal'].mean().item())
            nonhs.append(out['nonh_map'].mean().item())
        n = max(len(errors), 1)
        return {
            'point_l2_mean': float(sum(errors) / n),
            'init_point_l2_mean': float(sum(init_errors) / n),
            'refine_gain': float((sum(init_errors) - sum(errors)) / n),
            'inlier_3px': float(sum(inliers3) / max(len(inliers3), 1)),
            'support_mean': float(sum(supports) / max(len(supports), 1)),
            'nonh_mean': float(sum(nonhs) / max(len(nonhs), 1)),
            'num_pairs': len(errors),
        }
    finally:
        if was_training:
            model.train()
