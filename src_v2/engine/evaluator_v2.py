from __future__ import annotations
import torch
from torch.utils.data import DataLoader
from src.geometry.dlt import transform_points


@torch.no_grad()
def evaluate_labeled_points_v2(model, dataset, device='cuda', max_points: int | None = 6):
    was_training = model.training
    model.eval()
    try:
        dom_errors, best_errors, dom_inliers, best_inliers = [], [], [], []
        active = []
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

            out = model.forward_oneline(org_images, input_tensors, h4p, patch_indices,
                                        use_attention=True, use_mask_weighting=True)
            Hs_ab = torch.linalg.inv(out['Hs'])
            K = Hs_ab.size(1)
            errs = []
            inls = []
            for k in range(K):
                pred_b = transform_points(pts_a, Hs_ab[:, k])
                direct = torch.linalg.norm(pred_b - pts_b, dim=-1)
                # Keep the V1 guard for inconsistent label ordering.
                pred_a = transform_points(pts_b, Hs_ab[:, k])
                swapped = torch.linalg.norm(pred_a - pts_a, dim=-1)
                per_point = torch.minimum(direct, swapped)
                errs.append(per_point.mean(dim=1))
                inls.append((per_point < 3.0).float().mean(dim=1))
            err = torch.stack(errs, dim=1)
            inl = torch.stack(inls, dim=1)
            dom_idx = out['dominant_index']
            dom_err = err.gather(1, dom_idx[:, None]).squeeze(1)
            best_err, best_idx = err.min(dim=1)
            dom_errors.extend(dom_err.cpu().tolist())
            best_errors.extend(best_err.cpu().tolist())
            dom_inliers.extend(inl.gather(1, dom_idx[:, None]).squeeze(1).cpu().tolist())
            best_inliers.extend(inl.max(dim=1).values.cpu().tolist())
            active.extend(best_idx.cpu().tolist())

        n = max(len(dom_errors), 1)
        return {
            'dominant_point_l2_mean': float(sum(dom_errors) / n),
            'best_of_k_point_l2_mean': float(sum(best_errors) / n),
            'dominant_vs_best_gap': float((sum(dom_errors) - sum(best_errors)) / n),
            'dominant_inlier_3px': float(sum(dom_inliers) / n),
            'best_of_k_inlier_3px': float(sum(best_inliers) / n),
            'num_pairs': len(dom_errors),
            'active_experts': int(len(set(active))) if active else 0,
        }
    finally:
        if was_training:
            model.train()
