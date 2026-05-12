import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import torch
from src.models.content_aware_homography import ContentAwareHomographyNet
from src.losses.triplet_homography_loss import ContentAwareTripletLoss
from src.data.transforms import make_h4p, make_patch_indices
from src_v3.losses.v3_losses import V3DominantHomographyLoss


def make_batch():
    org = torch.rand(1, 2, 96, 128)
    x, y, ph, pw = 16, 16, 64, 96
    return {
        'org_images': org,
        'input_tensors': org[:, :, y:y + ph, x:x + pw],
        'h4p': make_h4p(x, y, ph, pw).unsqueeze(0),
        'patch_indices': make_patch_indices(x, y, ph, pw, 128).unsqueeze(0),
    }


def test_loss_backward():
    m = ContentAwareHomographyNet()
    batch = make_batch()
    out = m.forward_oneline(**batch, use_attention=True, use_mask_weighting=False)
    losses = ContentAwareTripletLoss()(out)
    losses['loss'].backward()
    assert torch.isfinite(losses['loss']).item()


def test_stage1_uses_unit_loss_mask():
    m = ContentAwareHomographyNet()
    batch = make_batch()
    out = m.forward_oneline(**batch, use_attention=True, use_mask_weighting=False)
    assert torch.all(out['mask_ap'] == 1)


def test_triplet_loss_is_nonnegative():
    criterion = ContentAwareTripletLoss()
    anchor = torch.zeros(1, 1, 4, 4)
    positive = torch.full((1, 1, 4, 4), 3.0)
    negative = torch.ones(1, 1, 4, 4)
    loss_map = criterion.loss_map(anchor, positive, negative)
    assert torch.all(loss_map >= 0.0)


def test_v3_init_triplet_uses_initial_warp_tensor():
    criterion = V3DominantHomographyLoss()
    b, c, h, w = 2, 1, 4, 4
    out = {
        'Fa': torch.ones(b, c, h, w),
        'Fb': torch.zeros(b, c, h, w),
        'pred_ib': torch.zeros(b, c, h, w),
        'pred_ib_feature': torch.full((b, c, h, w), 0.25),
        'init_pred_ib': torch.full((b, c, h, w), 0.5),
        'init_pred_ib_feature': torch.full((b, c, h, w), 0.5),
        'mask_ap': torch.ones(b, 1, h, w),
        'support_ap': torch.ones(b, 1, h, w),
        'support_init': torch.ones(b, 1, h, w),
        'offset_logvar': torch.zeros(b, 8),
    }
    losses = criterion.pair_losses(out)
    assert torch.isfinite(losses['loss']).item()
    assert torch.isfinite(losses['init_triplet']).item()
