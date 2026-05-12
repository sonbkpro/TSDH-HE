import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import torch
from src.models.content_aware_homography import ContentAwareHomographyNet
from src.data.transforms import make_h4p, make_patch_indices


def test_forward_shapes():
    m = ContentAwareHomographyNet()
    org = torch.rand(1, 2, 96, 128)
    x, y, ph, pw = 16, 16, 64, 96
    out = m.forward_oneline(
        org,
        org[:, :, y:y + ph, x:x + pw],
        make_h4p(x, y, ph, pw).unsqueeze(0),
        make_patch_indices(x, y, ph, pw, 128).unsqueeze(0),
        use_attention=True,
        use_mask_weighting=True,
    )
    assert out['H'].shape == (1,3,3)
    assert out['Ma'].shape == (1,1,64,96)
    assert out['pred_ib'].shape == (1,1,64,96)
