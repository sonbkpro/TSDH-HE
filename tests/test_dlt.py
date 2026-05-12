import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import torch
from src.geometry.dlt import canonical_corners, solve_homography_dlt, transform_points


def test_dlt_identity():
    pts = canonical_corners(2, 64, 96)
    H = solve_homography_dlt(pts, pts)
    eye = torch.eye(3).unsqueeze(0).repeat(2,1,1)
    assert torch.allclose(H, eye, atol=1e-4)


def test_transform_translation():
    pts = torch.tensor([[[0.,0.],[10.,0.],[10.,10.],[0.,10.]]])
    dst = pts + torch.tensor([[[5.,3.]]])
    H = solve_homography_dlt(pts, dst)
    out = transform_points(pts, H)
    assert torch.allclose(out, dst, atol=1e-4)


def test_dlt_half_inputs_solve_in_float32():
    pts = canonical_corners(1, 64, 96, dtype=torch.float16)
    H = solve_homography_dlt(pts, pts)
    eye = torch.eye(3).unsqueeze(0)
    assert H.dtype == torch.float32
    assert torch.allclose(H, eye, atol=1e-4)
