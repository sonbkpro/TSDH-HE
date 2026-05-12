import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import torch
from src.geometry.warp import warp_perspective


def test_warp_half_inputs_use_float32_geometry():
    src = torch.rand(1, 1, 8, 8, dtype=torch.float16)
    H = torch.eye(3, dtype=torch.float16).unsqueeze(0)
    warped = warp_perspective(src, H)
    assert warped.dtype == torch.float16
    assert torch.allclose(warped.float(), src.float(), atol=1e-3)
