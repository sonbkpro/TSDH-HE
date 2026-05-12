from __future__ import annotations
import torch


_LOW_PRECISION_DTYPES = (torch.float16, torch.bfloat16)


def _linalg_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.float32 if dtype in _LOW_PRECISION_DTYPES else dtype


def canonical_corners(batch: int, h: int, w: int, device=None, dtype=None) -> torch.Tensor:
    pts = torch.tensor([[0., 0.], [w - 1., 0.], [w - 1., h - 1.], [0., h - 1.]], device=device, dtype=dtype)
    return pts.unsqueeze(0).repeat(batch, 1, 1)


def solve_homography_dlt(src_pts: torch.Tensor, dst_pts: torch.Tensor) -> torch.Tensor:
    """Differentiable DLT for exactly 4 or more correspondences.
    src_pts, dst_pts: [B, N, 2] in pixel coordinates. Returns H [B,3,3] mapping src -> dst.
    """
    if src_pts.shape != dst_pts.shape or src_pts.ndim != 3 or src_pts.size(-1) != 2:
        raise ValueError('src_pts and dst_pts must be [B,N,2] and have identical shape')
    b, n, _ = src_pts.shape
    if n < 4:
        raise ValueError('At least 4 correspondences are required')
    solve_dtype = _linalg_dtype(src_pts.dtype)
    src = src_pts.to(solve_dtype)
    dst = dst_pts.to(solve_dtype)
    x, y = src[..., 0], src[..., 1]
    u, v = dst[..., 0], dst[..., 1]
    zeros = torch.zeros_like(x)
    ones = torch.ones_like(x)
    row1 = torch.stack([x, y, ones, zeros, zeros, zeros, -u * x, -u * y], dim=-1)
    row2 = torch.stack([zeros, zeros, zeros, x, y, ones, -v * x, -v * y], dim=-1)
    A = torch.stack([row1, row2], dim=2).reshape(b, 2 * n, 8)
    rhs = torch.stack([u, v], dim=2).reshape(b, 2 * n, 1)
    try:
        h8 = torch.linalg.solve(A, rhs).squeeze(-1) if n == 4 else torch.linalg.lstsq(A, rhs).solution.squeeze(-1)
    except RuntimeError:
        h8 = torch.linalg.pinv(A) @ rhs
        h8 = h8.squeeze(-1)
    last = torch.ones(b, 1, device=src_pts.device, dtype=h8.dtype)
    H = torch.cat([h8, last], dim=1).reshape(b, 3, 3)
    return H


def offsets_to_homography(offsets: torch.Tensor, h: int, w: int) -> torch.Tensor:
    if offsets.ndim != 2 or offsets.size(1) != 8:
        raise ValueError('offsets must be [B,8]')
    src = canonical_corners(offsets.size(0), h, w, offsets.device, offsets.dtype)
    dst = src + offsets.reshape(-1, 4, 2)
    return solve_homography_dlt(src, dst)


def transform_points(points: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    """Transform [B,N,2] points by H [B,3,3]."""
    b, n, _ = points.shape
    ones = torch.ones(b, n, 1, device=points.device, dtype=points.dtype)
    ph = torch.cat([points, ones], dim=-1)
    out = ph @ H.transpose(1, 2)
    denom = out[..., 2:]
    eps = torch.finfo(denom.dtype).eps
    denom = torch.where(denom.abs() < eps, denom.sign().clamp(min=0).mul(2).sub(1) * eps, denom)
    return out[..., :2] / denom
