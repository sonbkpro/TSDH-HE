from __future__ import annotations
from contextlib import nullcontext
import torch
import torch.nn.functional as F


_LOW_PRECISION_DTYPES = (torch.float16, torch.bfloat16)


def _geometry_dtype(*tensors: torch.Tensor) -> torch.dtype:
    dtype = tensors[0].dtype
    for tensor in tensors[1:]:
        dtype = torch.promote_types(dtype, tensor.dtype)
    return torch.float32 if dtype in _LOW_PRECISION_DTYPES else dtype


def _disable_autocast(device_type: str):
    if device_type in {'cuda', 'cpu'}:
        return torch.amp.autocast(device_type, enabled=False)
    return nullcontext()


def warp_perspective(src: torch.Tensor, H_src_to_dst: torch.Tensor, out_h: int | None = None, out_w: int | None = None) -> torch.Tensor:
    """Pure PyTorch STN-style inverse warping.
    src: [B,C,H,W]. H maps source pixel coordinates to destination pixel coordinates.
    Returns source warped into the destination canvas.
    """
    if src.ndim != 4:
        raise ValueError('src must be [B,C,H,W]')
    b, _, h, w = src.shape
    out_h = h if out_h is None else out_h
    out_w = w if out_w is None else out_w
    compute_dtype = _geometry_dtype(src, H_src_to_dst)
    with _disable_autocast(src.device.type):
        H = H_src_to_dst.to(compute_dtype)
        ys, xs = torch.meshgrid(
            torch.arange(out_h, device=src.device, dtype=compute_dtype),
            torch.arange(out_w, device=src.device, dtype=compute_dtype),
            indexing='ij',
        )
        ones = torch.ones_like(xs)
        dst = torch.stack([xs, ys, ones], dim=-1).reshape(1, out_h * out_w, 3).repeat(b, 1, 1)
        H_inv = torch.linalg.inv(H)
        src_xyw = dst @ H_inv.transpose(1, 2)
        denom = src_xyw[..., 2]
        eps = torch.finfo(denom.dtype).eps
        denom = torch.where(denom.abs() < eps, denom.sign().clamp(min=0).mul(2).sub(1) * eps, denom)
        x = src_xyw[..., 0] / denom
        y = src_xyw[..., 1] / denom
        x_norm = 2.0 * x / max(w - 1, 1) - 1.0
        y_norm = 2.0 * y / max(h - 1, 1) - 1.0
        grid = torch.stack([x_norm, y_norm], dim=-1).reshape(b, out_h, out_w, 2)
        warped = F.grid_sample(src.to(compute_dtype), grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    return warped.to(src.dtype)


def gather_patch_from_full(full: torch.Tensor, patch_indices: torch.Tensor, patch_h: int, patch_w: int) -> torch.Tensor:
    if full.ndim != 4:
        raise ValueError('full must be [B,C,H,W]')
    b, c, h, w = full.shape
    indices = patch_indices.to(device=full.device, dtype=torch.long)
    if indices.ndim == 1:
        indices = indices.unsqueeze(0).expand(b, -1)
    if indices.shape != (b, patch_h * patch_w):
        raise ValueError(f'patch_indices must be [B,{patch_h * patch_w}] or [{patch_h * patch_w}]')
    flat = full.reshape(b, c, h * w)
    gathered = torch.gather(flat, 2, indices.unsqueeze(1).expand(-1, c, -1))
    return gathered.reshape(b, c, patch_h, patch_w)


def _pixel_to_norm_matrix(batch: int, h: int, w: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    m = torch.tensor(
        [[w / 2.0, 0.0, w / 2.0], [0.0, h / 2.0, h / 2.0], [0.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    )
    m_inv = torch.inverse(m)
    return m.unsqueeze(0).expand(batch, -1, -1), m_inv.unsqueeze(0).expand(batch, -1, -1)


def warp_official_full(src: torch.Tensor, H_dst_to_src: torch.Tensor) -> torch.Tensor:
    """Released Oneline-style warp.

    ``H_dst_to_src`` is the matrix returned by the official DLT branch. It is
    used as a sampling transform from output/target coordinates to source
    coordinates, matching ``utils.transform`` in JirongZhang/DeepHomography.
    """
    if src.ndim != 4:
        raise ValueError('src must be [B,C,H,W]')
    b, c, h, w = src.shape
    compute_dtype = _geometry_dtype(src, H_dst_to_src)
    with _disable_autocast(src.device.type):
        src_f = src.to(compute_dtype)
        H = H_dst_to_src.to(compute_dtype)
        m, m_inv = _pixel_to_norm_matrix(b, h, w, src.device, compute_dtype)
        H_norm = m_inv.bmm(H).bmm(m)

        x_t = torch.linspace(-1.0, 1.0, w, device=src.device, dtype=compute_dtype)
        y_t = torch.linspace(-1.0, 1.0, h, device=src.device, dtype=compute_dtype)
        yy, xx = torch.meshgrid(y_t, x_t, indexing='ij')
        ones = torch.ones_like(xx)
        grid = torch.stack([xx, yy, ones], dim=0).reshape(1, 3, h * w).expand(b, -1, -1)

        src_xyw = H_norm.bmm(grid)
        denom = src_xyw[:, 2:3, :].reshape(b, h, w)
        small = torch.tensor(1e-7, device=src.device, dtype=compute_dtype)
        denom = torch.where(denom.abs() < small, denom + 1e-6, denom)
        x_norm = (src_xyw[:, 0, :].reshape(b, h, w) / denom).reshape(-1)
        y_norm = (src_xyw[:, 1, :].reshape(b, h, w) / denom).reshape(-1)

        # Match the released implementation's scale_h=True coordinate mapping.
        x = (x_norm + 1.0) * w / 2.0
        y = (y_norm + 1.0) * h / 2.0
        x0 = torch.floor(x).long()
        x1 = x0 + 1
        y0 = torch.floor(y).long()
        y1 = y0 + 1
        x0 = x0.clamp(0, w - 1)
        x1 = x1.clamp(0, w - 1)
        y0 = y0.clamp(0, h - 1)
        y1 = y1.clamp(0, h - 1)

        batch_base = (torch.arange(b, device=src.device, dtype=torch.long) * h * w).repeat_interleave(h * w)
        idx_a = batch_base + y0 * w + x0
        idx_b = batch_base + y1 * w + x0
        idx_c = batch_base + y0 * w + x1
        idx_d = batch_base + y1 * w + x1

        src_flat = src_f.permute(0, 2, 3, 1).reshape(b * h * w, c)
        Ia = src_flat[idx_a]
        Ib = src_flat[idx_b]
        Ic = src_flat[idx_c]
        Id = src_flat[idx_d]

        x0f, x1f = x0.to(compute_dtype), x1.to(compute_dtype)
        y0f, y1f = y0.to(compute_dtype), y1.to(compute_dtype)
        wa = ((x1f - x) * (y1f - y)).unsqueeze(1)
        wb = ((x1f - x) * (y - y0f)).unsqueeze(1)
        wc = ((x - x0f) * (y1f - y)).unsqueeze(1)
        wd = ((x - x0f) * (y - y0f)).unsqueeze(1)
        out = wa * Ia + wb * Ib + wc * Ic + wd * Id
        out = out.reshape(b, h, w, c).permute(0, 3, 1, 2)
    return out.to(src.dtype)


def transform_official_patch(src: torch.Tensor, H_dst_to_src: torch.Tensor, patch_indices: torch.Tensor,
                             patch_h: int, patch_w: int) -> torch.Tensor:
    warped_full = warp_official_full(src, H_dst_to_src)
    return gather_patch_from_full(warped_full, patch_indices, patch_h, patch_w)
