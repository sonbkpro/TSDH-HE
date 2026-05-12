from __future__ import annotations
import cv2
import numpy as np
import torch


def tensor_gray_to_uint8(x: torch.Tensor) -> np.ndarray:
    x = x.detach().float().cpu()
    if x.ndim == 4: x = x[0]
    if x.shape[0] == 1: x = x[0]
    arr = x.numpy()
    if arr.min() < 0.0 or arr.max() > 1.0:
        return cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)


def make_alignment_overlay(warped_a: torch.Tensor, target_b: torch.Tensor) -> np.ndarray:
    wa = tensor_gray_to_uint8(warped_a)
    tb = tensor_gray_to_uint8(target_b)
    out = np.zeros((tb.shape[0], tb.shape[1], 3), dtype=np.uint8)
    out[..., 0] = tb          # red channel: target
    out[..., 1] = wa          # green channel: warped
    out[..., 2] = wa          # blue channel: warped
    return out


def save_image(path: str, img: np.ndarray) -> None:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)
