from __future__ import annotations
import cv2
import numpy as np
import torch


OFFICIAL_IMG_W = 640
OFFICIAL_IMG_H = 360
OFFICIAL_PATCH_W = 560
OFFICIAL_PATCH_H = 315
OFFICIAL_RHO = 16

# The released Oneline code reads images with cv2 and applies these constants
# directly before averaging the three channels into one grayscale tensor.
OFFICIAL_MEAN = np.reshape(np.array([118.93, 113.97, 102.60], dtype=np.float32), (1, 1, 3))
OFFICIAL_STD = np.reshape(np.array([69.85, 68.81, 72.45], dtype=np.float32), (1, 1, 3))


def resize_if_needed(img: np.ndarray, min_h: int, min_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max(min_h / max(h, 1), min_w / max(w, 1), 1.0)
    if scale > 1.0:
        img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_LINEAR)
    return img


def random_crop_pair(a: np.ndarray, b: np.ndarray, crop_h: int, crop_w: int, rng: np.random.Generator):
    a = resize_if_needed(a, crop_h, crop_w)
    b = resize_if_needed(b, crop_h, crop_w)
    h, w = a.shape[:2]
    if b.shape[:2] != (h, w):
        b = cv2.resize(b, (w, h), interpolation=cv2.INTER_LINEAR)
    y = int(rng.integers(0, h - crop_h + 1))
    x = int(rng.integers(0, w - crop_w + 1))
    return a[y:y+crop_h, x:x+crop_w], b[y:y+crop_h, x:x+crop_w]


def to_gray_float_tensor(img: np.ndarray) -> torch.Tensor:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32) / 255.0
    return torch.from_numpy(img).unsqueeze(0)


def resize_official(img: np.ndarray, img_h: int = OFFICIAL_IMG_H, img_w: int = OFFICIAL_IMG_W) -> np.ndarray:
    if img is None:
        raise ValueError('image is None')
    h, w = img.shape[:2]
    if h != img_h or w != img_w:
        img = cv2.resize(img, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
    return img


def to_official_gray_tensor(img: np.ndarray, img_h: int = OFFICIAL_IMG_H, img_w: int = OFFICIAL_IMG_W) -> torch.Tensor:
    img = resize_official(img, img_h, img_w).astype(np.float32)
    img = (img - OFFICIAL_MEAN) / OFFICIAL_STD
    img = np.mean(img, axis=2, keepdims=True)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(np.float32))


def make_patch_indices(x: int, y: int, patch_h: int, patch_w: int, full_w: int) -> torch.Tensor:
    yy, xx = np.meshgrid(np.arange(patch_h), np.arange(patch_w), indexing='ij')
    indices = (yy.reshape(-1) + int(y)) * int(full_w) + (xx.reshape(-1) + int(x))
    return torch.from_numpy(indices.astype(np.float32))


def make_h4p(x: int, y: int, patch_h: int, patch_w: int) -> torch.Tensor:
    points = [
        (x, y),
        (x, y + patch_h),
        (x + patch_w, y + patch_h),
        (x + patch_w, y),
    ]
    return torch.tensor(np.reshape(points, (-1)), dtype=torch.float32)


def crop_official_patch(full_pair: torch.Tensor, x: int, y: int, patch_h: int, patch_w: int) -> torch.Tensor:
    return full_pair[:, int(y):int(y) + int(patch_h), int(x):int(x) + int(patch_w)]
