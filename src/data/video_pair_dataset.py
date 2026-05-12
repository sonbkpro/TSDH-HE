from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from .transforms import (
    OFFICIAL_IMG_H,
    OFFICIAL_IMG_W,
    OFFICIAL_PATCH_H,
    OFFICIAL_PATCH_W,
    OFFICIAL_RHO,
    crop_official_patch,
    make_h4p,
    make_patch_indices,
    random_crop_pair,
    to_gray_float_tensor,
    to_official_gray_tensor,
)


def _rng_for_index(seed: int | None, index: int) -> np.random.Generator:
    if seed is None:
        return np.random.default_rng()
    worker = torch.utils.data.get_worker_info()
    worker_id = worker.id if worker else 0
    return np.random.default_rng(int(seed) + int(index) + worker_id * 100000)


class VideoFramePairDataset(Dataset):
    """Samples (frame_t, frame_t+k) from mp4 files, where k is random in [gap_min, gap_max]."""
    def __init__(self, video_dir: str, crop_h: int = 315, crop_w: int = 560, gap_min: int = 1, gap_max: int = 5,
                 pairs_per_epoch: int = 12000, seed: int | None = None, max_read_attempts: int = 20,
                 img_h: int = OFFICIAL_IMG_H, img_w: int = OFFICIAL_IMG_W, rho: int = OFFICIAL_RHO,
                 official_oneline: bool = True):
        self.video_dir = Path(video_dir)
        self.crop_h, self.crop_w = crop_h, crop_w
        self.img_h, self.img_w = int(img_h), int(img_w)
        self.rho = int(rho)
        self.official_oneline = bool(official_oneline)
        self.gap_min, self.gap_max = int(gap_min), int(gap_max)
        self.pairs_per_epoch = int(pairs_per_epoch)
        self.seed = None if seed is None else int(seed)
        self.max_read_attempts = int(max_read_attempts)
        self.videos = sorted([p for p in self.video_dir.glob('*.mp4')])
        if not self.videos:
            raise FileNotFoundError(f'No .mp4 videos found in {self.video_dir}')
        self.meta = []
        for p in self.videos:
            cap = cv2.VideoCapture(str(p))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if n > self.gap_max + 1:
                self.meta.append((p, n))
        if not self.meta:
            raise RuntimeError('No video has enough frames for the requested temporal gap')

    def __len__(self):
        return self.pairs_per_epoch

    @staticmethod
    def _read_frame(path: Path, idx: int):
        cap = cv2.VideoCapture(str(path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise RuntimeError(f'Could not read frame {idx} from {path}')
        return frame

    def __getitem__(self, index):
        rng = _rng_for_index(self.seed, int(index))
        last_error = None
        for _ in range(self.max_read_attempts):
            path, n = self.meta[int(rng.integers(0, len(self.meta)))]
            gap = int(rng.integers(self.gap_min, self.gap_max + 1))
            t = int(rng.integers(0, n - gap))
            try:
                a = self._read_frame(path, t)
                b = self._read_frame(path, t + gap)
                if not self.official_oneline:
                    a, b = random_crop_pair(a, b, self.crop_h, self.crop_w, rng)
            except (RuntimeError, ValueError, cv2.error) as e:
                last_error = e
                continue
            if self.official_oneline:
                return build_oneline_sample(
                    a, b, self.crop_h, self.crop_w, rng, self.img_h, self.img_w, self.rho,
                    metadata={'video': str(path), 't': t, 'gap': gap},
                )
            return {'ia': to_gray_float_tensor(a), 'ib': to_gray_float_tensor(b), 'video': str(path), 't': t, 'gap': gap}
        raise RuntimeError(f'Could not sample a readable frame pair after {self.max_read_attempts} attempts; last error: {last_error}')


class ImagePairListDataset(Dataset):
    """Official Oneline-style dataset backed by a text file of image-pair paths."""
    def __init__(self, pair_list: str, image_root: str, patch_h: int = OFFICIAL_PATCH_H, patch_w: int = OFFICIAL_PATCH_W,
                 img_h: int = OFFICIAL_IMG_H, img_w: int = OFFICIAL_IMG_W, rho: int = OFFICIAL_RHO,
                 seed: int | None = None):
        self.pairs = [line.strip().split()[:2] for line in Path(pair_list).read_text().splitlines() if line.strip()]
        if not self.pairs:
            raise FileNotFoundError(f'No image pairs found in {pair_list}')
        self.image_root = Path(image_root)
        self.patch_h, self.patch_w = int(patch_h), int(patch_w)
        self.img_h, self.img_w = int(img_h), int(img_w)
        self.rho = int(rho)
        self.seed = seed

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        rel_a, rel_b = self.pairs[index]
        img_a = cv2.imread(str(self.image_root / rel_a))
        img_b = cv2.imread(str(self.image_root / rel_b))
        if img_a is None or img_b is None:
            raise FileNotFoundError(f'Could not read {self.image_root / rel_a} or {self.image_root / rel_b}')
        rng = np.random.default_rng(None if self.seed is None else int(self.seed) + int(index))
        return build_oneline_sample(
            img_a, img_b, self.patch_h, self.patch_w, rng, self.img_h, self.img_w, self.rho,
            metadata={'path1': str(self.image_root / rel_a), 'path2': str(self.image_root / rel_b)},
        )


def build_oneline_sample(img_a: np.ndarray, img_b: np.ndarray, patch_h: int, patch_w: int,
                         rng: np.random.Generator | None, img_h: int = OFFICIAL_IMG_H,
                         img_w: int = OFFICIAL_IMG_W, rho: int = OFFICIAL_RHO,
                         metadata: dict | None = None, crop_xy: tuple[int, int] | None = None) -> dict:
    patch_h, patch_w = int(patch_h), int(patch_w)
    img_h, img_w = int(img_h), int(img_w)
    rho = int(rho)
    if patch_h > img_h or patch_w > img_w:
        raise ValueError(f'Patch size {patch_h}x{patch_w} cannot exceed image size {img_h}x{img_w}')
    full_a = to_official_gray_tensor(img_a, img_h, img_w)
    full_b = to_official_gray_tensor(img_b, img_h, img_w)
    full_pair = torch.cat([full_a, full_b], dim=0)
    if crop_xy is None:
        if rng is None:
            rng = np.random.default_rng()
        x_high = img_w - rho - patch_w
        y_high = img_h - rho - patch_h
        if x_high <= rho or y_high <= rho:
            raise ValueError(
                f'No valid official crop for image={img_h}x{img_w}, patch={patch_h}x{patch_w}, rho={rho}'
            )
        x = int(rng.integers(rho, x_high))
        y = int(rng.integers(rho, y_high))
    else:
        x, y = int(crop_xy[0]), int(crop_xy[1])
        if x < 0 or y < 0 or x + patch_w > img_w or y + patch_h > img_h:
            raise ValueError(f'Crop ({x},{y}) with patch {patch_h}x{patch_w} is outside image {img_h}x{img_w}')
    input_tensors = crop_official_patch(full_pair, x, y, patch_h, patch_w)
    sample = {
        'org_images': full_pair,
        'input_tensors': input_tensors,
        'patch_indices': make_patch_indices(x, y, patch_h, patch_w, img_w),
        'h4p': make_h4p(x, y, patch_h, patch_w),
        'ia': input_tensors[:1],
        'ib': input_tensors[1:],
        'crop_xy': torch.tensor([x, y], dtype=torch.int64),
        'full_size': torch.tensor([img_h, img_w], dtype=torch.int64),
    }
    if metadata:
        sample.update(metadata)
    return sample


class LabeledPointPairsDataset(Dataset):
    """Loads validation .npy dictionaries with keys: path1, path2, matche_pts.
    matche_pts format example: [[(x1,y1),(x2,y2)], ...].
    """
    def __init__(self, npy_dir: str, image_root: str, patch_h: int = OFFICIAL_PATCH_H, patch_w: int = OFFICIAL_PATCH_W,
                 img_h: int = OFFICIAL_IMG_H, img_w: int = OFFICIAL_IMG_W, crop_x: int = 40, crop_y: int = 23):
        self.npy_files = sorted(Path(npy_dir).glob('*.npy'))
        if not self.npy_files:
            raise FileNotFoundError(f'No .npy labels found in {npy_dir}')
        self.image_root = Path(image_root)
        self.patch_h, self.patch_w = int(patch_h), int(patch_w)
        self.img_h, self.img_w = int(img_h), int(img_w)
        self.crop_x, self.crop_y = int(crop_x), int(crop_y)

    def __len__(self): return len(self.npy_files)

    def __getitem__(self, idx):
        item = np.load(self.npy_files[idx], allow_pickle=True).item()
        p1, p2 = self.image_root / item['path1'], self.image_root / item['path2']
        img1, img2 = cv2.imread(str(p1)), cv2.imread(str(p2))
        if img1 is None or img2 is None:
            raise FileNotFoundError(f'Could not read {p1} or {p2}')
        scale1 = np.array([self.img_w / img1.shape[1], self.img_h / img1.shape[0]], dtype=np.float32)
        scale2 = np.array([self.img_w / img2.shape[1], self.img_h / img2.shape[0]], dtype=np.float32)
        pts_a = np.array([m[0] for m in item['matche_pts']], dtype=np.float32)
        pts_b = np.array([m[1] for m in item['matche_pts']], dtype=np.float32)
        sample = build_oneline_sample(
            img1, img2, self.patch_h, self.patch_w, None, self.img_h, self.img_w,
            crop_xy=(self.crop_x, self.crop_y),
        )
        sample.update({
            'pts_a': torch.from_numpy(pts_a * scale1),
            'pts_b': torch.from_numpy(pts_b * scale2),
            'path1': str(p1), 'path2': str(p2)
        })
        return sample
