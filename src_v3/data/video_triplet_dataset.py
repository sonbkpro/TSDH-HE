from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from src.data.video_pair_dataset import build_oneline_sample, _rng_for_index
from src.data.transforms import OFFICIAL_IMG_H, OFFICIAL_IMG_W, OFFICIAL_PATCH_H, OFFICIAL_PATCH_W, OFFICIAL_RHO


class VideoFrameTripletDataset(Dataset):
    """Samples (t, t+gap, t+2gap) and returns three Oneline-style pair samples.

    This is the default V3 dataset because three frames give temporal composition
    constraints while staying close to global homography estimation.
    """
    def __init__(self, video_dir: str, crop_h: int = OFFICIAL_PATCH_H, crop_w: int = OFFICIAL_PATCH_W,
                 gap_min: int = 1, gap_max: int = 3, pairs_per_epoch: int = 12000,
                 seed: int | None = None, max_read_attempts: int = 20,
                 img_h: int = OFFICIAL_IMG_H, img_w: int = OFFICIAL_IMG_W,
                 rho: int = OFFICIAL_RHO):
        self.video_dir = Path(video_dir)
        self.crop_h, self.crop_w = int(crop_h), int(crop_w)
        self.img_h, self.img_w = int(img_h), int(img_w)
        self.rho = int(rho)
        self.gap_min, self.gap_max = int(gap_min), int(gap_max)
        self.pairs_per_epoch = int(pairs_per_epoch)
        self.seed = None if seed is None else int(seed)
        self.max_read_attempts = int(max_read_attempts)
        self.videos = sorted(self.video_dir.glob('*.mp4'))
        if not self.videos:
            raise FileNotFoundError(f'No .mp4 videos found in {self.video_dir}')
        self.meta = []
        for p in self.videos:
            cap = cv2.VideoCapture(str(p))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if n > 2 * self.gap_max + 1:
                self.meta.append((p, n))
        if not self.meta:
            raise RuntimeError('No video has enough frames for V3 triplet sampling')

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

    @staticmethod
    def _prefix(d: dict, prefix: str) -> dict:
        return {f'{prefix}_{k}': v for k, v in d.items() if torch.is_tensor(v)}

    def __getitem__(self, index):
        rng = _rng_for_index(self.seed, int(index))
        last_error = None
        for _ in range(self.max_read_attempts):
            path, n = self.meta[int(rng.integers(0, len(self.meta)))]
            gap = int(rng.integers(self.gap_min, self.gap_max + 1))
            t = int(rng.integers(0, n - 2 * gap))
            try:
                f0 = self._read_frame(path, t)
                f1 = self._read_frame(path, t + gap)
                f2 = self._read_frame(path, t + 2 * gap)
            except (RuntimeError, ValueError, cv2.error) as e:
                last_error = e
                continue
            # Use the same crop for 01, 12, and 02 to make cycle loss geometrically consistent.
            x_high = self.img_w - self.rho - self.crop_w
            y_high = self.img_h - self.rho - self.crop_h
            x = int(rng.integers(self.rho, x_high))
            y = int(rng.integers(self.rho, y_high))
            p01 = build_oneline_sample(f0, f1, self.crop_h, self.crop_w, rng, self.img_h, self.img_w, self.rho, crop_xy=(x, y))
            p12 = build_oneline_sample(f1, f2, self.crop_h, self.crop_w, rng, self.img_h, self.img_w, self.rho, crop_xy=(x, y))
            p02 = build_oneline_sample(f0, f2, self.crop_h, self.crop_w, rng, self.img_h, self.img_w, self.rho, crop_xy=(x, y))
            out = {}
            out.update(self._prefix(p01, 'p01'))
            out.update(self._prefix(p12, 'p12'))
            out.update(self._prefix(p02, 'p02'))
            out['gap'] = torch.tensor(gap, dtype=torch.int64)
            out['t'] = torch.tensor(t, dtype=torch.int64)
            return out
        raise RuntimeError(f'Could not sample triplet after {self.max_read_attempts} attempts; last error: {last_error}')
