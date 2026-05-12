import sys, tempfile
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import cv2, numpy as np
from src.data.video_pair_dataset import VideoFramePairDataset


def test_video_dataset():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td); (d/'train').mkdir()
        wr = cv2.VideoWriter(str(d/'train/000001.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 5, (64,48))
        for i in range(8): wr.write(np.full((48,64,3), i*20, np.uint8))
        wr.release()
        ds = VideoFramePairDataset(str(d/'train'), crop_h=32, crop_w=32, gap_min=1, gap_max=3, pairs_per_epoch=2)
        s = ds[0]
        assert s['ia'].shape == (1,32,32)
        assert s['org_images'].shape == (2,360,640)
        assert s['input_tensors'].shape == (2,32,32)
        assert 1 <= s['gap'] <= 3


def test_video_dataset_retries_unreadable_sample(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        d = Path(td); (d/'train').mkdir()
        wr = cv2.VideoWriter(str(d/'train/000001.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 5, (64,48))
        for i in range(8): wr.write(np.full((48,64,3), i*20, np.uint8))
        wr.release()
        original_read_frame = VideoFramePairDataset._read_frame
        calls = {'n': 0}

        def flaky_read_frame(path, idx):
            calls['n'] += 1
            if calls['n'] == 1:
                raise RuntimeError('simulated decode failure')
            return original_read_frame(path, idx)

        monkeypatch.setattr(VideoFramePairDataset, '_read_frame', staticmethod(flaky_read_frame))
        ds = VideoFramePairDataset(str(d/'train'), crop_h=32, crop_w=32, gap_min=1, gap_max=3, pairs_per_epoch=2, max_read_attempts=3)
        s = ds[0]
        assert s['ia'].shape == (1,32,32)
        assert s['patch_indices'].shape == (32*32,)
        assert calls['n'] > 1
