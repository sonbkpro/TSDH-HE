# Content-Aware Unsupervised Deep Homography Estimation — PyTorch Implementation

This repository implements the official released **Oneline** variant of ECCV 2020 **Content-Aware Unsupervised Deep Homography Estimation** for small-baseline image/video pairs.

The Oneline variant is the version released in `JirongZhang/DeepHomography`: it predicts `H_ab` directly, uses the official patch/canvas geometry, and optimizes the released triplet-margin feature loss.

- Pure PyTorch DLT homography solver.
- Pure PyTorch official-style STN perspective sampler.
- Video dataloader for `dataset/train/000001.mp4`, `dataset/train/000002.mp4`, ...
- Optional official `Train_List.txt` image-pair dataloader.
- Random frame-pair sampling: `frame_t` and `frame_t+k`, where `k ∈ [1, 5]` by default.
- Official resize/crop protocol: full image `360 x 640`, patch `315 x 560`, `rho = 16`.
- Five supported semantic categories: regular, low-texture, low-light, small-foreground, large-foreground.
- Official Oneline two-stage training:
  - Stage 1: attention is enabled, but loss mask `mask_ap` is forced to ones.
  - Stage 2: attention and learned RANSAC-like loss mask are enabled.
- Official Oneline triplet-margin loss.
- Validation using manually labeled point correspondences stored in `.npy` files.
- Inference and visualization scripts.
- Smoke tests and unit tests.

---

## 1. Repository layout

```text
content_aware_deep_homography/
├── configs/
│   └── train_default.yaml
├── scripts/
│   ├── train.py
│   ├── infer_pair.py
│   ├── infer_video.py
│   ├── eval_labeled_points.py
│   ├── visualize_alignment.py
│   └── smoke_test.py
├── src/
│   ├── data/
│   ├── engine/
│   ├── geometry/
│   ├── losses/
│   ├── models/
│   └── utils/
└── tests/
```

---

## 2. Install

```bash
cd content_aware_deep_homography
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The implementation does **not** require Kornia. DLT and warping are implemented directly in PyTorch.

---

## 3. Data format

### Training videos

Place videos as:

```text
dataset/train/000001.mp4
dataset/train/000002.mp4
dataset/train/000003.mp4
...
```

The training dataloader samples a random video, then samples:

```text
Ia = frame_t
Ib = frame_t+k, where k ∈ [1, 5]
```

Each pair is resized to `640 x 360`, normalized with the official released constants, converted to one channel, then cropped to:

```text
315 x 560
```

### Optional official image-pair list

For closest parity with the official repo, preprocess videos into images and set:

```yaml
data:
  train_list: Data/Train_List.txt
  train_image_root: Data/Train
```

If these are unset, training samples frame pairs directly from `dataset/train/*.mp4` while using the same Oneline patch protocol.

---

## 4. Validation label format

The validation loader supports your `.npy` format:

```python
{
    'path1': '0000011_10001.jpg',
    'path2': '0000011_10005.jpg',
    'matche_pts': [
        [(349, 236), (357, 236)],
        [(397, 189), (401, 183)],
        ...
    ]
}
```

Put files as:

```text
dataset/val_labels/*.npy
dataset/val_images/0000011_10001.jpg
dataset/val_images/0000011_10005.jpg
```

Evaluate:

```bash
python scripts/eval_labeled_points.py \
  --ckpt runs/content_aware_homography/last.pt \
  --npy_dir dataset/val_labels \
  --image_root dataset/val_images
```

Metric:

```text
average L2 pixel error between predicted warped points and human-labeled target points
```

---

## 5. Train

Edit `configs/train_default.yaml` if needed, then run:

```bash
python scripts/train.py --config configs/train_default.yaml
```

Official-scale defaults are used:

```yaml
batch_size: 32
num_workers: 4
amp: true
```

The released Oneline recipe uses:

```text
Adam lr = 1e-4
batch size = 32
Adam amsgrad = true
weight decay = 1e-4
triplet margin = 1.0
stage-2 finetune lr = 6.4e-5
```

The default config keeps the iteration-based training loop from this repo and applies the released Oneline loss, preprocessing, geometry, and two-stage mask behavior.

---

## 6. Inference on two images

```bash
python scripts/infer_pair.py \
  --ckpt runs/content_aware_homography/last.pt \
  --image_a path/to/a.jpg \
  --image_b path/to/b.jpg \
  --out alignment_overlay.png
```

The script prints both the official sampling matrix `H_dst_to_src_sampling` and the point-transform matrix `H_ab_point_transform = inv(H_dst_to_src_sampling)`.

---

## 7. Visualize masks and alignment

```bash
python scripts/visualize_alignment.py \
  --ckpt runs/content_aware_homography/last.pt \
  --image_a path/to/a.jpg \
  --image_b path/to/b.jpg \
  --out_prefix vis/example
```

Outputs:

```text
vis/example_overlay.png
vis/example_mask_a.png
vis/example_mask_b.png
vis/example_warped_a.png
```

---

## 8. Smoke test

```bash
python scripts/smoke_test.py
pytest -q
```

Expected:

```text
SMOKE TEST PASSED
tests passed
```

---

## 9. Important implementation notes

1. **Feature extractor** follows Table 1(a): `Conv 1→4→8→1`, kernel 3, stride 1.
2. **Mask predictor** follows Table 1(b): `Conv 1→4→8→16→32→1`, kernel 3, stride 1, sigmoid output.
3. **Homography estimator** follows a ResNet-34-style backbone and predicts 8 corner offsets.
4. **Homography conversion** uses differentiable DLT from the full-image patch corners `h4p`.
5. **Warping** follows the released Oneline transform direction: the predicted matrix is a destination-to-source sampling homography.
6. **Loss** follows the released Oneline implementation:
   - anchor: target patch feature `F_b`,
   - positive: warped source patch feature `F'_a`,
   - negative: source patch feature `F_a`,
   - normalized by `mask_ap`.
7. **Two-stage training** follows the released Oneline behavior controlled by `stage1_use_*` and `stage2_use_*` config fields.

---

## 10. Practical advice

For your videos, start with:

```yaml
batch_size: 2 or 4
pairs_per_epoch: 2000
val_every: 1000
ckpt_every: 2000
```

After confirming stable training and visualizations, increase iterations toward the paper-scale schedule.

This method is designed for **small-baseline** video pairs. It is not expected to work well for large-baseline panorama-style stitching.
