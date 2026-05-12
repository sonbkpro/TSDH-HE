# V2: Multi-Hypothesis Content-Aware Homography Estimation

This V2 extension keeps all V1 files unchanged and adds a separate `src_v2/`,
`scripts_v2/`, and `configs/train_v2.yaml`.

## Motivation

V1 predicts one homography and one content-aware mask. V2 predicts `K=4`
homography hypotheses, soft region-assignment masks, confidence scores, and one
dominant homography. This is intended for general homography estimation from
slightly moving-camera videos, where some regions may belong to different
planes or occasional dynamic objects.

## New files

```text
src_v2/
  models/
    multi_estimator.py
    v2_model.py
  losses/
    multi_homography_loss.py
    sparse_point_loss.py
  data/
    video_triplet_dataset.py
  engine/
    trainer_v2.py
    evaluator_v2.py

scripts_v2/
  train_v2.py
  eval_v2_labeled_points.py
  compare_v1_v2.py
  smoke_test_v2.py

configs/
  train_v2.yaml
```

## Data format

Training videos use the same format as V1:

```text
dataset/train/000001.mp4
dataset/train/000002.mp4
...
```

Validation labels use the same V1 `.npy` format:

```python
{
    "path1": "0000011_10001.jpg",
    "path2": "0000011_10005.jpg",
    "matche_pts": [
        [(x1, y1), (x2, y2)],
        ...
    ]
}
```

with images in:

```text
dataset/val_images/
dataset/val_labels/
```

## Smoke test

```bash
python scripts_v2/smoke_test_v2.py
```

## Train V2 from scratch

```bash
python scripts_v2/train_v2.py --config configs/train_v2.yaml
```

Default settings:
- `K=4`
- same V1 geometry: image `360x640`, patch `315x560`, rho `16`
- pairwise training first
- batch size `8` as a safe starting point for `K=4`

## Temporal fine-tuning

After pairwise training is stable, set this in `configs/train_v2.yaml`:

```yaml
train:
  use_temporal_cycle: true
```

Then resume/fine-tune using temporal triplets. The temporal cycle loss enforces:

```text
H_02 ≈ H_01 @ H_12
```

under the Oneline destination-to-source matrix convention.

## Evaluate V2

```bash
python scripts_v2/eval_v2_labeled_points.py \
  --ckpt runs/v2_multi_hypothesis_homography/last.pt \
  --npy_dir dataset/val_labels \
  --image_root dataset/val_images
```

Reported metrics:
- `dominant_point_l2_mean`
- `best_of_k_point_l2_mean`
- `dominant_vs_best_gap`
- `dominant_inlier_3px`
- `best_of_k_inlier_3px`
- `active_experts`

## Compare V1 and V2

```bash
python scripts_v2/compare_v1_v2.py \
  --v1_ckpt runs/content_aware_homography/last.pt \
  --v2_ckpt runs/v2_multi_hypothesis_homography/last.pt \
  --npy_dir dataset/val_labels \
  --image_root dataset/val_images
```

## Important notes

1. V2 is intentionally initialized from scratch, as requested.
2. V1 is not modified.
3. V2's full Oneline forward path is computationally heavier than V1 because it
   performs `K` homography warps per pair.
4. On 2 GPUs, start with `batch_size: 8`. Increase only after checking GPU memory.
5. If the best-of-K metric improves but dominant metric does not, the hypotheses
   are useful but the consensus selector must be improved.
