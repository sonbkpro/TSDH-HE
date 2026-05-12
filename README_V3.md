# V3: Temporal Static-Support Dominant Homography Estimation

V3 keeps V1 unchanged and abandons the ineffective K-homography decomposition used in V2. It estimates one dominant global homography, but adds four mechanisms around the V1 formulation:

1. **Temporal triplet learning** from `(I_t, I_{t+gap}, I_{t+2gap})`.
2. **Static-support map** using the V1 mask architecture, interpreted as reliable dominant-plane support.
3. **Differentiable local consensus refinement** around one regressed homography, not multiple planes.
4. **Uncertainty/confidence head** for homography reliability.

## Why three frames by default?

Your camera is slightly moving and labels mostly lie on the dominant background plane. Three frames are the best default because they allow the composition constraint:

```text
H_02 ≈ H_01 @ H_12
```

using the official V1 matrix convention. This gives temporal stability without moving away from single global homography estimation.

## Files added

```text
src_v3/
  models/dominant_homography_v3.py
  models/consensus_refiner.py
  models/uncertainty_head.py
  losses/v3_losses.py
  data/video_triplet_dataset.py
  engine/trainer_v3.py
  engine/evaluator_v3.py

scripts_v3/
  train_v3.py
  eval_v3_labeled_points.py
  smoke_test_v3.py
  visualize_alignment_v3.py
  visualize_static_support.py

configs/train_v3.yaml
```

## Smoke test

```bash
python scripts_v3/smoke_test_v3.py
```

Expected output includes:

```text
H_shape: (1, 3, 3)
H_init_shape: (1, 3, 3)
offsets_shape: (1, 8)
support_shape: (1, 1, 32, 48)
confidence_shape: (1, 1)
candidate_weights_shape: (1, 5)
```

## Training

Put videos in:

```text
dataset/train/000001.mp4
dataset/train/000002.mp4
...
```

Run:

```bash
python scripts_v3/train_v3.py --config configs/train_v3.yaml
```

Default training uses temporal triplets and all V3 mechanisms from the start, but with conservative weights:

```yaml
triplet: 1.0
init_triplet: 0.25
support: 0.01
uncertainty: 0.005
consensus: 0.03
temporal_cycle: 0.05
```

## Evaluation

```bash
python scripts_v3/eval_v3_labeled_points.py \
  --ckpt runs/v3_dominant_temporal_support/last.pt \
  --npy_dir dataset/val_labels \
  --image_root dataset/val_images
```

Metrics:

```text
point_l2_mean          final refined homography error
init_point_l2_mean     direct V1-style prediction before consensus
refine_gain            init error - final error
inlier_3px             final inlier ratio
confidence_mean        predicted global confidence
num_pairs              number of labeled validation pairs
```

## Visualization

V1-compatible alignment visualization for evaluation checks:

```bash
python scripts_v3/visualize_alignment_v3.py \
  --ckpt runs/v3_dominant_temporal_support/last.pt \
  --image_a path/to/a.jpg \
  --image_b path/to/b.jpg \
  --out_prefix vis/v3_example
```

Outputs:

```text
vis/v3_example_overlay.png
vis/v3_example_mask_a.png
vis/v3_example_mask_b.png
vis/v3_example_warped_a.png
vis/v3_example_support_a_patch.png
vis/v3_example_support_b_patch.png
vis/v3_example_support_init.png
vis/v3_example_support_final.png
vis/v3_example_support_a_full.png
vis/v3_example_support_b_full.png
```

Static-support patch visualization:

```bash
python scripts_v3/visualize_static_support.py \
  --ckpt runs/v3_dominant_temporal_support/last.pt \
  --image_a path/to/a.jpg \
  --image_b path/to/b.jpg \
  --out v3_static_support.png
```

The visualization concatenates:

```text
Ia patch | Ib patch | warped Ia by V3 | static-support map
```

## Important note

This V3 still estimates one global dominant homography. It does not claim to solve every plane exactly. Multi-plane or dynamic regions are treated as non-homographic residuals and should receive low static support.
