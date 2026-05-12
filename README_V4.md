# V4 / TSDH-Net: Temporal Support-Decomposed Homography

This repository keeps the original V1/V2/V3 code unchanged and adds V4 in:

```text
src_v4/
scripts_v4/
configs/train_v4.yaml
```

## Research goal

V4 avoids the ineffective `K`-homography decomposition. It estimates one dominant global homography and learns which regions are temporally reliable for that homography.

Core idea:

```text
I_t, I_{t+1}, I_{t+2}
  -> initial V1-style homographies H01, H12, H02
  -> pair residuals R01, R12, R02
  -> pixel-level cycle residual C012
  -> temporal support map S
  -> final dominant homography H
  -> non-homographic residual map N
```

The main new modules are:

```text
src_v4/models/temporal_support_refiner.py
src_v4/models/nonhomographic_residual_head.py
src_v4/models/tsdh_net.py
src_v4/losses/tsdh_losses.py
src_v4/engine/trainer_v4.py
src_v4/engine/evaluator_v4.py
```

## Why V4 is different from V3

V3 adds temporal cycle, consensus, and uncertainty, but its support map is still mostly a pairwise mask.
V4 changes the formulation: the support map is predicted from geometric evidence:

```text
initial support
pairwise feature residual
long-range residual
pixel-level temporal cycle residual
```

This makes the support map a temporal homography-support estimator, not just a renamed V1 mask.

## Smoke tests

```bash
python scripts_v4/smoke_test_v4.py
python scripts_v4/smoke_test_triplet_v4.py
```

Expected output includes:

```text
ok: True
H_shape: (1, 3, 3)
support_shape: (1, 1, H, W)
nonh_shape: (1, 1, H, W)
cycle_residual_shape: (1, 1, H, W)
```

## Training

```bash
python scripts_v4/train_v4.py --config configs/train_v4.yaml
```

Resume:

```bash
python scripts_v4/train_v4.py \
  --config configs/train_v4.yaml \
  --resume runs/v4_tsdh_net/last.pt
```

## Evaluation on V1-style labeled points

```bash
python scripts_v4/eval_v4_labeled_points.py \
  --ckpt runs/v4_tsdh_net/last.pt \
  --npy_dir dataset/val_labels \
  --image_root dataset/val_images
```

This pairwise evaluation uses the residual-adaptive fallback of the temporal support refiner. Full temporal support is used during triplet training.

## Staged training schedule

V4 is staged for stability:

```text
Stage 1: V1-like warm-up. No temporal support, no nonH loss, no cycle loss.
Stage 2: pair residual-adaptive support.
Stage 3: full triplet temporal support decomposition.
Stage 4: stronger temporal support + nonH residual decomposition.
```

This is intentional. Training all terms from iteration 0 was unstable in V3.

## Important metrics

During validation:

```text
point_l2_mean
init_point_l2_mean
refine_gain
inlier_3px
support_mean
nonh_mean
num_pairs
```

`refine_gain = init_point_l2_mean - point_l2_mean`.
If it is positive, temporal/residual support improves the final homography over the initial V1-style estimate.

## Suggested paper claim

A safe IEEE TIP-style claim is:

> We propose Temporal Support-Decomposed Homography Estimation, where one dominant global homography is estimated only from regions that remain temporally consistent under homography composition, while non-homographic residual regions are explicitly modeled rather than forced into the global transform.

Do not claim that one homography completely solves multi-plane geometry. V4 estimates the dominant global homography robustly.

## Stability patch notes

This package includes the strict V4 stability patch:

1. During `s1_v1_warmup`, `TSDHNet.forward_pair(..., use_temporal_support=False)` now returns the V1-style initial branch as the final output (`H = H_init`). It no longer trains the second support-decomposed estimator during warm-up.
2. V4 validation now uses the same direct/swapped point-order guard as the V1 evaluator, so `point_l2_mean` is comparable with V1.
3. Trainer validation disables temporal support automatically during `s1_v1_warmup`, so validation measures the same branch being trained.

Expected Stage-1 validation behavior:

```text
point_l2_mean ≈ init_point_l2_mean
refine_gain ≈ 0
support_mean ≈ 1.0
nonh_mean ≈ 0.0
```

If you manually evaluate a Stage-1 checkpoint, use:

```bash
python scripts_v4/eval_v4_labeled_points.py \
  --ckpt runs/v4_tsdh_net/last.pt \
  --npy_dir dataset/val_labels \
  --image_root dataset/val_images \
  --no_temporal_support
```
