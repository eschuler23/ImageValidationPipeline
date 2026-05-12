# Content Validation – Supervised Training

This folder contains a training script that compares multiple backbones for a
**binary** content-validation classifier using `ground_truth.csv` plus the
`Images/` folder (multi-project root). It supports deterministic blur augmentation,
tracks preprocessing metadata, and can save validation grids for visual review.

## Clean pipeline entry point (recommended)
The pipeline entry point is `main.py`. It only loads images + ground truth and
then calls documented step functions from `pipeline_steps/`.

```bash
uv run python Image_Processing/Content_validation/main.py \
  --csv-path ground_truth.csv \
  --image-root Images \
  --project-column project \
  --filename-column filename \
  --label-column "usability considering nfp" \
  --positive-labels "usable" \
  --negative-labels "not usable" \
  --models resnet18 \
  --weights imagenet \
  --device auto \
  --decode-percent-newlines \
  --save-val-grid
```

Note: `main.py` runs **one model per run** to avoid long multi-model jobs.

## Multi-model runs (one model per call)
Run **one model per call** to avoid tool timeouts. Repeat the `main.py` command
with a different `--models` value for each backbone you want to evaluate.

## 1) Open the project
- Open `/Users/raven/Projects/Bachelors` in VSCode or Terminal.

## 2) Create / activate the VINV
```bash
cd /Users/raven/Projects/Bachelors
uv venv .venv
source .venv/bin/activate
```

## 3) Install dependencies
```bash
uv pip install torch torchvision pillow matplotlib
```

Optional (for a broader model catalog later):
```bash
uv pip install timm
```

## 4) Check ground-truth completeness
```bash
uv run python Image_Processing/Content_validation/check_ground_truth.py \
  --csv-path ground_truth.csv \
  --image-root Images \
  --label-column "usability considering nfp" \
  --filename-column filename \
  --match-mode filename
```

If the CSV filenames contain `%0A`, add:
```bash
  --decode-percent-newlines
```

If project names do not match folder names, pass a mapping JSON (CSV project ->
folder name):
```bash
uv run python Image_Processing/Content_validation/check_ground_truth.py \
  --csv-path ground_truth.csv \
  --image-root Images \
  --label-column "usability considering nfp" \
  --project-column project \
  --filename-column filename \
  --match-mode project \
  --project-map Image_Processing/Content_validation/project_map.json
```

## 4b) Ground-truth label overview
Use this to summarize label distributions (counts + percentages) before training.
It can also be reused later to analyze confusion-matrix slices by filename.

```bash
uv run python Image_Processing/Content_validation/ground_truth_analysis.py \
  --csv-path ground_truth.csv \
  --label-columns "usability considering nfp" "usability considering blur" "Blurr" \
  --usability-column "usability considering nfp" \
  --usable-labels "usable" \
  --unusable-labels "not usable" \
  --decode-percent-newlines \
  --output-json Image_Processing/Content_validation/reviews/ground_truth_summary.json
```

For blur usability, swap these arguments:
```bash
  --usability-column "usability considering blur" \
  --usable-labels "focused enough" \
  --unusable-labels "too blurry"
```

## 5) Train a model sweep (content-validation labels)
```bash
uv run python Image_Processing/Content_validation/train_binary_classifier.py \
  --csv-path ground_truth.csv \
  --image-root Images \
  --project-column project \
  --filename-column filename \
  --label-column "usability considering nfp" \
  --positive-labels "usable" \
  --negative-labels "not usable" \
  --models resnet18 resnet50 squeezenet1_1 \
  --weights imagenet \
  --device auto \
  --decode-percent-newlines \
  --save-val-grid
```

## 5b) Learning-rate sweep helper (one model per run)
Use this when you want to compare different learning rates for the *same*
model without manual repetition. This helper calls `main.py` once per lr and
writes a consolidated review file.

```bash
uv run python Image_Processing/Content_validation/sweep_lr.py \
  --models resnet50 \
  --lrs 1e-5 3e-5 1e-4 3e-4 \
  --run-name lr_sweep_resnet50 \
  -- \
  --csv-path ground_truth.csv \
  --image-root Images \
  --project-column project \
  --filename-column filename \
  --label-column "usability considering nfp" \
  --positive-labels "usable" \
  --negative-labels "not usable" \
  --weights imagenet \
  --device auto \
  --decode-percent-newlines \
  --save-val-grid
```

The consolidated review markdown is written to:
```
Image_Processing/Content_validation/reviews/<run_name>_review.md
```

Outputs will be saved under:
```
Image_Processing/Content_validation/runs/<run_name>/
```

Key outputs per run:
- `data_stats.json` (label + match counts)
- `preprocessing.json` (augmentation + transform metadata)
- `dataset_manifest.csv` (per-sample split + augmentation)
- `val_predictions.csv` + `val_grid_*.png` (when `--save-val-grid` is used)

## 6) Optional: deterministic blur augmentation
Deterministic blur variants are generated **before training** so labels can
flip deterministically. Rotation/flip is **on-the-fly only**.
```bash
uv run python Image_Processing/Content_validation/train_binary_classifier.py \
  --csv-path ground_truth.csv \
  --image-root Images \
  --project-column project \
  --filename-column filename \
  --label-column "usability considering nfp" \
  --positive-labels "usable" \
  --negative-labels "not usable" \
  --models resnet18 \
  --weights imagenet \
  --augment-blur-keep-range 0.5 1.5 0.5 \
  --augment-blur-flip-range 2.5 4.0 0.5 \
  --augment-blur-size-aware \
  --blur-size-small-max-mp 2 \
  --blur-size-medium-max-mp 10 \
  --blur-switch-small 2 \
  --blur-switch-medium 10 \
  --blur-switch-large 20 \
  --device auto \
  --save-val-grid
```

Size-aware blur switching (pixel-based; small images flip sooner, very large images later):
```bash
uv run python Image_Processing/Content_validation/train_binary_classifier.py \
  --csv-path ground_truth.csv \
  --image-root Images \
  --project-column project \
  --filename-column filename \
  --label-column "usability considering nfp" \
  --positive-labels "usable" \
  --negative-labels "not usable" \
  --models resnet18 \
  --weights imagenet \
  --augment \
  --augment-random-vflip \
  --augment-random-rotations 90 180 270 \
  --augment-random-blur \
  --augment-blur-size-aware \
  --blur-size-small-max-mp 2 \
  --blur-size-medium-max-mp 10 \
  --blur-switch-small 2 \
  --blur-switch-medium 10 \
  --blur-switch-large 20 \
  --augment-random-blur-reject-radius 4 \
  --augment-blur-keep-range 0.5 20 0.5 \
  --augment-blur-flip-range 0.5 20 0.5 \
  --device auto \
  --save-val-grid
```

## 6b) Optional: on-the-fly augmentation (label-preserving)
```bash
uv run python Image_Processing/Content_validation/train_binary_classifier.py \
  --csv-path ground_truth.csv \
  --image-root Images \
  --project-column project \
  --filename-column filename \
  --label-column "usability considering nfp" \
  --positive-labels "usable" \
  --negative-labels "not usable" \
  --models resnet18 \
  --weights imagenet \
  --augment \
  --augment-random-vflip \
  --augment-random-rotations 90 180 270 \
  --augment-jpeg-quality 60 95 \
  --augment-noise-std 0.02 \
  --augment-color-jitter \
  --augment-brightness-jitter \
  --device auto \
  --decode-percent-newlines \
  --save-val-grid
```

Optional: random blur with size-aware label flip (on-the-fly). When enabled,
deterministic blur variants are **disabled**, and blur is sampled randomly per
epoch. Labels flip if the sampled radius exceeds the size threshold.
```bash
  --augment-random-blur \
  --augment-blur-keep-range 2 30 2 \
  --augment-blur-flip-range 2 30 2 \
  --augment-blur-size-aware \
  --blur-size-small-max-mp 2 \
  --blur-size-medium-max-mp 10 \
  --blur-switch-small 2 \
  --blur-switch-medium 10 \
  --blur-switch-large 20 \
  --augment-random-blur-reject-radius 4
```
If you prefer a continuous range, use:
```bash
  --augment-random-blur \
  --augment-random-blur-range 2 30
```

## 7) Optional: DTD initialization
If you have a DTD-pretrained checkpoint, load it with `--weights dtd`:
```bash
uv run python Image_Processing/Content_validation/train_binary_classifier.py \
  --csv-path ground_truth.csv \
  --image-root Images \
  --project-column project \
  --filename-column filename \
  --label-column "usability considering nfp" \
  --positive-labels "usable" \
  --negative-labels "not usable" \
  --models resnet50 \
  --weights dtd \
  --init-checkpoint checkpoints/resnet50_dtd.pt \
  --device auto \
  --save-val-grid
```

## Hardware notes (MacBook Air M2)
- **ResNet-18 + SqueezeNet** should run locally with small batches (e.g., `--batch-size 8`).
- **ResNet-50** may be slower; reduce batch size if you hit memory pressure.
- For speed, prefer `--device mps` (Apple Metal) if available.
- If you see out-of-memory errors, reduce `--batch-size` or `--image-size`.

## Notes
- Keep execution local; do not upload images.
- Disable `--augment` if you want only deterministic blur variants.
- Color/brightness jitter are off by default; enable only if color/brightness are non-diagnostic.
- Duplicate filenames are dropped by default (first occurrence kept). Use `--no-dedupe-filenames` to keep them.

## Blur threshold review (choose x/y/z ranges)
Generate grids to decide which blur radii keep labels vs flip to `not usable`:
```bash
uv run python Image_Processing/Content_validation/blur_threshold_calibration.py \
  --csv-path ground_truth.csv \
  --image-root Images \
  --project-column project \
  --filename-column filename \
  --label-column "usability considering nfp" \
  --positive-labels "usable" \
  --negative-labels "not usable" \
  --decode-percent-newlines \
  --max-images 12 \
  --blur-range 0.5 4.0 0.5 \
  --output-dir Image_Processing/Content_validation/reviews/blur_threshold_review
```

## 8) Optional: size category review
Pick one example per pixel-size bucket to validate blur thresholds:
```bash
uv run python Image_Processing/Content_validation/size_category_review.py \
  --csv-path ground_truth.csv \
  --image-root Images \
  --project-column project \
  --filename-column filename \
  --label-column "usability considering nfp" \
  --positive-labels "usable" \
  --negative-labels "not usable" \
  --decode-percent-newlines \
  --label-filter positive \
  --output-dir Image_Processing/Content_validation/reviews/size_category_review
```

## 9) Optional: size-aware blur grids
Visualize blur augmentation labels per pixel-size bucket:
```bash
uv run python Image_Processing/Content_validation/size_aware_blur_review.py \
  --csv-path ground_truth.csv \
  --image-root Images \
  --project-column project \
  --filename-column filename \
  --label-column "usability considering nfp" \
  --positive-labels "usable" \
  --negative-labels "not usable" \
  --decode-percent-newlines \
  --label-filter positive \
  --blur-range 0 20 2 \
  --output-dir Image_Processing/Content_validation/reviews/size_aware_blur_review
```
