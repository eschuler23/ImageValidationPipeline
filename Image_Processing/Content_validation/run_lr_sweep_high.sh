#!/usr/bin/env bash
set -euo pipefail

# Learning-rate sweep for ResNet50 (high LR range).
# Edit the variables below to adjust the sweep in the future.

PROJECT_ROOT="/Users/raven/Projects/Bachelors"
PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
SWEEP_SCRIPT="${PROJECT_ROOT}/Image_Processing/Content_validation/sweep_lr.py"

MODELS=(resnet50)
LRS=(5e-4 8e-4 1e-3 3e-3 5e-3)
RUN_NAME="lr_sweep_resnet50_high"
BATCH_SIZE=4

cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" "${SWEEP_SCRIPT}" \
  --models "${MODELS[@]}" \
  --lrs "${LRS[@]}" \
  --run-name "${RUN_NAME}" \
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
  --batch-size "${BATCH_SIZE}" \
  --decode-percent-newlines \
  --augment \
  --augment-random-vflip \
  --augment-random-rotations 90 180 270 \
  --augment-jpeg-quality 60 95 \
  --augment-noise-std 0.02 \
  --augment-random-blur \
  --augment-blur-size-aware \
  --blur-size-small-max-mp 2 \
  --blur-size-medium-max-mp 10 \
  --blur-switch-small 2 \
  --blur-switch-medium 10 \
  --blur-switch-large 20 \
  --augment-random-blur-reject-radius 4 \
  --augment-blur-keep-range 0.5 20 0.5 \
  --augment-blur-flip-range 0.5 20 0.5
