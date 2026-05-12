#!/usr/bin/env bash
set -euo pipefail

# Baseline stability check at the original learning rate (1e-4).
# Uses the same seeds as the high-LR stability check for fair comparison.

PROJECT_ROOT="/Users/raven/Projects/Bachelors"
PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
MAIN_SCRIPT="${PROJECT_ROOT}/Image_Processing/Content_validation/main.py"

MODEL="resnet50"
LR="1e-4"
SEEDS=(13 42 1337)
RUN_PREFIX="lr_stability_resnet50_baseline"
BATCH_SIZE=4

cd "${PROJECT_ROOT}"

for SEED in "${SEEDS[@]}"; do
  RUN_NAME="${RUN_PREFIX}_${MODEL}_lr-${LR}_seed-${SEED}"
  "${PYTHON_BIN}" "${MAIN_SCRIPT}" \
    --models "${MODEL}" \
    --lr "${LR}" \
    --seed "${SEED}" \
    --run-name "${RUN_NAME}" \
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
 done
