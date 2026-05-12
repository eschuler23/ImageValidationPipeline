#!/usr/bin/env bash
set -euo pipefail

# ResNet50 + DTD weights with deterministic blur augmentation.
# Uses the local .venv Python directly (no uv dependency).
#
# Usage:
#   bash Image_Processing/Content_validation/run_resnet50_dtd_deterministic_blur.sh
#
# Optional overrides:
#   EPOCHS=15 RUN_NAME=dtd_resnet50_detblur_ep15 bash Image_Processing/Content_validation/run_resnet50_dtd_deterministic_blur.sh
#   GRAD_ACCUM_STEPS=4 LR=2e-5 UNFREEZE_LR=2e-5 FREEZE_BATCHNORM=1 AMP=1 bash Image_Processing/Content_validation/run_resnet50_dtd_deterministic_blur.sh

PROJECT_ROOT="/Users/raven/Projects/Bachelors"
PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
MAIN_SCRIPT="${PROJECT_ROOT}/Image_Processing/Content_validation/main.py"
DTD_CHECKPOINT="${PROJECT_ROOT}/checkpoints/dtd/resnet_dtd_finetuned.pth"

EPOCHS="${EPOCHS:-15}"
RUN_NAME="${RUN_NAME:-dtd_resnet50_detblur_ep${EPOCHS}}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SEED="${SEED:-42}"
CHECKPOINT_VERBOSE="${CHECKPOINT_VERBOSE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
FREEZE_BACKBONE_EPOCHS="${FREEZE_BACKBONE_EPOCHS:-1}"
UNFREEZE_LAST_BLOCK="${UNFREEZE_LAST_BLOCK:-1}"
FREEZE_BATCHNORM="${FREEZE_BATCHNORM:-1}"
LR="${LR:-2e-5}"
UNFREEZE_LR="${UNFREEZE_LR:-2e-5}"
AMP="${AMP:-1}"

CHECKPOINT_VERBOSE_FLAG=()
if [[ "${CHECKPOINT_VERBOSE}" == "1" ]]; then
  CHECKPOINT_VERBOSE_FLAG+=(--checkpoint-verbose)
fi

UNFREEZE_LAST_BLOCK_FLAG=()
if [[ "${UNFREEZE_LAST_BLOCK}" == "1" ]]; then
  UNFREEZE_LAST_BLOCK_FLAG+=(--unfreeze-last-block)
fi

FREEZE_BATCHNORM_FLAG=()
if [[ "${FREEZE_BATCHNORM}" == "1" ]]; then
  FREEZE_BATCHNORM_FLAG+=(--freeze-batchnorm)
fi

AMP_FLAG=()
if [[ "${AMP}" == "1" ]]; then
  AMP_FLAG+=(--amp)
fi

cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" "${MAIN_SCRIPT}" \
  --models resnet50 \
  --weights dtd \
  --init-checkpoint "${DTD_CHECKPOINT}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --unfreeze-lr "${UNFREEZE_LR}" \
  --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
  --freeze-backbone-epochs "${FREEZE_BACKBONE_EPOCHS}" \
  --seed "${SEED}" \
  --run-name "${RUN_NAME}" \
  --csv-path ground_truth.csv \
  --image-root Images \
  --project-column project \
  --filename-column filename \
  --label-column "usability considering nfp" \
  --positive-labels "usable" \
  --negative-labels "not usable" \
  --batch-size "${BATCH_SIZE}" \
  --decode-percent-newlines \
  --augment-blur-keep-range 0.5 1.5 0.5 \
  --augment-blur-flip-range 2.5 4.0 0.5 \
  --augment-blur-size-aware \
  --blur-size-small-max-mp 2 \
  --blur-size-medium-max-mp 10 \
  --blur-switch-small 2 \
  --blur-switch-medium 10 \
  --blur-switch-large 20 \
  --device auto \
  "${CHECKPOINT_VERBOSE_FLAG[@]}" \
  "${UNFREEZE_LAST_BLOCK_FLAG[@]}" \
  "${FREEZE_BATCHNORM_FLAG[@]}" \
  "${AMP_FLAG[@]}"
