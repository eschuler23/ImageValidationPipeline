#!/usr/bin/env bash
set -euo pipefail

# ResNet50 + DTD + deterministic blur, with:
# - Gradient accumulation x4 (effective batch size x4)
# - BatchNorm frozen
# - Backbone frozen initially, then unfreeze last block
# - AdamW LR=2e-5
# - AMP enabled when available
#
# Usage:
#   bash Image_Processing/Content_validation/run_resnet50_dtd_detblur_accum_amp.sh
#
# Optional overrides:
#   EPOCHS=15 RUN_NAME=dtd_resnet50_detblur_accum_amp_ep15 \
#   BATCH_SIZE=8 SEED=42 bash Image_Processing/Content_validation/run_resnet50_dtd_detblur_accum_amp.sh

PROJECT_ROOT="/Users/raven/Projects/Bachelors"
PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
MAIN_SCRIPT="${PROJECT_ROOT}/Image_Processing/Content_validation/main.py"
DTD_CHECKPOINT="${PROJECT_ROOT}/checkpoints/dtd/resnet_dtd_finetuned.pth"

EPOCHS="${EPOCHS:-15}"
RUN_NAME="${RUN_NAME:-dtd_resnet50_detblur_accum_amp_ep${EPOCHS}}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SEED="${SEED:-42}"

cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" "${MAIN_SCRIPT}" \
  --models resnet50 \
  --weights dtd \
  --init-checkpoint "${DTD_CHECKPOINT}" \
  --epochs "${EPOCHS}" \
  --lr 2e-5 \
  --unfreeze-lr 2e-5 \
  --grad-accum-steps 4 \
  --freeze-backbone-epochs 1 \
  --unfreeze-last-block \
  --freeze-batchnorm \
  --amp \
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
  --device auto
