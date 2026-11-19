#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

DATASET_ROOT="${DATASET_ROOT:-temporal_dataset_combined_v2}"
OUTPUT_DIR="${OUTPUT_DIR:-result/rfdetr_temporal_3cls_lora_frozen_rfdetr_30ep}"
DEVICE="${DEVICE:-mps}"
PYTHON_BIN="${PYTHON_BIN:-.venv-macos/bin/python}"

NO_ALBUMENTATIONS_UPDATE=1 \
MPLCONFIGDIR="$PWD/.tmp/matplotlib" \
PYTORCH_ENABLE_MPS_FALLBACK=1 \
"${PYTHON_BIN}" tools/train_rfdetr_temporal.py \
  --dataset-root "${DATASET_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --variant small \
  --num-classes 3 \
  --resolution 640 \
  --backbone-lora \
  --freeze-rfdetr-base \
  --temporal-num-ref-frames 3 \
  --temporal-fusion-layers 1 \
  --temporal-tdte-layers 1 \
  --temporal-tdtd-layers 1 \
  --temporal-ref-frame-mode surrounding \
  --temporal-ref-frame-step 2 \
  --temporal-key-frame-position center \
  --epochs 30 \
  --batch-size 1 \
  --grad-accum-steps 4 \
  --lr 1e-4 \
  --lr-encoder 1e-5 \
  --lr-drop 25 \
  --num-workers 0 \
  --device "${DEVICE}" \
  --checkpoint-interval 5 \
  --realtime-fps-target 30
