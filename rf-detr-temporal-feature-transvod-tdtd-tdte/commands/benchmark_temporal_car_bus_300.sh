#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

DATASET_ROOT="${1:-temporal_dataset_combined}"
CHECKPOINT="${2:-result/rfdetr_temporal_3cls_lora_frozen_rfdetr_noema_30ep/checkpoint_29.ckpt}"
OUTPUT="${3:-reports/inference_speed/temporal_checkpoint29_car_bus_300.json}"
DEVICE="${DEVICE:-mps}"
PYTHON_BIN="${PYTHON_BIN:-.venv-macos/bin/python}"
SPLIT="${SPLIT:-val}"
MAX_IMAGES="${MAX_IMAGES:-300}"
WARMUP="${WARMUP:-10}"

NO_ALBUMENTATIONS_UPDATE=1 \
MPLCONFIGDIR="${PROJECT_ROOT}/.tmp/matplotlib" \
PYTORCH_ENABLE_MPS_FALLBACK=1 \
"${PYTHON_BIN}" tools/benchmark_inference.py \
  --checkpoint "${CHECKPOINT}" \
  --variant small \
  --num-classes 3 \
  --resolution 640 \
  --backbone-lora \
  --temporal-num-ref-frames 3 \
  --temporal-fusion-layers 1 \
  --temporal-tdte-layers 1 \
  --temporal-tdtd-layers 1 \
  --temporal-key-frame-position center \
  --temporal-ref-frame-mode surrounding \
  --temporal-ref-frame-step 2 \
  --benchmark-mode dataset \
  --dataset-root "${DATASET_ROOT}" \
  --split "${SPLIT}" \
  --max-images "${MAX_IMAGES}" \
  --classes car bus \
  --label-space local \
  --warmup "${WARMUP}" \
  --batch-size 1 \
  --device "${DEVICE}" \
  --dtype fp32 \
  --output "${OUTPUT}"
