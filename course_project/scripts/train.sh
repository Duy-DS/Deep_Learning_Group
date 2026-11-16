#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODE="${1:-multivariate}"
HORIZON="${2:-24}"
CONFIG_PATH="${3:-configs/default.yaml}"

SECONDS=0

echo "========================================================================"
echo "[BƯỚC 1/2] GLOBAL PRE-TRAINING (12 TRẠM) | Mode=${MODE} | H=${HORIZON}"
echo "========================================================================"
python training_loop.py \
  --config "${CONFIG_PATH}" \
  --mode "${MODE}" \
  --horizon "${HORIZON}"

echo ""
echo "========================================================================"
echo "[BƯỚC 2/2] PER-STATION FINE-TUNING (TỪNG TRẠM) | Mode=${MODE} | H=${HORIZON}"
echo "========================================================================"
python fine_tune.py \
  --config "${CONFIG_PATH}" \
  --mode "${MODE}" \
  --horizon "${HORIZON}"

echo ""
echo "[HOÀN TẤT] Toàn bộ Pre-training & Fine-tuning xong sau ${SECONDS} giây."
