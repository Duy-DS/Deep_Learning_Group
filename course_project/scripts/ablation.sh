#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ABLATIONS="${1:-all}"
EXTRA_FLAGS="${2:-}"
CONFIG_PATH="${3:-configs/default.yaml}"

SECONDS=0

echo "========================================================================"
echo "[ABLATION STUDIES] Chạy thí nghiệm: ${ABLATIONS} ${EXTRA_FLAGS}"
echo "========================================================================"
python run_ablation.py \
  --config "${CONFIG_PATH}" \
  --ablations ${ABLATIONS} \
  ${EXTRA_FLAGS}

echo ""
echo "[HOÀN TẤT] Báo cáo Ablation đã xuất tại outputs/ablation/ (Tổng thời gian: ${SECONDS}s)."
