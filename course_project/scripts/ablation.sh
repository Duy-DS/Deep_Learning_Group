#!/usr/bin/env bash
set -euo pipefail

# Tự động chuyển về thư mục gốc dự án (cha của thư mục scripts/)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Chọn nhóm thí nghiệm: all | ablation1 | ablation2 | ablation3 (mặc định: all)
ABLATIONS="${1:-all}"
# Truyền thêm "--with-finetune" ở tham số thứ 2 nếu muốn chạy kèm fine-tune 12 trạm
EXTRA_FLAGS="${2:-}"

DATA_DIR="data/raw"
OUTPUT_DIR="outputs"
SEEDS="42 1024 2026"

SECONDS=0

echo "========================================================================"
echo "[ABLATION STUDIES] Chạy thí nghiệm: ${ABLATIONS} ${EXTRA_FLAGS}"
echo "========================================================================"
python run_ablation.py \
  --data-dir "${DATA_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --ablations ${ABLATIONS} \
  --ab3-horizons 96 192 \
  --lookback 96 \
  --batch-size 512 \
  --train-stride 6 \
  --lr 5e-5 \
  --weight-decay 1e-2 \
  --dropout 0.3 \
  --epochs 20 \
  --patience 6 \
  --seeds ${SEEDS} \
  --ft-batch-size 256 \
  --ft-epochs 12 \
  --ft-lr 3e-5 \
  ${EXTRA_FLAGS}

echo ""
echo "[HOÀN TẤT] Báo cáo Ablation đã xuất tại ${OUTPUT_DIR}/ablation/ (Tổng thời gian: ${SECONDS}s)."
