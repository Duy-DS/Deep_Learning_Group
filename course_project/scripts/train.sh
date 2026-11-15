#!/usr/bin/env bash
set -euo pipefail

# Tự động chuyển về thư mục gốc dự án (cha của thư mục scripts/)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Cấu hình mặc định (có thể truyền qua tham số: ./scripts/run_train_and_finetune.sh multivariate 24)
MODE="${1:-multivariate}"
HORIZON="${2:-24}"
DATA_DIR="data/raw"
OUTPUT_DIR="outputs"
MODELS="lstm dlinear informer patchtst"
SEEDS="42 1024 2026"

SECONDS=0

echo "========================================================================"
echo "[BƯỚC 1/2] GLOBAL PRE-TRAINING (12 TRẠM) | Mode=${MODE} | H=${HORIZON}"
echo "========================================================================"
python training_loop.py \
  --data-dir "${DATA_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --models ${MODELS} \
  --mode "${MODE}" \
  --lookback 96 \
  --horizon "${HORIZON}" \
  --batch-size 512 \
  --train-stride 6 \
  --lr 5e-5 \
  --weight-decay 1e-2 \
  --dropout 0.3 \
  --epochs 20 \
  --patience 6 \
  --seeds ${SEEDS}

echo ""
echo "========================================================================"
echo "[BƯỚC 2/2] PER-STATION FINE-TUNING (TỪNG TRẠM) | Mode=${MODE} | H=${HORIZON}"
echo "========================================================================"
python fine_tune.py \
  --data-dir "${DATA_DIR}" \
  --output-root "${OUTPUT_DIR}" \
  --models ${MODELS} \
  --mode "${MODE}" \
  --lookback 96 \
  --horizon "${HORIZON}" \
  --stride 6 \
  --batch-size 256 \
  --lr 3e-5 \
  --weight-decay 1e-2 \
  --epochs 12 \
  --patience 4 \
  --seed 42

echo ""
echo "[HOÀN TẤT] Toàn bộ Pre-training & Fine-tuning xong sau ${SECONDS} giây."
