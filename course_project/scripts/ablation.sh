#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON_BIN="python"
if ! command -v python >/dev/null 2>&1 && command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

ABLATIONS="all"
CONFIG_PATH="configs/default.yaml"
EXTRA_ARGS=()

# Hỗ trợ cả cách gọi cũ ($1=ablations, $2=extra_flags, $3=config)
# lẫn gọi trực tiếp cờ CLI (--with-wp02-baseline --with-finetune)
if [[ $# -gt 0 ]]; then
  if [[ "$1" == -* ]]; then
    EXTRA_ARGS=("$@")
  else
    ABLATIONS="$1"
    shift
    if [[ $# -gt 0 && -n "$1" ]]; then
      read -r -a _flags <<<"$1"
      EXTRA_ARGS+=("${_flags[@]}")
      shift
    fi
    if [[ $# -gt 0 && -n "$1" ]]; then
      if [[ "$1" == -* ]]; then
        EXTRA_ARGS+=("$@")
      else
        CONFIG_PATH="$1"
        shift
        if [[ $# -gt 0 ]]; then
          EXTRA_ARGS+=("$@")
        fi
      fi
    fi
  fi
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[LỖI] Không tìm thấy tệp cấu hình: ${CONFIG_PATH}" >&2
  exit 1
fi

for req_file in "utils_metrics.py" "train_baseline.py" "training_loop.py" "fine_tune.py" "run_ablation.py"; do
  if [[ ! -f "${req_file}" ]]; then
    echo "[LỖI] Thiếu tệp bắt buộc tại thư mục gốc: ${req_file}" >&2
    exit 1
  fi
done

if [[ ! -f "data/preprocess/train.csv" ]]; then
  echo "[CẢNH BÁO] Không thấy data/preprocess/train.csv. Hãy chắc chắn bạn đã chạy pipeline WP01 (preprocess.py) trước." >&2
fi

SECONDS=0

echo "========================================================================"
echo "[ABLATION STUDIES & WP02 BENCHMARK] Thí nghiệm: ${ABLATIONS} ${EXTRA_ARGS[*]:-}"
echo "========================================================================"
"${PYTHON_BIN}" run_ablation.py \
  --config "${CONFIG_PATH}" \
  --ablations ${ABLATIONS} \
  "${EXTRA_ARGS[@]}"

echo ""
echo "[HOÀN TẤT] Báo cáo Markdown & ExperimentMetricsContract JSON đã xuất tại outputs/ablation/ (Thời gian: ${SECONDS}s)."
