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

MODE="multivariate"
HORIZON="24"
CONFIG_PATH="configs/default.yaml"
RAW_EXTRA_ARGS=()

# Hỗ trợ cả cách gọi vị trí ($1=mode, $2=horizon, $3=config, $4=flags)
# lẫn gọi trực tiếp cờ CLI (--include-wp02-baseline / --ckpt-source wp02)
if [[ $# -gt 0 ]]; then
  if [[ "$1" == -* ]]; then
    RAW_EXTRA_ARGS=("$@")
  else
    MODE="$1"
    shift
    if [[ $# -gt 0 && -n "$1" ]]; then
      if [[ "$1" == -* ]]; then
        RAW_EXTRA_ARGS+=("$@")
      else
        HORIZON="$1"
        shift
        if [[ $# -gt 0 && -n "$1" ]]; then
          if [[ "$1" == -* ]]; then
            RAW_EXTRA_ARGS+=("$@")
          else
            CONFIG_PATH="$1"
            shift
            if [[ $# -gt 0 ]]; then
              read -r -a _flags <<<"$*"
              RAW_EXTRA_ARGS+=("${_flags[@]}")
            fi
          fi
        fi
      fi
    fi
  fi
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[LỖI] Không tìm thấy tệp cấu hình: ${CONFIG_PATH}" >&2
  exit 1
fi

for req_file in "utils_metrics.py" "train_baseline.py" "training_loop.py" "fine_tune.py"; do
  if [[ ! -f "${req_file}" ]]; then
    echo "[LỖI] Thiếu tệp bắt buộc tại thư mục gốc: ${req_file}" >&2
    exit 1
  fi
done

if [[ ! -f "data/preprocess/train.csv" ]]; then
  echo "[CẢNH BÁO] Không thấy data/preprocess/train.csv. Hãy chắc chắn bạn đã chạy pipeline WP01 (preprocess.py) trước." >&2
fi

# Phân luồng tham số riêng cho Bước 1 (training_loop.py) và Bước 2 (fine_tune.py)
PRETRAIN_ARGS=()
FINETUNE_ARGS=()
USE_WP02_SOURCE=false
HAS_WP02_FLAG=false

idx=0
while [[ $idx -lt ${#RAW_EXTRA_ARGS[@]} ]]; do
  arg="${RAW_EXTRA_ARGS[$idx]}"
  case "${arg}" in
  --include-wp02-baseline)
    PRETRAIN_ARGS+=("${arg}")
    HAS_WP02_FLAG=true
    idx=$((idx + 1))
    ;;
  --dlinear-lr | --dropout | --warmup-ratio | --min-lr-ratio | --max-grad-norm)
    PRETRAIN_ARGS+=("${arg}" "${RAW_EXTRA_ARGS[$((idx + 1))]}")
    idx=$((idx + 2))
    ;;
  --seeds)
    PRETRAIN_ARGS+=("${arg}")
    idx=$((idx + 1))
    first_seed=""
    while [[ $idx -lt ${#RAW_EXTRA_ARGS[@]} && "${RAW_EXTRA_ARGS[$idx]}" != -* ]]; do
      if [[ -z "${first_seed}" ]]; then
        first_seed="${RAW_EXTRA_ARGS[$idx]}"
      fi
      PRETRAIN_ARGS+=("${RAW_EXTRA_ARGS[$idx]}")
      idx=$((idx + 1))
    done
    if [[ -n "${first_seed}" ]]; then
      FINETUNE_ARGS+=("--seed" "${first_seed}")
    fi
    ;;
  --unfreeze-backbone)
    FINETUNE_ARGS+=("${arg}")
    idx=$((idx + 1))
    ;;
  --ckpt-source)
    val="${RAW_EXTRA_ARGS[$((idx + 1))]}"
    FINETUNE_ARGS+=("${arg}" "${val}")
    if [[ "${val}" == "wp02" ]]; then
      USE_WP02_SOURCE=true
    fi
    idx=$((idx + 2))
    ;;
  --wp02-ckpt-dir | --seed | --stride)
    FINETUNE_ARGS+=("${arg}" "${RAW_EXTRA_ARGS[$((idx + 1))]}")
    idx=$((idx + 2))
    ;;
  --stations)
    FINETUNE_ARGS+=("${arg}")
    idx=$((idx + 1))
    while [[ $idx -lt ${#RAW_EXTRA_ARGS[@]} && "${RAW_EXTRA_ARGS[$idx]}" != -* ]]; do
      FINETUNE_ARGS+=("${RAW_EXTRA_ARGS[$idx]}")
      idx=$((idx + 1))
    done
    ;;
  *)
    PRETRAIN_ARGS+=("${arg}")
    FINETUNE_ARGS+=("${arg}")
    idx=$((idx + 1))
    ;;
  esac
done

# Nếu Fine-tune yêu cầu checkpoint từ WP02, tự động bật chạy kèm WP02 Baseline ở Bước 1
if [[ "${USE_WP02_SOURCE}" == true && "${HAS_WP02_FLAG}" == false ]]; then
  PRETRAIN_ARGS+=("--include-wp02-baseline")
fi

SECONDS=0

echo "========================================================================"
echo "[BƯỚC 1/2] GLOBAL PRE-TRAINING (12 TRẠM) | Mode=${MODE} | H=${HORIZON}"
echo "========================================================================"
"${PYTHON_BIN}" training_loop.py \
  --config "${CONFIG_PATH}" \
  --mode "${MODE}" \
  --horizon "${HORIZON}" \
  "${PRETRAIN_ARGS[@]}"

echo ""
echo "========================================================================"
echo "[BƯỚC 2/2] PER-STATION FINE-TUNING (TỪNG TRẠM) | Mode=${MODE} | H=${HORIZON}"
echo "========================================================================"
"${PYTHON_BIN}" fine_tune.py \
  --config "${CONFIG_PATH}" \
  --mode "${MODE}" \
  --horizon "${HORIZON}" \
  "${FINETUNE_ARGS[@]}"

echo ""
echo "[HOÀN TẤT] Toàn bộ Pre-training, Fine-tuning & xuất ExperimentMetricsContract xong sau ${SECONDS} giây."
