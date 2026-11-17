#!/usr/bin/env bash
# Tải và xác thực FER2013 từ thư mục gốc dự án BTL02.
set -eu

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ -f "${PROJECT_ROOT}/.venv/Scripts/activate" ]]; then
    # Môi trường Windows khi script chạy trong Git Bash.
    # shellcheck disable=SC1091
    source "${PROJECT_ROOT}/.venv/Scripts/activate"
elif [[ -f "${PROJECT_ROOT}/.venv/bin/activate" ]]; then
    # Môi trường Unix/macOS.
    # shellcheck disable=SC1091
    source "${PROJECT_ROOT}/.venv/bin/activate"
fi

exec python -m src.data.downloader "$@"