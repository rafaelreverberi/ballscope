#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
ACTIVATE_SCRIPT="${VENV_DIR}/bin/activate"

detect_platform() {
  case "$(uname -s)-$(uname -m)" in
    Darwin-arm64)
      echo "mac_apple_silicon"
      ;;
    Linux-aarch64)
      echo "linux_aarch64"
      ;;
    *)
      echo "unsupported"
      ;;
  esac
}

PLATFORM="$(detect_platform)"
if [[ "${PLATFORM}" == "unsupported" ]]; then
  echo "[ERROR] Unsupported platform. BallScope supports Apple Silicon macOS and Linux aarch64 (Jetson)." >&2
  exit 1
fi

if [[ ! -f "${ACTIVATE_SCRIPT}" ]]; then
  echo "[ERROR] Virtual environment not found at ${VENV_DIR}" >&2
  echo "[INFO] Run ./setup.sh first to create and configure .venv." >&2
  exit 1
fi

cd "${ROOT_DIR}"
# shellcheck source=/dev/null
source "${ACTIVATE_SCRIPT}"
exec python "${ROOT_DIR}/main.py" "$@"
