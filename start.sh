#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
ACTIVATE_SCRIPT="${VENV_DIR}/bin/activate"
BALLSCOPE_PORT="${BALLSCOPE_PORT:-8000}"

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

find_listener_pids() {
  local port="$1"
  if command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:"${port}" -sTCP:LISTEN -t 2>/dev/null || true
    return
  fi
  if command -v fuser >/dev/null 2>&1; then
    fuser -n tcp "${port}" 2>/dev/null | tr ' ' '\n' | sed '/^$/d' || true
    return
  fi
  echo ""
}

show_process_info() {
  local pid="$1"
  if ps -p "${pid}" -o pid= -o comm= >/dev/null 2>&1; then
    ps -p "${pid}" -o pid= -o comm=
  else
    echo "${pid} (unknown)"
  fi
}

free_uvicorn_port() {
  local port="$1"
  mapfile -t pids < <(find_listener_pids "${port}")
  if [[ "${#pids[@]}" -eq 0 ]]; then
    echo "[INFO] Port ${port} is free."
    return 0
  fi

  echo "[WARN] Port ${port} is already in use. Stopping listener process(es):"
  for pid in "${pids[@]}"; do
    echo "  - $(show_process_info "${pid}")"
  done

  for pid in "${pids[@]}"; do
    kill "${pid}" 2>/dev/null || true
  done
  sleep 1

  mapfile -t remaining < <(find_listener_pids "${port}")
  if [[ "${#remaining[@]}" -gt 0 ]]; then
    echo "[WARN] Graceful stop failed for port ${port}, forcing kill:"
    for pid in "${remaining[@]}"; do
      echo "  - $(show_process_info "${pid}")"
      kill -9 "${pid}" 2>/dev/null || true
    done
    sleep 1
  fi

  mapfile -t final < <(find_listener_pids "${port}")
  if [[ "${#final[@]}" -gt 0 ]]; then
    echo "[ERROR] Could not free port ${port}. Still active:"
    for pid in "${final[@]}"; do
      echo "  - $(show_process_info "${pid}")"
    done
    return 1
  fi

  echo "[OK] Port ${port} is now free."
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
free_uvicorn_port "${BALLSCOPE_PORT}"
exec python "${ROOT_DIR}/main.py" "$@"
