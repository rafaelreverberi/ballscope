#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/setup_$(date +%Y%m%d_%H%M%S).log"
touch "${LOG_FILE}"
exec > >(tee -a "${LOG_FILE}") 2>&1

if [[ -t 1 ]]; then
  C_RESET="\033[0m"
  C_BOLD="\033[1m"
  C_BLUE="\033[34m"
  C_GREEN="\033[32m"
  C_YELLOW="\033[33m"
  C_RED="\033[31m"
else
  C_RESET=""
  C_BOLD=""
  C_BLUE=""
  C_GREEN=""
  C_YELLOW=""
  C_RED=""
fi

section() { printf "\n${C_BOLD}${C_BLUE}==> %s${C_RESET}\n" "$*"; }
info() { printf "[INFO] %s\n" "$*"; }
ok() { printf "${C_GREEN}[OK]${C_RESET} %s\n" "$*"; }
warn() { printf "${C_YELLOW}[WARN]${C_RESET} %s\n" "$*"; }
fail() { printf "${C_RED}[ERROR]${C_RESET} %s\n" "$*" >&2; exit 1; }

on_error() {
  local exit_code=$?
  printf "${C_RED}[ERROR]${C_RESET} Setup failed (exit=${exit_code}). Check log: %s\n" "${LOG_FILE}" >&2
  exit "${exit_code}"
}
trap on_error ERR

is_jetson() {
  if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "aarch64" ]]; then
    return 1
  fi
  if [[ -f /etc/nv_tegra_release ]]; then
    return 0
  fi
  if [[ -f /proc/device-tree/model ]] && grep -qi 'jetson' /proc/device-tree/model; then
    return 0
  fi
  return 1
}

detect_platform() {
  case "$(uname -s)-$(uname -m)" in
    Darwin-arm64)
      echo "mac_apple_silicon"
      ;;
    Linux-aarch64)
      if is_jetson; then
        echo "jetson"
      else
        echo "unsupported"
      fi
      ;;
    *)
      echo "unsupported"
      ;;
  esac
}

describe_host() {
  local platform="$1"
  if [[ "${platform}" == "mac_apple_silicon" ]]; then
    local model chip
    model="$(sysctl -n hw.model 2>/dev/null || echo 'unknown-mac-model')"
    chip="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'Apple Silicon')"
    printf "Mac Apple Silicon | model=%s | chip=%s\n" "${model}" "${chip}"
    return
  fi

  if [[ "${platform}" == "jetson" ]]; then
    local model
    model="unknown-jetson"
    if [[ -f /proc/device-tree/model ]]; then
      model="$(tr -d '\0' </proc/device-tree/model 2>/dev/null || echo 'unknown-jetson')"
    fi
    printf "NVIDIA Jetson | model=%s\n" "${model}"
    return
  fi

  printf "unsupported host\n"
}

show_jetson_wheel_help_and_exit() {
  warn "Jetson detected, but no CUDA-enabled Torch wheel is configured."
  warn "Torch wheels are NOT bundled in this repository."
  printf "\nPlaceholders for now (implement auto-download later):\n"
  printf "  1) Copy %s/jetson_torch_wheels.example.env to %s/jetson_torch_wheels.env\n" "${ROOT_DIR}" "${ROOT_DIR}"
  printf "  2) Fill in TORCH_WHEEL_URL (and optional TORCHVISION_WHEEL_URL)\n"
  printf "  3) Run ./setup.sh again\n\n"
  exit 2
}

section "BallScope Installer"
info "Log file: ${LOG_FILE}"

PLATFORM="$(detect_platform)"
if [[ "${PLATFORM}" == "unsupported" ]]; then
  fail "Supported platforms: Apple Silicon macOS + NVIDIA Jetson (Linux/aarch64). Intel Mac is intentionally not supported."
fi

ok "Device detected: $(describe_host "${PLATFORM}")"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  fail "${PYTHON_BIN} not found. Install Python 3.10+ first."
fi
ok "Python found: $("${PYTHON_BIN}" --version 2>&1)"

if [[ "${PLATFORM}" == "jetson" ]]; then
  section "Jetson System Dependencies"
  if command -v apt-get >/dev/null 2>&1; then
    info "Installing apt packages (sudo required)..."
    sudo apt-get update
    sudo apt-get install -y \
      python3-venv \
      python3-pip \
      ffmpeg \
      v4l-utils \
      gstreamer1.0-tools \
      gstreamer1.0-plugins-base \
      gstreamer1.0-plugins-good \
      gstreamer1.0-plugins-bad \
      gstreamer1.0-plugins-ugly \
      libportaudio2
    ok "Jetson system packages installed"
  else
    warn "apt-get not found; skipping system packages"
  fi
fi

if [[ "${PLATFORM}" == "mac_apple_silicon" ]]; then
  section "macOS Dependencies"
  if command -v brew >/dev/null 2>&1; then
    info "Installing Homebrew packages (ffmpeg, portaudio)..."
    brew install ffmpeg portaudio || true
    ok "Homebrew package step completed"
  else
    warn "Homebrew not found. If needed, install ffmpeg and portaudio manually."
  fi
fi

section "Python Environment"
if [[ ! -d "${VENV_DIR}" ]]; then
  info "Creating virtual environment at ${VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  ok "Virtual environment created"
else
  info "Using existing virtual environment: ${VENV_DIR}"
fi

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip setuptools wheel
ok "pip/setuptools/wheel upgraded"

if [[ "${PLATFORM}" == "mac_apple_silicon" ]]; then
  section "Install Python Packages (Mac Apple Silicon)"
  python -m pip install -r "${ROOT_DIR}/requirements-mac-apple-silicon.txt"
  ok "Python packages installed for Apple Silicon"
fi

if [[ "${PLATFORM}" == "jetson" ]]; then
  section "Jetson Torch Wheel"

  if [[ -f "${ROOT_DIR}/jetson_torch_wheels.env" ]]; then
    info "Loading wheel config from jetson_torch_wheels.env"
    # shellcheck disable=SC1091
    source "${ROOT_DIR}/jetson_torch_wheels.env"
  else
    warn "No jetson_torch_wheels.env found."
  fi

  if [[ -n "${TORCH_WHEEL_URL:-}" ]]; then
    info "Installing torch wheel from TORCH_WHEEL_URL"
    python -m pip install "${TORCH_WHEEL_URL}"
  elif [[ -n "${TORCH_WHEEL_PATH:-}" ]]; then
    info "Installing torch wheel from TORCH_WHEEL_PATH"
    python -m pip install "${TORCH_WHEEL_PATH}"
  else
    show_jetson_wheel_help_and_exit
  fi

  if [[ -n "${TORCHVISION_WHEEL_URL:-}" ]]; then
    info "Installing torchvision wheel from TORCHVISION_WHEEL_URL"
    python -m pip install "${TORCHVISION_WHEEL_URL}"
  elif [[ -n "${TORCHVISION_WHEEL_PATH:-}" ]]; then
    info "Installing torchvision wheel from TORCHVISION_WHEEL_PATH"
    python -m pip install "${TORCHVISION_WHEEL_PATH}"
  else
    warn "No torchvision wheel configured (optional)."
  fi

  section "Install Python Packages (Jetson)"
  python -m pip install -r "${ROOT_DIR}/requirements-jetson.txt"
  ok "Python packages installed for Jetson"
fi

section "Verification"
python - <<'PY'
import cv2
import fastapi
import ultralytics
import torch
from ballscope.runtime_device import resolve_torch_device

print('imports: OK')
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if hasattr(torch.backends, 'mps'):
    print('mps available:', torch.backends.mps.is_available())
print('auto device:', resolve_torch_device('auto', torch))
print('opencv:', cv2.__version__)
print('fastapi:', fastapi.__version__)
print('ultralytics:', ultralytics.__version__)
PY
ok "Verification successful"

section "Done"
ok "Setup completed successfully"
info "Activate env: source .venv/bin/activate"
info "Start app: python main.py"
printf "\n${C_BOLD}Don't forget:${C_RESET} activate the virtual environment before starting BallScope.\n"
printf "Run:\n"
printf "  source .venv/bin/activate\n"
printf "  python main.py\n"
info "Full log: ${LOG_FILE}"
