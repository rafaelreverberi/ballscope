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

HF_MODELS_REPO="RafaelReverberi/ballscope-assets"
HF_MODELS_FOLDER="models"
HF_JETSON_WHEELS_REPO="RafaelReverberi/ballscope-jetson-wheels"
HF_JETSON_WHEELS_FOLDER="wheels"

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

TOTAL_STEPS=7
CURRENT_STEP=0
AUTOSTART_RESULT="skipped"

section() { printf "\n${C_BOLD}${C_BLUE}==> %s${C_RESET}\n" "$*"; }
info() { printf "[INFO] %s\n" "$*"; }
ok() { printf "${C_GREEN}[OK]${C_RESET} %s\n" "$*"; }
warn() { printf "${C_YELLOW}[WARN]${C_RESET} %s\n" "$*"; }
fail() { printf "${C_RED}[ERROR]${C_RESET} %s\n" "$*" >&2; exit 1; }

step_section() {
  local title="$1"
  CURRENT_STEP=$((CURRENT_STEP + 1))
  local pct=$((CURRENT_STEP * 100 / TOTAL_STEPS))
  local filled=$((pct / 5))
  local empty=$((20 - filled))
  local fill_bar empty_bar
  fill_bar="$(printf '%*s' "${filled}" '' | tr ' ' '#')"
  empty_bar="$(printf '%*s' "${empty}" '' | tr ' ' '-')"
  section "[${CURRENT_STEP}/${TOTAL_STEPS}] ${title} [${fill_bar}${empty_bar}] ${pct}%"
}

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

count_top_level_matches() {
  local dir="$1"
  local pattern="$2"
  if [[ ! -d "${dir}" ]]; then
    echo "0"
    return
  fi
  find "${dir}" -maxdepth 1 -type f -name "${pattern}" | wc -l | tr -d ' '
}

choose_existing_install_action() {
  prompt() { printf "%s" "$*" >&2; }
  if [[ -n "${BALLSCOPE_EXISTING_INSTALL_ACTION:-}" ]]; then
    case "${BALLSCOPE_EXISTING_INSTALL_ACTION}" in
      continue|reinstall|abort)
        echo "${BALLSCOPE_EXISTING_INSTALL_ACTION}"
        return 0
        ;;
      *)
        fail "Invalid BALLSCOPE_EXISTING_INSTALL_ACTION='${BALLSCOPE_EXISTING_INSTALL_ACTION}'. Use: continue | reinstall | abort"
        ;;
    esac
  fi

  if [[ ! -t 0 ]]; then
    warn "Non-interactive shell detected. Existing install action defaults to: continue"
    echo "continue"
    return 0
  fi

  prompt "\n"
  prompt "${C_BOLD}Action required:${C_RESET} Existing setup artifacts were detected.\n"
  prompt "Choose how to continue:\n"
  prompt "  [1] Continue with current setup files (default)\n"
  prompt "      Use this if you only want to update/check setup.\n"
  prompt "  [2] Reinstall clean (delete setup artifacts and rebuild)\n"
  prompt "      Deletes: .venv, models/*.pt, wheels/*.whl\n"
  prompt "  [3] Abort now\n"
  prompt "      Exit without changing anything.\n"
  prompt "Input now: 1 / 2 / 3 (default 1): "

  local choice
  read -r choice
  if [[ -z "${choice}" ]]; then
    choice="1"
  fi
  case "${choice}" in
    1) echo "continue" ;;
    2) echo "reinstall" ;;
    3) echo "abort" ;;
    *) fail "Invalid choice '${choice}'. Please run ./setup.sh again and enter 1, 2, or 3." ;;
  esac
}

cleanup_setup_artifacts() {
  warn "Reinstall selected. Removing setup artifacts now..."
  rm -rf "${VENV_DIR}"
  find "${ROOT_DIR}/models" -maxdepth 1 -type f -name '*.pt' -delete 2>/dev/null || true
  find "${ROOT_DIR}/wheels" -maxdepth 1 -type f -name '*.whl' -delete 2>/dev/null || true
  ok "Clean reinstall state prepared (.venv removed, models/*.pt and wheels/*.whl removed)"
}

choose_autostart_action() {
  prompt() { printf "%s" "$*" >&2; }
  if [[ -n "${BALLSCOPE_AUTOSTART_ACTION:-}" ]]; then
    case "${BALLSCOPE_AUTOSTART_ACTION}" in
      enable|skip)
        echo "${BALLSCOPE_AUTOSTART_ACTION}"
        return 0
        ;;
      *)
        fail "Invalid BALLSCOPE_AUTOSTART_ACTION='${BALLSCOPE_AUTOSTART_ACTION}'. Use: enable | skip"
        ;;
    esac
  fi

  if [[ ! -t 0 ]]; then
    warn "Non-interactive shell detected. Autostart action defaults to: skip"
    echo "skip"
    return 0
  fi

  prompt "\n"
  prompt "${C_BOLD}Action required:${C_RESET} Configure system autostart for BallScope?\n"
  prompt "Choose:\n"
  prompt "  [1] Yes, enable autostart (overwrites existing BallScope autostart service)\n"
  prompt "  [2] No, skip (default)\n"
  prompt "Input now: 1 / 2 (default 2): "

  local choice
  read -r choice
  if [[ -z "${choice}" ]]; then
    choice="2"
  fi

  case "${choice}" in
    1) echo "enable" ;;
    2) echo "skip" ;;
    *) fail "Invalid choice '${choice}'. Please run ./setup.sh again and enter 1 or 2." ;;
  esac
}

normalize_repo_permissions_if_sudo() {
  if [[ "${EUID}" -ne 0 || -z "${SUDO_USER:-}" || "${SUDO_USER}" == "root" ]]; then
    return 0
  fi
  if [[ "${BALLSCOPE_SKIP_CHOWN:-0}" == "1" ]]; then
    warn "Skipping ownership normalization because BALLSCOPE_SKIP_CHOWN=1"
    return 0
  fi

  local target_user target_group
  target_user="${SUDO_USER}"
  target_group="$(id -gn "${target_user}" 2>/dev/null || echo "${target_user}")"
  info "Normalizing repository ownership to ${target_user}:${target_group} (sudo run detected)"
  chown -R "${target_user}:${target_group}" "${ROOT_DIR}"
  ok "Repository ownership normalized for ${target_user}"
}

install_autostart_linux() {
  local service_name service_path run_user run_group
  service_name="ballscope.service"
  service_path="/etc/systemd/system/${service_name}"
  run_user="${SUDO_USER:-$(id -un)}"
  run_group="$(id -gn "${run_user}" 2>/dev/null || echo "${run_user}")"

  if ! command -v systemctl >/dev/null 2>&1; then
    warn "systemctl not found; skipping Linux autostart setup."
    return 1
  fi

  if [[ "${EUID}" -ne 0 ]]; then
    warn "Linux autostart setup requires sudo/root. Run setup with sudo to install systemd autostart."
    return 1
  fi

  cat > "${service_path}" <<EOF
[Unit]
Description=BallScope Autostart Service
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${run_user}
Group=${run_group}
WorkingDirectory=${ROOT_DIR}
ExecStart=${ROOT_DIR}/start.sh
Restart=always
RestartSec=2
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

  systemctl daemon-reload
  systemctl enable --now "${service_name}"
  ok "Linux autostart enabled via systemd (${service_name})"
  return 0
}

install_power_sudoers_linux() {
  local run_user sudoers_path tmpfile
  run_user="${SUDO_USER:-$(id -un)}"
  sudoers_path="/etc/sudoers.d/ballscope-power"

  if [[ "${EUID}" -ne 0 ]]; then
    warn "Power-control sudoers setup requires root; skipping."
    return 1
  fi

  tmpfile="$(mktemp)"
  cat > "${tmpfile}" <<EOF
# BallScope power-control permissions (installed by setup.sh)
${run_user} ALL=(root) NOPASSWD: /usr/bin/systemctl reboot, /usr/bin/systemctl poweroff, /bin/systemctl reboot, /bin/systemctl poweroff, /sbin/shutdown -r now, /sbin/shutdown -h now, /usr/sbin/shutdown -r now, /usr/sbin/shutdown -h now
EOF

  if command -v visudo >/dev/null 2>&1; then
    if ! visudo -cf "${tmpfile}" >/dev/null; then
      rm -f "${tmpfile}"
      warn "Power-control sudoers validation failed; skipping."
      return 1
    fi
  fi

  install -o root -g root -m 0440 "${tmpfile}" "${sudoers_path}"
  rm -f "${tmpfile}"
  ok "Linux power-control sudoers installed (${sudoers_path}) for user ${run_user}"
  return 0
}

install_autostart_macos() {
  local label target_user target_uid target_home agent_dir plist_path
  label="com.ballscope.start"
  target_user="${SUDO_USER:-$(id -un)}"
  target_uid="$(id -u "${target_user}" 2>/dev/null || true)"
  target_home="$(eval echo "~${target_user}")"

  if [[ -z "${target_uid}" || ! -d "${target_home}" ]]; then
    warn "Could not resolve macOS user context for autostart. Skipping."
    return 1
  fi

  agent_dir="${target_home}/Library/LaunchAgents"
  plist_path="${agent_dir}/${label}.plist"
  mkdir -p "${agent_dir}"

  cat > "${plist_path}" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
  <dict>
    <key>Label</key>
    <string>${label}</string>
    <key>ProgramArguments</key>
    <array>
      <string>/bin/bash</string>
      <string>-lc</string>
      <string>cd "${ROOT_DIR}" &amp;&amp; ./start.sh</string>
    </array>
    <key>WorkingDirectory</key>
    <string>${ROOT_DIR}</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>${ROOT_DIR}/logs/autostart_stdout.log</string>
    <key>StandardErrorPath</key>
    <string>${ROOT_DIR}/logs/autostart_stderr.log</string>
  </dict>
</plist>
EOF

  chown "${target_user}":"$(id -gn "${target_user}" 2>/dev/null || echo staff)" "${plist_path}" || true

  if [[ "${EUID}" -eq 0 ]]; then
    launchctl bootout "gui/${target_uid}/${label}" >/dev/null 2>&1 || true
    if launchctl bootstrap "gui/${target_uid}" "${plist_path}" >/dev/null 2>&1; then
      launchctl enable "gui/${target_uid}/${label}" >/dev/null 2>&1 || true
      ok "macOS autostart enabled via LaunchAgent (${label}) for user ${target_user}"
      return 0
    fi
  else
    launchctl bootout "gui/${target_uid}/${label}" >/dev/null 2>&1 || true
    if launchctl bootstrap "gui/${target_uid}" "${plist_path}" >/dev/null 2>&1; then
      launchctl enable "gui/${target_uid}/${label}" >/dev/null 2>&1 || true
      ok "macOS autostart enabled via LaunchAgent (${label})"
      return 0
    fi
  fi

  warn "Autostart plist was written, but launchctl could not bootstrap it in this session."
  warn "You can load it manually later: launchctl bootstrap gui/${target_uid} ${plist_path}"
  return 1
}

configure_optional_autostart() {
  step_section "Autostart (Optional)"
  local action
  action="$(choose_autostart_action)"
  info "Selected autostart action: ${action}"
  if [[ "${action}" != "enable" ]]; then
    info "Autostart setup skipped."
    AUTOSTART_RESULT="skipped"
    return 0
  fi

  if [[ "${PLATFORM}" == "jetson" ]]; then
    if install_autostart_linux; then
      if ! install_power_sudoers_linux; then
        warn "Autostart enabled, but power-control sudoers setup failed."
      fi
      AUTOSTART_RESULT="enabled"
    else
      AUTOSTART_RESULT="failed"
    fi
    return 0
  fi
  if [[ "${PLATFORM}" == "mac_apple_silicon" ]]; then
    if install_autostart_macos; then
      AUTOSTART_RESULT="enabled"
    else
      AUTOSTART_RESULT="failed"
    fi
    return 0
  fi
  AUTOSTART_RESULT="unsupported"
  warn "Autostart is not supported on this platform."
}

print_final_summary() {
  section "Summary"
  local model_count wheel_count
  model_count="$(count_top_level_matches "${ROOT_DIR}/models" '*.pt')"
  wheel_count="$(count_top_level_matches "${ROOT_DIR}/wheels" '*.whl')"

  info "Platform: ${PLATFORM}"
  info "Venv: ${VENV_DIR}"
  info "Python: $(python --version 2>&1)"
  info "Models in ${ROOT_DIR}/models: ${model_count}"
  info "Wheels in ${ROOT_DIR}/wheels: ${wheel_count}"
  info "Autostart setup result: ${AUTOSTART_RESULT}"

  section "Proof / Checks"
  if [[ "${PLATFORM}" == "jetson" ]] && command -v systemctl >/dev/null 2>&1; then
    info "Run these checks on Jetson:"
    printf "  systemctl is-enabled ballscope.service\n"
    printf "  systemctl is-active ballscope.service\n"
    printf "  systemctl status ballscope.service --no-pager\n"
  elif [[ "${PLATFORM}" == "mac_apple_silicon" ]]; then
    local check_user check_uid
    check_user="${SUDO_USER:-$(id -un)}"
    check_uid="$(id -u "${check_user}" 2>/dev/null || true)"
    info "Run these checks on macOS:"
    printf "  launchctl print gui/%s/com.ballscope.start\n" "${check_uid:-<uid>}"
    printf "  ls -l ~/Library/LaunchAgents/com.ballscope.start.plist\n"
  fi
}

ensure_hf_folder_files() {
  local repo_id="$1"
  local remote_folder="$2"
  local suffix="$3"
  local dest_dir="$4"
  local label="$5"

  mkdir -p "${dest_dir}"

  python - "$repo_id" "$remote_folder" "$suffix" "$dest_dir" "$label" <<'PY'
import json
import os
import ssl
import sys
import urllib.error
import urllib.parse
import urllib.request

repo_id, remote_folder, suffix, dest_dir, label = sys.argv[1:6]
api_url = f"https://huggingface.co/api/models/{repo_id}/tree/main/{remote_folder}?recursive=1"

try:
    import certifi  # type: ignore
    ssl_context = ssl.create_default_context(cafile=certifi.where())
except Exception:
    ssl_context = ssl.create_default_context()

try:
    with urllib.request.urlopen(api_url, context=ssl_context) as response:
        entries = json.load(response)
except urllib.error.URLError as exc:
    print(f"[ERROR] Failed to query Hugging Face API: {exc}", file=sys.stderr)
    print("[ERROR] TLS certificate validation failed. On macOS, ensure Python trust store is configured.", file=sys.stderr)
    sys.exit(2)

files = [
    entry["path"]
    for entry in entries
    if entry.get("type") == "file" and entry.get("path", "").endswith(suffix)
]

if not files:
    print(f"[ERROR] No '{suffix}' files found in {repo_id}/{remote_folder}", file=sys.stderr)
    sys.exit(3)

files.sort()
print(f"[INFO] {label}: found {len(files)} file(s) in {repo_id}/{remote_folder}")

for index, rel_path in enumerate(files, start=1):
    filename = os.path.basename(rel_path)
    out_path = os.path.join(dest_dir, filename)
    encoded_path = urllib.parse.quote(rel_path, safe='/')
    file_url = f"https://huggingface.co/{repo_id}/resolve/main/{encoded_path}?download=true"

    try:
      with urllib.request.urlopen(file_url, context=ssl_context) as response:
          total = int(response.headers.get("Content-Length", "0"))
          done = 0
          with open(out_path, "wb") as fh:
              while True:
                  chunk = response.read(1024 * 1024)
                  if not chunk:
                      break
                  fh.write(chunk)
                  done += len(chunk)
                  if total > 0:
                      pct = int(done * 100 / total)
                      bar_len = 20
                      filled = int(bar_len * done / total)
                      bar = "#" * filled + "-" * (bar_len - filled)
                      print(f"\r[INFO] [{index}/{len(files)}] {filename} [{bar}] {pct}%", end="", flush=True)
                  else:
                      print(f"\r[INFO] [{index}/{len(files)}] {filename} ({done} bytes)", end="", flush=True)
          print()
    except urllib.error.URLError as exc:
      print(f"[ERROR] Failed to download {rel_path}: {exc}", file=sys.stderr)
      sys.exit(4)

print(f"[OK] {label}: downloaded to {dest_dir}")
PY
}

download_models_from_hf() {
  step_section "Download Models"
  info "Downloading .pt files from https://huggingface.co/${HF_MODELS_REPO}/tree/main/${HF_MODELS_FOLDER}"
  ensure_hf_folder_files "${HF_MODELS_REPO}" "${HF_MODELS_FOLDER}" ".pt" "${ROOT_DIR}/models" "Model assets"
  ok "Model download completed"
}

verify_torch_cuda() {
  python - <<'PY'
import sys
import torch

print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit(8)
PY
}

choose_jetson_torch_mode() {
  prompt() { printf "%s" "$*" >&2; }
  if [[ -n "${BALLSCOPE_JETSON_TORCH_MODE:-}" ]]; then
    case "${BALLSCOPE_JETSON_TORCH_MODE}" in
      manual|hf|preinstalled)
        echo "${BALLSCOPE_JETSON_TORCH_MODE}"
        return 0
        ;;
      *)
        fail "Invalid BALLSCOPE_JETSON_TORCH_MODE='${BALLSCOPE_JETSON_TORCH_MODE}'. Use: manual | hf | preinstalled"
        ;;
    esac
  fi

  if [[ ! -t 0 ]]; then
    warn "Non-interactive shell detected. Defaulting to BALLSCOPE_JETSON_TORCH_MODE=hf"
    echo "hf"
    return 0
  fi

  prompt "\n"
  prompt "${C_BOLD}Action required:${C_RESET} Jetson needs CUDA-enabled PyTorch before other Python dependencies.\n"
  prompt "Choose setup mode:\n"
  prompt "  [1] I will install PyTorch manually in .venv\n"
  prompt "      Setup exits after venv creation/activation instructions.\n"
  prompt "  [2] Download and install PyTorch wheels from Hugging Face (recommended)\n"
  prompt "      Wheels are downloaded to wheels/ and installed automatically.\n"
  prompt "  [3] PyTorch is already installed in .venv\n"
  prompt "      Setup verifies CUDA and then continues.\n"
  prompt "Input now: 1 / 2 / 3 (default 2): "

  local choice
  read -r choice
  if [[ -z "${choice}" ]]; then
    choice="2"
  fi
  case "${choice}" in
    1) echo "manual" ;;
    2) echo "hf" ;;
    3) echo "preinstalled" ;;
    *) fail "Invalid choice '${choice}'. Please run ./setup.sh again and enter 1, 2, or 3." ;;
  esac
}

install_jetson_torch_from_hf_wheels() {
  step_section "Install Jetson PyTorch (HF Wheels)"
  info "Downloading .whl files from https://huggingface.co/${HF_JETSON_WHEELS_REPO}/tree/main/${HF_JETSON_WHEELS_FOLDER}"
  ensure_hf_folder_files "${HF_JETSON_WHEELS_REPO}" "${HF_JETSON_WHEELS_FOLDER}" ".whl" "${ROOT_DIR}/wheels" "Jetson wheel assets"

  local torch_wheel torchvision_wheel
  torch_wheel="$(find "${ROOT_DIR}/wheels" -maxdepth 1 -type f -name 'torch-*.whl' | sort | head -n 1 || true)"
  torchvision_wheel="$(find "${ROOT_DIR}/wheels" -maxdepth 1 -type f -name 'torchvision-*.whl' | sort | head -n 1 || true)"

  [[ -n "${torch_wheel}" ]] || fail "No torch-*.whl found in ${ROOT_DIR}/wheels after download"

  info "Installing torch wheel first: $(basename "${torch_wheel}")"
  python -m pip install "${torch_wheel}"

  if [[ -n "${torchvision_wheel}" ]]; then
    info "Installing torchvision wheel: $(basename "${torchvision_wheel}")"
    python -m pip install "${torchvision_wheel}"
  else
    warn "No torchvision-*.whl found in ${ROOT_DIR}/wheels (optional)."
  fi

  if ! verify_torch_cuda; then
    fail "PyTorch installed but CUDA is not available. Verify JetPack/L4T-compatible wheels."
  fi
  ok "Jetson PyTorch + CUDA verified"
}

step_section "BallScope Installer"
info "Log file: ${LOG_FILE}"

PLATFORM="$(detect_platform)"
if [[ "${PLATFORM}" == "unsupported" ]]; then
  fail "Supported platforms: Apple Silicon macOS + NVIDIA Jetson (Linux/aarch64). Intel Mac is intentionally not supported."
fi
ok "Device detected: $(describe_host "${PLATFORM}")"
if [[ "${PLATFORM}" == "jetson" && "${EUID}" -ne 0 ]]; then
  warn "Jetson setup is recommended with sudo: sudo ./setup.sh"
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  fail "${PYTHON_BIN} not found. Install Python 3.10+ first."
fi
ok "Python found: $("${PYTHON_BIN}" --version 2>&1)"

EXISTING_VENV=0
EXISTING_MODELS_COUNT="$(count_top_level_matches "${ROOT_DIR}/models" '*.pt')"
EXISTING_WHEELS_COUNT="$(count_top_level_matches "${ROOT_DIR}/wheels" '*.whl')"
if [[ -d "${VENV_DIR}" ]]; then
  EXISTING_VENV=1
fi

if [[ "${EXISTING_VENV}" -eq 1 || "${EXISTING_MODELS_COUNT}" -gt 0 || "${EXISTING_WHEELS_COUNT}" -gt 0 ]]; then
  section "Existing Installation Detected"
  if [[ "${EXISTING_VENV}" -eq 1 ]]; then
    warn "Found existing virtual environment: ${VENV_DIR}"
  fi
  if [[ "${EXISTING_MODELS_COUNT}" -gt 0 ]]; then
    warn "Found existing model files: ${EXISTING_MODELS_COUNT} file(s) in ${ROOT_DIR}/models"
  fi
  if [[ "${EXISTING_WHEELS_COUNT}" -gt 0 ]]; then
    warn "Found existing wheel files: ${EXISTING_WHEELS_COUNT} file(s) in ${ROOT_DIR}/wheels"
  fi

  EXISTING_ACTION="$(choose_existing_install_action)"
  info "Selected existing-install action: ${EXISTING_ACTION}"
  case "${EXISTING_ACTION}" in
    continue)
      info "Continuing with existing setup artifacts."
      ;;
    reinstall)
      cleanup_setup_artifacts
      ;;
    abort)
      warn "Setup aborted by user."
      exit 0
      ;;
    *)
      fail "Unsupported existing-install action: ${EXISTING_ACTION}"
      ;;
  esac
fi

if [[ "${PLATFORM}" == "jetson" ]]; then
  TOTAL_STEPS=10
else
  TOTAL_STEPS=8
fi

if [[ "${PLATFORM}" == "jetson" ]]; then
  step_section "Jetson System Dependencies"
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
  step_section "macOS Dependencies"
  BREW_BIN=""
  if command -v brew >/dev/null 2>&1; then
    BREW_BIN="$(command -v brew)"
  elif [[ -x /opt/homebrew/bin/brew ]]; then
    BREW_BIN="/opt/homebrew/bin/brew"
  elif [[ -x /usr/local/bin/brew ]]; then
    BREW_BIN="/usr/local/bin/brew"
  fi

  if [[ -n "${BREW_BIN}" ]]; then
    info "Installing Homebrew packages (ffmpeg, gstreamer, plugins, portaudio)..."
    if [[ "${EUID}" -eq 0 && -n "${SUDO_USER:-}" && "${SUDO_USER}" != "root" ]]; then
      info "macOS setup is running with sudo; executing Homebrew install as ${SUDO_USER}"
      if sudo -u "${SUDO_USER}" -H "${BREW_BIN}" install \
        ffmpeg \
        gstreamer \
        gst-plugins-base \
        gst-plugins-good \
        gst-plugins-bad \
        gst-plugins-ugly \
        portaudio; then
        ok "Homebrew package step completed"
      else
        warn "Homebrew package install failed. Check the log above."
      fi
    elif [[ "${EUID}" -eq 0 ]]; then
      warn "Running as root without SUDO_USER; cannot safely run Homebrew install automatically."
      warn "Run without sudo on macOS or install Homebrew packages manually."
    elif "${BREW_BIN}" install \
      ffmpeg \
      gstreamer \
      gst-plugins-base \
      gst-plugins-good \
      gst-plugins-bad \
      gst-plugins-ugly \
      portaudio; then
      ok "Homebrew package step completed"
    else
      warn "Homebrew package install failed. Check the log above."
    fi
  else
    warn "Homebrew not found. If needed, install ffmpeg, gstreamer (+ plugins), and portaudio manually."
  fi
fi

step_section "Python Environment"
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

if [[ "${PLATFORM}" == "jetson" ]]; then
  step_section "Jetson PyTorch Choice"
  JETSON_TORCH_MODE="$(choose_jetson_torch_mode)"
  info "Selected mode: ${JETSON_TORCH_MODE}"

  case "${JETSON_TORCH_MODE}" in
    manual)
      warn "Setup stops now so you can install CUDA-enabled PyTorch manually in .venv."
      printf "\nNext steps:\n"
      printf "  1) source .venv/bin/activate\n"
      printf "  2) Install torch/torchvision manually (Jetson CUDA wheels)\n"
      printf "  3) Run ./setup.sh again and choose option [3] to continue\n\n"
      info "No models or extra dependencies were installed in manual mode."
      info "Full log: ${LOG_FILE}"
      exit 0
      ;;
    hf)
      install_jetson_torch_from_hf_wheels
      ;;
    preinstalled)
      step_section "Verify Existing Jetson PyTorch"
      if ! verify_torch_cuda; then
        fail "PyTorch in .venv has no CUDA. Install Jetson-compatible CUDA wheels, then run setup again."
      fi
      ok "Existing PyTorch + CUDA verified"
      ;;
    *)
      fail "Unsupported Jetson torch mode: ${JETSON_TORCH_MODE}"
      ;;
  esac

  step_section "Install Python Packages (Jetson)"
  python -m pip install -r "${ROOT_DIR}/requirements-jetson.txt"
  ok "Python packages installed for Jetson"
fi

if [[ "${PLATFORM}" == "mac_apple_silicon" ]]; then
  step_section "Install Python Packages (Mac Apple Silicon)"
  python -m pip install -r "${ROOT_DIR}/requirements-mac-apple-silicon.txt"
  ok "Python packages installed for Apple Silicon"
fi

download_models_from_hf

step_section "Verification"
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

configure_optional_autostart
normalize_repo_permissions_if_sudo

step_section "Done"
ok "Setup completed successfully"
info "Activate env: source .venv/bin/activate"
info "Start app: python main.py"
info "One-liner from project root: ./start.sh"
printf "\n${C_BOLD}Don't forget:${C_RESET} activate the virtual environment before starting BallScope.\n"
printf "Run:\n"
printf "  source .venv/bin/activate\n"
printf "  python main.py\n"
printf "\nOr use the one-liner from project root:\n"
printf "  ./start.sh\n"
print_final_summary
info "Full log: ${LOG_FILE}"
