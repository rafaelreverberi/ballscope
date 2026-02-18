# AGENTS.md

Instructions for AI coding agents working in this repository.

## Mission
Keep BallScope fully functional on both target platforms:
- Apple Silicon macOS (`arm64`)
- NVIDIA Jetson (`Linux aarch64`)

Do not optimize one platform by breaking the other.

## Hard Compatibility Rules
- Preserve one shared codebase for app behavior.
- Keep runtime behavior equivalent across Mac and Jetson whenever possible.
- Platform-specific code is allowed only where required (camera backend, Torch wheel install, hardware acceleration).
- Intel macOS is intentionally unsupported; do not add Intel-specific support unless explicitly requested.

## Device and Acceleration Rules
- Default inference mode must remain `BALLSCOPE_AI_DEVICE=auto`.
- In `auto` mode:
  - Jetson should prefer `cuda:0` when available.
  - Apple Silicon should prefer `mps` when available.
  - fallback is `cpu`.
- Do not hardcode CUDA-only logic in shared paths.

## Camera Rules
- Jetson defaults should remain Linux device paths (`/dev/video*`).
- Mac defaults should remain numeric camera indices (`0`, `1`, ...).
- UI/device source hints should mention both formats.

## Setup and Dependencies
- Keep Python packages installed into local `.venv` via `setup.sh`.
- Keep installer logs in `logs/setup_*.log`.
- Jetson Torch wheels are external and must not be committed to the repo.
- Keep `jetson_torch_wheels.example.env` as the placeholder template.

## Change Management
When making changes that touch runtime/platform behavior:
1. Verify Mac and Jetson compatibility assumptions.
2. Update docs (`README.md`, `docs/setup.md`, and relevant platform docs).
3. Avoid regressions in existing Jetson capture/recording workflows.

## Testing Expectations
At minimum after platform-related changes:
- run Python syntax checks on edited modules
- verify app startup path (`python main.py`) assumptions remain valid
- verify device-resolution logic still maps `auto` correctly

## Documentation Language
Keep all project documentation in English unless explicitly requested otherwise.
