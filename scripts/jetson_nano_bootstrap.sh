#!/usr/bin/env bash
set -euo pipefail

# Defensive deployment bootstrap for Jetson Nano / Orin Nano.
# This script starts ROS2 vision + spoof manager nodes in DRY-RUN mode.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

if [[ -z "${VIRTUAL_ENV:-}" && -d "${ROOT_DIR}/.venv" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.venv/bin/activate"
fi

echo "[jetson-bootstrap] starting vision node..."
python -m drone_interceptor.ros2.vision_node --config configs/default.yaml &
VISION_PID=$!

echo "[jetson-bootstrap] starting spoof node (defensive dry-run)..."
python -m drone_interceptor.ros2.spoof_node --config configs/default.yaml --spoof-enable &
SPOOF_PID=$!

cleanup() {
  echo "[jetson-bootstrap] stopping nodes..."
  kill "${VISION_PID}" "${SPOOF_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[jetson-bootstrap] nodes running. press Ctrl+C to stop."
wait
