#!/usr/bin/env bash
# Bridge a Cloudflare quick TCP tunnel to a local port and run the open-loop
# eval client against it.
#
# Usage:
#   ./run_client.sh
#
# Environment overrides:
#   TUNNEL_HOST    Cloudflare tunnel hostname (no scheme)
#   LOCAL_PORT     Local port to bind for the ZMQ client (default: 5555)
#   DATASET_PATH   Dataset directory (default: demo_data/cube_to_bowl_5)
#   EMBODIMENT_TAG Embodiment tag (default: REAL_G1)
#   TRAJ_IDS       Space-separated trajectory IDs (default: "1 2")
#   ACTION_HORIZON Action horizon (default: 8)

set -euo pipefail

TUNNEL_HOST="${TUNNEL_HOST:-giants-leaves-itunes-slide.trycloudflare.com}"
LOCAL_PORT="${LOCAL_PORT:-5555}"
DATASET_PATH="${DATASET_PATH:-demo_data/simplerenv_bridge_sample}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-REAL_G1}"
TRAJ_IDS="${TRAJ_IDS:-1 2}"
ACTION_HORIZON="${ACTION_HORIZON:-8}"

if ! command -v cloudflared >/dev/null 2>&1; then
  echo "ERROR: cloudflared is not installed. Install it from:"
  echo "  https://github.com/cloudflare/cloudflared/releases/latest"
  exit 1
fi

echo "Starting cloudflared bridge: 127.0.0.1:${LOCAL_PORT} -> ${TUNNEL_HOST}"
cloudflared access tcp \
  --hostname "${TUNNEL_HOST}" \
  --url "127.0.0.1:${LOCAL_PORT}" &
CF_PID=$!
trap 'kill ${CF_PID} 2>/dev/null || true' EXIT

# Give the bridge a moment to come up
sleep 3

if ! kill -0 "${CF_PID}" 2>/dev/null; then
  echo "ERROR: cloudflared bridge failed to start."
  exit 1
fi

echo "Bridge is up (pid=${CF_PID}). Launching client..."

uv run python gr00t/eval/open_loop_eval.py \
  --dataset-path "${DATASET_PATH}" \
  --embodiment-tag "${EMBODIMENT_TAG}" \
  --host 127.0.0.1 \
  --port "${LOCAL_PORT}" \
  --traj-ids ${TRAJ_IDS} \
  --action-horizon "${ACTION_HORIZON}"
