#!/usr/bin/env bash
# Launch the Ovis2.5-MLX inference daemon on localhost:8000.
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
export OVIS_MLX_DIR="${OVIS_MLX_DIR:-$PWD/ovis_mlx_model}"
export OVIS_HF_DIR="${OVIS_HF_DIR:-$PWD/Ovis2.5-2B}"
# OVIS_MAX_PIXELS caps incoming images (unified-memory OOM guardrail).
export OVIS_MAX_PIXELS="${OVIS_MAX_PIXELS:-$((1280 * 1280))}"
exec uvicorn server.server:app --host 127.0.0.1 --port 8000
