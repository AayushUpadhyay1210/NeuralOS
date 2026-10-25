#!/usr/bin/env bash
# NeuralOS setup for Linux and macOS. Run from the repository root:  bash scripts/setup.sh
# Pass --no-rl to skip PyTorch / Stable-Baselines3 (baselines and the UI still work).
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python3}"
WITH_RL=1
[[ "${1:-}" == "--no-rl" ]] && WITH_RL=0

"$PY" -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10+ required"' 
[ -d .venv ] || "$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip

if [ "$WITH_RL" = 1 ]; then
  if [ "$(uname -s)" = "Linux" ]; then
    # CPU-only wheel avoids ~3 GB of CUDA libraries. Falls back to PyPI if the index is unreachable.
    pip install torch --index-url https://download.pytorch.org/whl/cpu || pip install torch
  fi
  pip install -e "backend[rl,dev]"
else
  pip install -e "backend[dev]"
fi

(cd frontend && npm ci)
echo
echo "Setup complete. Activate the environment with: source .venv/bin/activate"
