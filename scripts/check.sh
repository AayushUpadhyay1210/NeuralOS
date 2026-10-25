#!/usr/bin/env bash
# Runs every automated check: backend tests, lint, types; frontend tests, lint, types, build.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
python -m pytest
ruff check backend tests
mypy
(cd frontend && npm run typecheck && npm run lint && npm test && npm run build)
echo "All checks passed."
