#!/bin/bash

set -euo pipefail

ALPASIM_COMMIT="cd713e0d0563352ce45b99d0a53a7173d581bde2"

cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
_garage_root="$PWD"

# 1. Clone the Alpasim repo
if [ ! -d lib/alpasim/alpasim/.git ]; then
  git clone --branch e2e_challenge https://github.com/NVlabs/alpasim.git lib/alpasim/alpasim
fi
cd lib/alpasim/alpasim
git remote set-url origin https://github.com/NVlabs/alpasim.git
git fetch --quiet origin e2e_challenge
git checkout --quiet --detach "${ALPASIM_COMMIT}"
git --no-pager log -1 --format='%h %ad %s' --date=short

# 2. Install the Alpasim's virtual environment
uv sync --extra all
# uv rebuilds alpasim_grpc, whose build hook compiles the protos, only when src/grpc/pyproject.toml changes.
uv run --no-sync compile-protos

# 3. Install py123d_garage into the Alpasim's virtual environment
# HACK: remove the exclude-newer when py123d is old enough.
uv pip install --python .venv/bin/python --exclude-newer "$(date -u +%FT%TZ)" -e "${_garage_root}"
uv run python -c "import py123d_garage.evaluation.alpasim.evaluate"
