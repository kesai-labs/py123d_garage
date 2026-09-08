#!/bin/bash

set -euo pipefail

ALPASIM_COMMIT="54952f4a3a58d6a8b917fa61b52785c93a954619"

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

# 3. Install py123d_garage into the Alpasim's virtual environment
uv pip install --python .venv/bin/python --exclude-newer "$(date -u +%FT%TZ)" -e "${_garage_root}"
uv run python -c "import py123d_garage.evaluation.alpasim.evaluate"
