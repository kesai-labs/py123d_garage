#!/bin/bash

set -euo pipefail

NAVSAFE_COMMIT="bd68458df1e2cc8e0f0dfcf62c06b71a86ae229c"

cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
_garage_root="$PWD"

# 1. Clone the NavSafe repo with the Isaac Lab submodule; the other submodules are the code of its VLA baselines
if [ ! -d lib/navsafe/navsafe/.git ]; then
  git clone https://github.com/VAIL-UCLA/NavSafe-infty.git lib/navsafe/navsafe
fi
cd lib/navsafe/navsafe
git cat-file -e "${NAVSAFE_COMMIT}^{commit}" 2>/dev/null || git fetch --quiet origin
git checkout --quiet --detach "${NAVSAFE_COMMIT}"
git --no-pager log -1 --format='%h %ad %s' --date=short
git submodule update --init third_party/IsaacLab

# 2. Install NavSafe's locked environment (Isaac Sim, Isaac Lab, torch), then the adapter that runs our policy
uv sync --locked --python 3.12
uv pip install --python .venv/bin/python --no-deps -e "${_garage_root}/lib/navsafe/plugin"

# 3. Isaac Sim's RTX renderer crashes at start without libGLU; on Ubuntu hosts that lack it, unpack the distro's package
# into lib/navsafe/runtime, which scripts/evaluation/navsafe.sh puts on the library path
_runtime="${_garage_root}/lib/navsafe/runtime"
if ! ldconfig -p | grep -q libGLU.so.1 && [ ! -e "${_runtime}/libglu/usr/lib/x86_64-linux-gnu/libGLU.so.1" ]; then
  mkdir -p "${_runtime}/libglu"
  (cd "${_runtime}" && apt-get download libglu1-mesa && dpkg-deb -x libglu1-mesa_*.deb libglu && rm libglu1-mesa_*.deb)
fi
