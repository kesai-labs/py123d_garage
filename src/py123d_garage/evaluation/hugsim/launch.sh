#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

# HUGSIM's closed_loop.py runs this with the CUDA device and the episode folder of one episode.
set -euo pipefail

# The policy runs in the py123d_garage environment, from the checkout this launcher is in. The simulator's PYTHONPATH
# would shadow its modules, so only the HUGSIM checkout's agent protocol (agents/client.py) joins it.
_src_dir="$(realpath "$(dirname "${BASH_SOURCE[0]}")/../../..")"
PYTHONPATH="${_src_dir}:${HUGSIM_ROOT}" exec "${PY123D_GARAGE_PYTHON}" -m py123d_garage.evaluation.hugsim.agent "$@"
