#!/usr/bin/env bash
# The nuPlan track: the challenge preset with the nuPlan cameras, on the MTGS assets under ALPASIM_NUPLAN_ROOT.

set -euo pipefail

export PRESET="+e2e_challenge_nuplan=dev +cameras=nuplan_3cam"
export CHECKPOINT_FILE="${CHECKPOINT_FILE:-outputs/checkpoints/nuplan_latent_transfuser/model.pth}"
export SCENE_IDS="[2021.05.25.14.16.10_veh-35_00083_00485-0c318d7923d15b78]"

"$(dirname "${BASH_SOURCE[0]}")/latent_transfuser.sh" "$@"
