#!/usr/bin/env bash

set -euo pipefail

# One thread per process: parallelism comes from running several routes at once.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 NUMBA_THREADING_LAYER=workqueue

# Declare input files relative to py123d_garage
_checkpoint_file="${CHECKPOINT_FILE:-outputs/checkpoints/carla_transfuser/model.pth}"
_sensor_rig_file="${SENSOR_RIG_FILE:-$(dirname "${_checkpoint_file}")/sensor_rig_0.yaml}"
_route_file="${ROUTE_FILE:-lib/carla/scenes/bench2drive/1711.xml}"

# Run the evaluation in CARLA
python -m py123d_garage.evaluation.carla.evaluate \
	hydra.run.dir='outputs/evaluation/carla/transfuser/${now:%Y.%m.%d.%H.%M.%S}' \
	policy_config.evaluation_checkpoint_file="${_checkpoint_file}" \
	policy_config.evaluation_sensor_rig_file="${_sensor_rig_file}" \
	routes_file="${_route_file}" \
	benchmark=bench2drive \
	port="${CARLA_PORT:-2000}" \
	traffic_manager_port="${CARLA_TM_PORT:-8000}" \
	"$@"
