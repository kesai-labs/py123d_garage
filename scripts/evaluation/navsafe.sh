#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

set -euo pipefail

# Go to the root of the repository, so that relative paths work.
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# NavSafe drives the 280 scenario bundles of its download, with the actor assets and gait banks its edits insert. The
# renderer opens them by the paths the evaluator resolves, so both see the download's real path.
: "${NAVSAFE_DATA_ROOT:?set NAVSAFE_DATA_ROOT to the NavSafe download, with full_test, asset and gait_bank}"
NAVSAFE_DATA_ROOT="$(realpath "${NAVSAFE_DATA_ROOT}")"

# Declare input and output directories relative to py123d_garage; the checkpoint defaults to seed 0 of the newest posttrain run
_posttrain_run_dirs=(outputs/training/all/*/posttrain/)
_posttrain_checkpoint_files=("${_posttrain_run_dirs[-1]%/}"/model_0_*.pth)
_checkpoint_file="$(realpath "${CHECKPOINT_FILE:-${_posttrain_checkpoint_files[-1]}}")"
[[ -f "${_checkpoint_file}" ]] || { echo "missing ${_checkpoint_file}" >&2; exit 1; }
_out_dir="$(realpath -m "${OUTPUT_DIR:-outputs/evaluation/navsafe/$(date +%F_%H-%M-%S)}")"
# GPU indices as nvidia-smi lists them, and lanes of the sweep on each: a lane is a renderer and one episode at a time.
IFS=, read -r -a _gpus <<<"${GPUS:-0}"
_lane_gpus=()
for _gpu in "${_gpus[@]}"; do
	for ((_run = 0; _run < ${RUNS_PER_GPU:-1}; _run++)); do
		_lane_gpus+=("${_gpu}")
	done
done
_navsafe_root="${PWD}/lib/navsafe/navsafe"
mkdir -p "${_out_dir}/renderer_logs"

# The adapter of lib/navsafe/plugin runs the policy in this environment, from this checkout's sources, next to the
# checkout and environment of lib/navsafe/tools/setup_navsafe.sh.
PY123D_GARAGE_PYTHON="$(command -v python)"
PY123D_GARAGE_SRC="${PWD}/src"
NAVSAFE_PYTHON="${_navsafe_root}/.venv/bin/python"
NAVSAFE_OUT_ROOT="${_out_dir}"
NAVSAFE_EVAL_SEED="${EVAL_SEED:-0}"
export PY123D_GARAGE_PYTHON PY123D_GARAGE_SRC NAVSAFE_DATA_ROOT NAVSAFE_PYTHON NAVSAFE_OUT_ROOT NAVSAFE_EVAL_SEED

# Without RENDERER_SQSH, NavSafe starts a docker renderer per scenario (needs NGC_ENV_FILE). With the NRE image's
# enroot .sqsh file, each lane gets one renderer for all scenarios, an srun step of this Slurm job through pyxis.
_renderer_pids=()
trap 'kill "${_renderer_pids[@]}" 2>/dev/null || true' EXIT
if [[ -n "${RENDERER_SQSH:-}" ]]; then
	_harmonizer_cache="$(realpath -m "${HARMONIZER_CACHE:-${HOME}/.cache/navsafe/harmonizer}")"
	mkdir -p "${_harmonizer_cache}"
	_nre_runfiles=/app/internal/scripts/pycena/runtime/pycena_nrm_full.runfiles
	# Steps starting together would each unpack the image, so its one named container is created first.
	srun --overlap --nodes=1 --ntasks=1 --cpu-bind=none --container-image="${RENDERER_SQSH}" \
		--container-name=navsafe_nre true
	for _lane in "${!_lane_gpus[@]}"; do
		# The image's launcher finds its runfiles only through RUNFILES_DIR, and the host's HOME is not in the container.
		srun --overlap --nodes=1 --ntasks=1 --cpu-bind=none --container-image="${RENDERER_SQSH}" \
			--container-name=navsafe_nre:no_exec --container-writable \
			--container-mounts="${NAVSAFE_DATA_ROOT}:${NAVSAFE_DATA_ROOT}:ro,${_harmonizer_cache}:${_harmonizer_cache}" \
			--container-workdir="${_nre_runfiles}/_main" \
			--output="${_out_dir}/renderer_logs/lane_${_lane}.log" \
			bash -c "export CUDA_VISIBLE_DEVICES=${_lane_gpus[${_lane}]} HOME=/tmp XDG_CACHE_HOME=/tmp/.cache OMP_NUM_THREADS=1 \
				RUNFILES_DIR=${_nre_runfiles} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
				exec /app/internal/scripts/pycena/runtime/pycena_nrm_full serve-grpc --host 127.0.0.1 \
				--port $((8080 + _lane)) --enable-editing-actors --renderer default --cache-size 4 \
				--enable-harmonizer --harmonizer-cache ${_harmonizer_cache} \
				--artifact-glob '${NAVSAFE_DATA_ROOT}/full_test/*/*.usdz'" &
		_renderer_pids+=($!)
	done
fi

# NavSafe renders the cameras every 0.1 s step, but the policy reads them only when it plans: at frames 0 and 5 of the
# 8 warm-up frames, then every 5th frame from the hand-off at frame 8. The renderer skips the other steps and serves
# the last frame again. Set NUREC_GRPC_RENDER_STEPS empty to render every step.
export NUREC_GRPC_RENDER_STEPS="${NUREC_GRPC_RENDER_STEPS-0,5,$(seq -s, 8 5 1210)}"

# Isaac Sim keeps the GPU driver's shader cache in its environment (kit/cache/nv_shadercache). At the driver's 4 GB
# limit, concurrent episodes evict each other's ray-tracing pipelines and every start recompiles them for minutes.
export __GL_SHADER_DISK_CACHE_SIZE=100000000000 __GL_SHADER_DISK_CACHE_SKIP_CLEANUP=1

# Run the lanes, each on every n-th scenario of the sweep: the 280 bundles, edited where their frozen recipe inserts
# actors, with NavSafe's protocol (semi-reactive traffic, LQR tracking, 2 Hz replanning, contact ends the episode).
_lane_pids=()
for _lane in "${!_lane_gpus[@]}"; do
	(
		export NAVSAFE_SHARD="${_lane}/${#_lane_gpus[@]}" NAVSAFE_GPU="${_lane_gpus[${_lane}]}"
		# The libGLU that setup_navsafe.sh unpacks on hosts that lack it.
		if [[ -d "${PWD}/lib/navsafe/runtime/libglu" ]]; then
			export LD_LIBRARY_PATH="${PWD}/lib/navsafe/runtime/libglu/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
		fi
		if [[ -n "${RENDERER_SQSH:-}" ]]; then
			export NAVSAFE_RENDERER=existing NUREC_GRPC_HOST=127.0.0.1 NUREC_GRPC_PORT=$((8080 + _lane))
			# The renderer indexes every bundle before it opens its port.
			until (exec 3<>"/dev/tcp/127.0.0.1/${NUREC_GRPC_PORT}") 2>/dev/null; do
				kill -0 "${_renderer_pids[${_lane}]}" 2>/dev/null || { echo "renderer of lane ${_lane} exited" >&2; exit 1; }
				sleep 10
			done
		else
			export NAVSAFE_RENDERER=docker NAVSAFE_RENDER_GPU="${NAVSAFE_GPU}" NUREC_PORT=$((8088 + _lane))
		fi
		"${_navsafe_root}/.venv/bin/navsafe" benchmark py123d_garage "${_checkpoint_file}" "$@"
	) >"${_out_dir}/lane_${_lane}.log" 2>&1 &
	_lane_pids+=($!)
done
_failed_lanes=0
for _pid in "${_lane_pids[@]}"; do
	wait "${_pid}" || _failed_lanes=$((_failed_lanes + 1))
done

# Summarize the scored episodes of all lanes, then fail if a lane left scenarios unscored.
"${_navsafe_root}/.venv/bin/navsafe" report --run "${_out_dir}"
((_failed_lanes == 0)) || { echo "${_failed_lanes} lane(s) left scenarios unscored" >&2; exit 1; }
