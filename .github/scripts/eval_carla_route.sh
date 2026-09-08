#!/usr/bin/env bash
# Boots CARLA, drives one Bench2Drive route through scripts/evaluation/carla/transfuser.sh, and checks the score.
# Usage: eval_carla_route.sh <checkpoint dir> <route id> <output dir> <min score>
# CARLA_ROOT is the installed simulator, DRIVE_TIMEOUT_S caps one attempt.

set -euo pipefail

_checkpoint_dir=$1
_route_id=$2
_out_dir=$3
_min_score=$4
_carla_root="${CARLA_ROOT:-/opt/carla/0915}"
_drive_timeout_s="${DRIVE_TIMEOUT_S:-1800}"
_attempts=3
_boot_timeout_s=90

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
# Port-block locks live under /tmp so every runner user sees them and parallel jobs never collide.
mkdir -p /tmp/py123d_garage_dev_ports && chmod 1777 /tmp/py123d_garage_dev_ports 2>/dev/null || true
mkdir -p lib/carla/simulator
ln -sfn "${_carla_root}" lib/carla/simulator/0915

carla_answers() {
	python -c "import carla; c = carla.Client('localhost', $1); c.set_timeout(5.0); c.get_world().get_snapshot()" 2>/dev/null
}

# The record of a finished drive, empty while the route crashed or never ran.
finished_record() {
	python - "$1" <<'PY'
import json, sys
from pathlib import Path
result = Path(sys.argv[1]) / "results.json"
records = json.loads(result.read_text())["_checkpoint"]["records"] if result.exists() else []
for record in records:
    if record["status"] in ("Completed", "Perfect"):
        print(f"{record['status']} {record['scores']['score_composed']:.2f}")
PY
}

_carla_pid=""
kill_carla() {
	[[ -n "${_carla_pid}" ]] && kill -9 -- "-${_carla_pid}" 2>/dev/null || true
	_carla_pid=""
}
trap kill_carla EXIT

_record=""
for _attempt in $(seq 1 "${_attempts}"); do
	# A fresh block per attempt: CARLA takes its first three ports, the traffic manager sits at 50.
	# A block counts as taken while a socket in any state still uses one of its ports.
	exec 9<&-
	while :; do
		_base=$((20000 + 100 * (RANDOM % 128)))
		touch "/tmp/py123d_garage_dev_ports/${_base}.lock" 2>/dev/null || true
		exec 9<"/tmp/py123d_garage_dev_ports/${_base}.lock"
		flock -n 9 || continue
		ss -tan | grep -qE ":(${_base}|$((_base + 1))|$((_base + 2))|$((_base + 50))) " || break
	done
	_port=${_base}
	_streaming_port=$((_base + 1))
	_tm_port=$((_base + 50))
	_attempt_dir="${_out_dir}/attempt_${_attempt}"
	mkdir -p "${_attempt_dir}"
	echo "Attempt ${_attempt}: port ${_port}, traffic manager port ${_tm_port}, streaming port ${_streaming_port}"

	# setsid makes CARLA its own process group, so the kill takes down the engine CarlaUE4.sh starts as a child.
	setsid "${_carla_root}/CarlaUE4.sh" -world-port="${_port}" -carla-streaming-port="${_streaming_port}" \
		-nosound -RenderOffScreen -abslog="${_attempt_dir}/carla_ue4.log" &
	_carla_pid=$!

	_carla_ready=0
	for _second in $(seq 1 "${_boot_timeout_s}"); do
		sleep 1
		if ! kill -0 "${_carla_pid}" 2>/dev/null; then
			echo "CARLA died after ${_second} s"
			break
		fi
		if carla_answers "${_port}"; then
			echo "CARLA serving on port ${_port} after ${_second} s"
			_carla_ready=1
			break
		fi
	done

	if [[ "${_carla_ready}" -eq 1 ]]; then
		# Host and GPU memory once a second: the peaks tell whether an attempt died of memory. Both cover the whole host.
		: >"${_attempt_dir}/memory_samples.txt"
		(
			while :; do
				echo "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -n 1) $(free -m | awk '/^Mem:/ {print $3}')" >>"${_attempt_dir}/memory_samples.txt"
				sleep 1
			done
		) &
		_sampler_pid=$!
		CHECKPOINT_FILE="${_checkpoint_dir}/model.pth" ROUTE_FILE="lib/carla/scenes/bench2drive/${_route_id}.xml" CARLA_PORT="${_port}" CARLA_TM_PORT="${_tm_port}" \
			timeout "${_drive_timeout_s}" bash scripts/evaluation/carla/transfuser.sh \
			hydra.run.dir="${_attempt_dir}" || echo "Drive exited with code $?"
		kill "${_sampler_pid}" 2>/dev/null || true
		_peak=$(awk 'BEGIN {gpu = 0; ram = 0} {if ($1 > gpu) gpu = $1; if ($2 > ram) ram = $2} END {printf "peak GPU %d MiB, RAM %d MiB", gpu, ram}' "${_attempt_dir}/memory_samples.txt")
		echo "Attempt ${_attempt}: ${_peak}"
		_record=$(finished_record "${_attempt_dir}")
	fi
	kill_carla
	[[ -n "${_record}" ]] && break
done

if [[ -z "${_record}" ]]; then
	echo "Route ${_route_id} did not finish in ${_attempts} attempts" >&2
	exit 1
fi

_status=${_record% *}
_score=${_record#* }
echo "Route ${_route_id}: ${_status}, driving score ${_score}, minimum ${_min_score}"
if [[ -n ${GITHUB_STEP_SUMMARY:-} ]]; then
	echo "### $(basename "${_checkpoint_dir}") on route ${_route_id}: ${_status}, driving score ${_score}, ${_peak}" >>"${GITHUB_STEP_SUMMARY}"
fi
python -c "import sys; sys.exit(0 if float('${_score}') >= float('${_min_score}') else 1)" ||
	{ echo "the driving score is below the minimum" >&2; exit 1; }
