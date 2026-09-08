#!/usr/bin/env bash
# Drives one Alpasim scene through a public scripts/evaluation/alpasim script and checks the score and how far the ego drove.
# Usage: eval_alpasim_scene.sh <checkpoint dir> <script name> <scene id> <output dir> <min score> <min meters>
# DRIVE_TIMEOUT_S caps one attempt.

set -euo pipefail

_checkpoint_dir=$1
_script=$2
_scene_id=$3
_out_dir=$4
_min_score=$5
_min_meters=$6
_drive_timeout_s="${DRIVE_TIMEOUT_S:-1800}"
_attempts=3

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
# Port-block locks live under /tmp so every runner user sees them and parallel jobs never collide.
mkdir -p /tmp/py123d_garage_dev_ports && chmod 1777 /tmp/py123d_garage_dev_ports 2>/dev/null || true
_project="ci-${RUNNER_NAME:-local}"
_project="${_project,,}"

# Containers of this runner's previous job, and of any job that died on this machine; jobs cap at 150 min.
docker ps -aq --filter "name=^${_project}-" | xargs -r docker rm -f >/dev/null
docker ps -a --filter "name=^ci-" --filter "name=^attempt_" --format '{{.ID}} {{.CreatedAt}}' |
	while read -r _id _date _time _zone _rest; do
		if [[ $(($(date +%s) - $(date -d "${_date} ${_time} ${_zone}" +%s))) -gt 10800 ]]; then
			docker rm -f "${_id}" >/dev/null
		fi
	done

# "<score> <meters>" of the best finished rollout, empty while none finished; meters are driven by the driver itself.
finished_drive() {
	lib/alpasim/alpasim/.venv/bin/python - "$1" <<'PY'
import json, sys
from pathlib import Path
import polars as pl
aggregate = Path(sys.argv[1]) / "aggregate"
summary = aggregate / "results-summary.json"
rollouts = json.loads(summary.read_text())["rollouts"] if summary.exists() else []
scores = {rollout["rollout_id"]: rollout["score"] for rollout in rollouts if rollout["score"] is not None and rollout["metrics"]}

def series(rows, name):
    return rows.filter(pl.col("name") == name).sort("timestamps_us")["values"].to_list()

if scores:
    metrics = pl.read_parquet(aggregate / "metrics_unprocessed.parquet")
    drives = []
    for rollout_id, score in scores.items():
        rows = metrics.filter(pl.col("rollout_id") == rollout_id)
        driven, distance = series(rows, "eval_relevant"), series(rows, "dist_traveled_m")
        event = [collision > 0 or offroad > 0 for collision, offroad in zip(series(rows, "collision_any"), series(rows, "offroad"), strict=True)]
        start = driven.index(1.0)
        end = next((index for index in range(start, len(event)) if event[index]), len(event) - 1)
        drives.append((distance[end] - distance[start], score))
    meters, score = max(drives)
    print(f"{score:.3f} {meters:.1f}")
PY
}

_drive=""
for _attempt in $(seq 1 "${_attempts}"); do
	_attempt_dir="${_out_dir}/attempt_${_attempt}"
	mkdir -p "${_attempt_dir}"
	# A fresh block per attempt: the services bind fixed ports on the host network from the block's start, the driver sits at 80.
	# A block counts as taken while a socket in any state still uses one of its ports.
	exec 9<&-
	while :; do
		_base=$((20000 + 100 * (RANDOM % 128)))
		touch "/tmp/py123d_garage_dev_ports/${_base}.lock" 2>/dev/null || true
		exec 9<"/tmp/py123d_garage_dev_ports/${_base}.lock"
		flock -n 9 || continue
		ss -tan | grep -qE ":($(seq -s "|" "${_base}" $((_base + 9)))|$((_base + 80))) " || break
	done
	_baseport=${_base}
	_port=$((_base + 80))
	echo "Attempt ${_attempt}: driver port ${_port}, service ports from ${_baseport}"

	# Host and GPU memory once a second: the peaks tell whether an attempt died of memory. Both cover the whole host.
	: >"${_attempt_dir}/memory_samples.txt"
	(
		while :; do
			echo "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -n 1) $(free -m | awk '/^Mem:/ {print $3}')" >>"${_attempt_dir}/memory_samples.txt"
			sleep 1
		done
	) &
	_sampler_pid=$!
	CHECKPOINT_FILE="${_checkpoint_dir}/model.pth" OUTPUT_DIR="${_attempt_dir}" SCENE_IDS="[${_scene_id}]" ALPASIM_DRIVER_PORT="${_port}" \
		COMPOSE_PROJECT_NAME="${_project}-a${_attempt}" \
		timeout "${_drive_timeout_s}" bash "scripts/evaluation/alpasim/${_script}" \
		wizard.baseport="${_baseport}" || echo "Drive exited with code $?"
	kill "${_sampler_pid}" 2>/dev/null || true
	_peak=$(awk 'BEGIN {gpu = 0; ram = 0} {if ($1 > gpu) gpu = $1; if ($2 > ram) ram = $2} END {printf "peak GPU %d MiB, RAM %d MiB", gpu, ram}' "${_attempt_dir}/memory_samples.txt")
	echo "Attempt ${_attempt}: ${_peak}"
	# A killed wizard leaves its services running.
	docker ps -aq --filter "name=^${_project}-a${_attempt}-" | xargs -r docker rm -f >/dev/null
	_drive=$(finished_drive "${_attempt_dir}")
	[[ -n "${_drive}" ]] && break
done

if [[ -z "${_drive}" ]]; then
	echo "Scene ${_scene_id} did not finish in ${_attempts} attempts" >&2
	exit 1
fi

read -r _score _meters <<<"${_drive}"
echo "Scene ${_scene_id}: score ${_score}, minimum ${_min_score}; drove ${_meters} m, minimum ${_min_meters} m"
if [[ -n ${GITHUB_STEP_SUMMARY:-} ]]; then
	echo "### $(basename "${_checkpoint_dir}") on ${_scene_id}: score ${_score}, drove ${_meters} m, ${_peak}" >>"${GITHUB_STEP_SUMMARY}"
fi
python3 -c "import sys; sys.exit(0 if float('${_score}') >= float('${_min_score}') else 1)" ||
	{ echo "the score is below the minimum" >&2; exit 1; }
python3 -c "import sys; sys.exit(0 if float('${_meters}') >= float('${_min_meters}') else 1)" ||
	{ echo "the drive is shorter than the minimum" >&2; exit 1; }
