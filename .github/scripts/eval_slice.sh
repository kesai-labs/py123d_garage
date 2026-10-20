#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

# Scores one checkpoint on a slice of one dataset with the open-loop or navsim script, and checks the error band.
# Usage: eval_slice.sh <protocol> <checkpoint file> <dataset> <num scenes> <output dir> [max ade m]

set -euo pipefail

_protocol=$1
_checkpoint_file=$2
_dataset=$3
_num_scenes=$4
_out_dir=$5
_max_ade_m=${6:-}

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# The slice comes from the dataset's own preset, so the served intervals and modalities match the checkpoint's config.
_source_override="$(python .github/scripts/eval_source.py "${_checkpoint_file}" "${_dataset}" "${_num_scenes}")"

CHECKPOINT_FILE="${_checkpoint_file}" bash "scripts/evaluation/${_protocol}.sh" \
	hydra.run.dir="${_out_dir}" \
	parallelization_config.inference_batch_size=2 \
	parallelization_config.max_workers=1 \
	"${_source_override}"

test -s "${_out_dir}/results.csv"
cat "${_out_dir}/results.csv"

# A band wide enough to pass anything that drives, so what it catches is a checkpoint that loads wrong or empty features.
if [[ -n ${_max_ade_m} ]]; then
	awk -F, -v max="${_max_ade_m}" '
		NR == 1 {
			for (column = 1; column <= NF; column++) {
				if ($column == "average_displacement_error_m") {
					ade_column = column
				}
			}
			if (!ade_column) {
				print "results.csv carries no average_displacement_error_m column" >"/dev/stderr"
				exit 1
			}
		}
		$1 == "average" {
			scored = 1
			printf "average ADE %.3f m, band %.3f m\n", $ade_column, max
			if ($ade_column >= max) {
				print "the average displacement error is outside the band" >"/dev/stderr"
				exit 1
			}
		}
		END {
			if (!scored) {
				print "results.csv carries no average row" >"/dev/stderr"
				exit 1
			}
		}
	' "${_out_dir}/results.csv"
fi

if [[ -n ${GITHUB_STEP_SUMMARY:-} ]]; then
	{
		echo "### $(basename "$(dirname "${_checkpoint_file}")") on ${_dataset}, ${_protocol}"
		echo '```'
		cat "${_out_dir}/results.csv"
		echo '```'
	} >>"${GITHUB_STEP_SUMMARY}"
fi
