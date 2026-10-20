#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

set -euo pipefail

# Go to the root of the repository, so that relative paths work.
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# Declare the tag to submit under, headed by SUBMISSION_NAME (py123d-garage-ltfv6 unless set, empty for none). The
# challenge rejects a reused tag and a push overwrites the registry's, so every submission carries a version: VERSION,
# or a TAG that ends in one, which then names the local image too.
_name="${SUBMISSION_NAME-py123d-garage-ltfv6}"
_tag="${_name:+${_name}-}${TAG:-dev}${VERSION:+-${VERSION}}"
[[ "${_tag}" == *-v[0-9]* ]] || { echo "set VERSION, e.g. v1, or end TAG in -v<n>" >&2; exit 1; }

# Look up the team's registry namespace, which needs the login from login_challenge.sh.
# The API reports the team under registration only while a registration is attached to the account.
_me="$(uv run --project lib/alpasim/alpasim lib/alpasim/alpasim/e2e_challenge/competitor_cli/alpasim_challenge.py me)"
_team_id="$(printf '%s' "${_me}" | python3 -c 'import json, sys; me = json.load(sys.stdin); print((me.get("registration") or {}).get("team_id") or me.get("team_id") or "")')"
if [ -z "${_team_id}" ]; then
	echo "no team_id in the challenge account; check the registration and the accepted terms:" >&2
	printf '%s\n' "${_me}" >&2
	exit 1
fi
_uri="696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/${_team_id}:${_tag}"

# Push the image under its versioned tag and submit it to the track
uv run --project lib/alpasim/alpasim lib/alpasim/alpasim/e2e_challenge/competitor_cli/alpasim_challenge.py ecr-login
docker tag "py123d-garage-ltfv6:${TAG:-dev}" "${_uri}"
docker push "${_uri}"
uv run --project lib/alpasim/alpasim lib/alpasim/alpasim/e2e_challenge/competitor_cli/alpasim_challenge.py submit --track "${TRACK:-pai}" "${_uri}"
