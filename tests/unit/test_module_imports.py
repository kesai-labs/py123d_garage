# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""Import every package module: catches broken imports, layer typos, and hook breakage."""

from __future__ import annotations

import importlib
import pkgutil

import py123d_garage


def test_every_module_imports() -> None:
    prefix = py123d_garage.__name__ + "."
    failures = []
    for module_info in pkgutil.walk_packages(py123d_garage.__path__, prefix):
        try:
            importlib.import_module(module_info.name)
        except ModuleNotFoundError as error:  # noqa: PERF203
            # The CARLA and AlpaSim evaluation modules import packages that only
            # those simulators' runtimes provide; the custom kernels need Triton.
            if error.name not in ("carla", "leaderboard", "alpasim_driver", "alpasim_grpc", "triton"):
                failures.append(f"{module_info.name}: {error!r}")
        except Exception as error:
            failures.append(f"{module_info.name}: {error!r}")
    assert not failures, "modules failed to import:\n" + "\n".join(failures)
