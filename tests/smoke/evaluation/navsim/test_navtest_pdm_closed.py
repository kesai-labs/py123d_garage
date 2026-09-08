"""
End-to-end benchmark: PDM-Closed scored by PDMMetric on the navtest dataset.

Requires nuplan_test logs carrying navtest anchors under PY123D_GARAGE_DATA_ROOT/nuplan/123D.
Evaluates a deterministic slice of the split and asserts the scores stay in the
known-good band (full-navtest reference: PDMS 0.897).
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pytest
from py123d.api.scene.arrow.arrow_scene_builder import ArrowSceneBuilder
from py123d.api.scene.scene_filter import SceneFilter
from py123d.common.execution import ThreadPoolExecutor
from py123d.geometry.transform import abs_to_rel_se2_array

from py123d_garage.datatypes.trajectory import TrajectorySE2
from py123d_garage.evaluation.navsim.help.pdm_agent import PDMAgent
from py123d_garage.evaluation.navsim.help.pdm_metric import PDMMetric
from py123d_garage.py123d_help.scene_filters.filter_fns.scenes_subset import _navtest_scene_uuids

pytestmark = pytest.mark.slow

_NUM_SCENES = 200
_MAX_FAILED_SCENES = 5  # route/centerline edge cases fail ~0.8% of navtest


@pytest.fixture(scope="module")
def navtest_root() -> Path:
    data_root = os.environ.get("PY123D_GARAGE_DATA_ROOT")
    root = Path(data_root) / "nuplan" / "123D" if data_root else None
    if root is None or not (root / "logs").exists():
        pytest.skip(
            "nuplan not on disk (set PY123D_GARAGE_DATA_ROOT to the data dir holding nuplan/123D)",
        )
    return root


def test_pdm_closed_benchmark(navtest_root: Path) -> None:
    scene_filter = SceneFilter(
        scene_uuids=list(_navtest_scene_uuids()),
        history_duration_s=2.0,
        future_duration_s=5.0,
    )
    # Shuffle with a fixed seed so the slice spans many logs instead of the first log only.
    random.seed(0)
    scene_filter.shuffle = True
    scene_filter.max_num_scenes = _NUM_SCENES
    builder = ArrowSceneBuilder(
        logs_root=navtest_root / "logs",
        maps_root=navtest_root / "maps",
    )
    scenes = builder.get_scenes(
        filter=scene_filter,
        executor=ThreadPoolExecutor(max_workers=4),
    )
    assert len(scenes) == _NUM_SCENES, f"expected {_NUM_SCENES} scenes, built {len(scenes)}"

    metric = PDMMetric()
    results, failures = [], []
    for scene in scenes:
        agent = PDMAgent()
        agent.initialize()
        try:
            trajectory = agent.compute_trajectory(scene)
            assert isinstance(trajectory, TrajectorySE2)

            ego_state_se3 = scene.get_ego_state_se3_at_iteration(0)
            assert ego_state_se3 is not None
            # PDM-Closed unrolls a kinematic model, so hand over its own yaw rather than
            # re-deriving one from the path: this scores the planner, not the derivation.
            relative = TrajectorySE2(
                pose_se2_array=abs_to_rel_se2_array(
                    origin=ego_state_se3.rear_axle_se2,
                    pose_se2_array=trajectory.pose_se2_array,
                ),
                timestamps_us=trajectory.timestamps_us,
            )
            results.append(
                metric.compute_metric(scene, agent_trajectory=relative),
            )
        except Exception as error:
            failures.append((scene.scene_uuid, repr(error)))

    assert len(failures) <= _MAX_FAILED_SCENES, f"too many failed scenes: {failures}"

    means = {key: float(np.mean([r[key] for r in results])) for key in results[0]}
    assert means["pdm_score"] > 0.70, means
    assert means["comfort"] > 0.80, means
    assert means["no_at_fault_collisions"] > 0.80, means
    assert means["drivable_area_compliance"] > 0.90, means
    assert means["traffic_light_compliance"] > 0.90, means
