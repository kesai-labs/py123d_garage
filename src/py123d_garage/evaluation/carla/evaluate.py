"""Drives a leaderboard routes file with a garage policy, on the CARLA build in lib/carla."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import hydra
from omegaconf import DictConfig

from py123d_garage.api.abstract_policy import AbstractPolicy, AnyPolicy
from py123d_garage.common.config_help import (
    CONFIG_PATH,
    build_from_string,
    finalize_evaluation,
    hydra_overrides,
    register_schema,
    run_dir,
    save_config,
)
from py123d_garage.common.logging_setup import setup_logging
from py123d_garage.config.schema.evaluation.carla_config import (
    CarlaBenchmarkConfig,
)

LOG = logging.getLogger(__name__)

_CARLA_LIB_DIR = Path(__file__).resolve().parents[4] / "lib" / "carla"
_AGENT_FILE = Path(__file__).resolve().parent / "help" / "leaderboard_agent.py"


register_schema("evaluate_carla", CarlaBenchmarkConfig)


@hydra.main(config_path=str(CONFIG_PATH), config_name="evaluate_carla", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main entrypoint for evaluating a policy on one CARLA routes file."""
    setup_logging()
    config_arguments = hydra_overrides()
    benchmark_config: CarlaBenchmarkConfig = finalize_evaluation(cfg, CarlaBenchmarkConfig, config_arguments)
    save_config(benchmark_config)
    output_dir = run_dir()
    LOG.info(f"Path where all results are stored: {output_dir}")

    policy = cast(
        AnyPolicy,
        build_from_string(benchmark_config.policy_config, AbstractPolicy),
    )
    policy.verify_contract(benchmark_config=benchmark_config)

    benchmark_dir = _CARLA_LIB_DIR / "benchmark" / benchmark_config.benchmark
    scenario_runner_root = benchmark_dir / "scenario_runner"
    simulator_dir = _CARLA_LIB_DIR / "simulator" / "0915"
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join(
                [
                    str(simulator_dir / "PythonAPI" / "carla"),
                    str(benchmark_dir),
                    str(scenario_runner_root),
                    environment.get("PYTHONPATH", ""),
                ],
            ),
            "SCENARIO_RUNNER_ROOT": str(scenario_runner_root),
            "PY123D_GARAGE_CONFIG": " ".join([*config_arguments, f"hydra.run.dir={output_dir}"]),
            "PYTHONUNBUFFERED": "1",
        },
    )

    command = [
        sys.executable,
        str(benchmark_dir / "leaderboard" / "leaderboard_evaluator.py"),
        "--routes",
        str(Path(benchmark_config.routes_file).absolute()),
        "--agent",
        str(_AGENT_FILE),
        "--agent-config",
        "",
        "--checkpoint",
        str(output_dir / "results.json"),
        "--debug-checkpoint",
        str(output_dir / "live_results.txt"),
        "--track",
        "SENSORS",
        "--port",
        str(benchmark_config.port),
        "--traffic-manager-port",
        str(benchmark_config.traffic_manager_port),
        "--traffic-manager-seed",
        str(benchmark_config.traffic_manager_seed),
        "--repetitions",
        str(benchmark_config.repetitions),
        "--timeout",
        str(benchmark_config.timeout),
        # The evaluator parses this with type=bool, where any non-empty string is True.
        "--resume",
        "1" if benchmark_config.resume else "",
        "--debug",
        str(benchmark_config.debug),
    ]
    LOG.info(f"Running: {' '.join(command)}")
    returncode = subprocess.run(command, env=environment, check=False).returncode
    if returncode != 0:
        LOG.error(f"Leaderboard evaluator exited with code {returncode}")
        sys.exit(returncode)
    LOG.info(f"Saved results to {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
