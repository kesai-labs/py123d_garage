from __future__ import annotations

import logging
import signal
from concurrent import futures

import grpc
import hydra
from alpasim_grpc.v0 import egodriver_pb2_grpc
from omegaconf import DictConfig

from py123d_garage.common.config_help import CONFIG_PATH, finalize_evaluation, hydra_overrides, register_schema
from py123d_garage.common.logging_setup import setup_logging
from py123d_garage.config.schema.evaluation.alpasim_config import AlpasimBenchmarkConfig
from py123d_garage.evaluation.alpasim.help.driver_service import GarageDriver

LOG = logging.getLogger(__name__)


register_schema("evaluate_alpasim", AlpasimBenchmarkConfig)


@hydra.main(config_path=str(CONFIG_PATH), config_name="evaluate_alpasim", version_base=None)
def main(cfg: DictConfig) -> None:
    setup_logging()
    config = finalize_evaluation(cfg, AlpasimBenchmarkConfig, hydra_overrides())
    driver = GarageDriver(config)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=config.max_workers))
    egodriver_pb2_grpc.add_EgodriverServiceServicer_to_server(driver, server)

    bound_port = server.add_insecure_port(f"{config.host}:{config.port}")
    if bound_port == 0:
        raise RuntimeError(f"failed to bind {config.host}:{config.port}")

    def request_stop(signum: int, frame: object) -> None:
        del frame
        LOG.info(f"received signal {signum}, stopping")
        server.stop(grace=0.0)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    server.start()
    LOG.info(f"garage driver listening on {config.host}:{bound_port}")
    server.wait_for_termination()
    driver.close()


if __name__ == "__main__":
    main()
