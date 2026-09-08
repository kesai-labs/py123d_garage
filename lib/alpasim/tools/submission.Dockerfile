FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TORCH_HOME=/tmp/torch

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3.12 python3.12-venv python3-pip git libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}" UV_LINK_MODE=copy
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv

COPY pyproject.toml /opt/py123d-garage/pyproject.toml
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python -r /opt/py123d-garage/pyproject.toml --extra alpasim

COPY . /opt/py123d-garage
RUN uv pip install --python /opt/venv/bin/python --no-deps /opt/py123d-garage

COPY --from=wheels . /tmp/wheels
RUN uv pip install --python /opt/venv/bin/python --find-links /tmp/wheels alpasim-grpc && rm -rf /tmp/wheels

ARG CHECKPOINT_FILE
ARG SENSOR_RIG_FILE
COPY --from=run ${CHECKPOINT_FILE} /opt/checkpoint/model.pth
COPY --from=run config.yaml /opt/checkpoint/config.yaml
COPY --from=rig ${SENSOR_RIG_FILE} /opt/checkpoint/sensor_rig.yaml

ARG GIT_HASH=unknown
ENV PY123D_GARAGE_CONFIG="\
hydra.run.dir=/tmp/alpasim_driver \
policy_config.evaluation_checkpoint_file=/opt/checkpoint/model.pth \
policy_config.evaluation_sensor_rig_file=/opt/checkpoint/sensor_rig.yaml \
git_hash=${GIT_HASH}"

ENV HOME=/tmp MPLCONFIGDIR=/tmp/matplotlib OMP_NUM_THREADS=8
ENTRYPOINT ["sh", "-c", "exec python -m py123d_garage.evaluation.alpasim.evaluate ${PY123D_GARAGE_CONFIG} host=\"${ALPASIM_DRIVER_HOST:-0.0.0.0}\" port=\"${ALPASIM_DRIVER_PORT:-6789}\" \"$@\"", "sh"]
