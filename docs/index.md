<div align="center">

<img src="assets/logo.png" alt="Py123d Garage" width="630">

</div>

<h1 align="center">Py123D Garage Documentation</h1>

## Setup

Clone the repository:

```bash
git clone git@github.com:kesai-labs/py123d_garage.git py123d_garage
cd py123d_garage
```

Create an environment:

```bash
conda create -n py123d_garage python=3.10 -y
conda activate py123d_garage
conda install -c conda-forge ffmpeg uv -y
```

Install dependencies:

```bash
uv pip install -e ".[train,carla,alpasim]" --group dev
```

## Get started

- Drive a released [checkpoint](https://huggingface.co/kesai-labs/py123d_garage_pretrained_checkpoints) in [AlpaSim](evaluation/alpasim.md).
- Train from scratch: prepare the [data](data/index.md), then run the [training](training.md).
- Before testing a new idea, check the [research logs](research_logs.md).
- Submit to the [Alpasim E2E Challenge 2026](evaluation/alpasim.md#challenge-submission).

## Further documentation

- [TransFuser](policy/transfuser.md)
- [Data](data/index.md)
- [Training](training.md)
- [Evaluation](evaluation/index.md)
- [Research logs](research_logs.md)
- [Citation](citation.md)
