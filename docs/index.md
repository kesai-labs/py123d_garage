<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/logo_dark.png">
  <img src="assets/logo_light.png" alt="Py123d Garage" width="575">
</picture>

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

To obtain pretrained checkpoints, see [release notes](https://github.com/kesai-labs/py123d_garage/releases).

## Further documentation

- [Data](data/index.md)
- [Training Cache System](config.md)
- [Training](training.md)
- [Evaluation](evaluation/index.md)
- [Configuration system](config.md)
- [Citation](citation.md)
