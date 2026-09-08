<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo_dark.png">
  <img src="docs/assets/logo_light.png" alt="Py123d Garage" width="575">
</picture>

[![python](https://img.shields.io/badge/python-3.10%20--%203.13-699cdb?logo=python&logoColor=white&labelColor=555555)](pyproject.toml)
[![Alpasim E2E](https://img.shields.io/badge/Alpasim%20E2E-2026-ffbe27?labelColor=555555)](https://huggingface.co/spaces/nvidia/AlpasimE2EClosedLoopChallenge2026)
[![CARLA](https://img.shields.io/badge/CARLA-2.0-de7061?labelColor=555555)](https://leaderboard.carla.org/challenge/)
[![NAVSIM](https://img.shields.io/badge/NAVSIM-v1-405b7d?labelColor=555555)](https://huggingface.co/spaces/AGC2024-P/e2e-driving-navtest)
[![license](https://img.shields.io/badge/license-Apache%202.0-b2b2b2?labelColor=555555)](LICENSE)

**Data. Training. Benchmarking. A starter kit for end-to-end driving research.**

[Setup](docs/index.md) | [Data](docs/data/index.md) | [Training](docs/training.md) | [Evaluation](docs/evaluation/index.md)

</div>

<p align="center">
<img src="https://github.com/user-attachments/assets/fabc48e4-4de1-4072-ad22-4069ff1a9868" width="100%">
</p>

Py123D Garage is an open-source framework to train end-to-end driving policies and evaluate them in closed loop. Built on 123D, it has three parts: one interface to large real-world driving datasets, a fast training loop, and closed-loop simulators that recreate the challenges of real AV deployment. It is the reference starting point for the [Alpasim E2E Challenge 2026](https://huggingface.co/spaces/nvidia/AlpasimE2EClosedLoopChallenge2026), with pretrained baselines to build on.

## License and citation

All assets and code in this repository are under the Apache 2.0 license unless specified otherwise. The datasets inherit their own distribution licenses. If you use this software, please cite it as follows:

```bibtex
@misc{py123d_garage,
  title        = {py123d_garage: end-to-end driving policies across datasets},
  author       = {KE:SAI},
  year         = {2026},
  howpublished = {\url{https://github.com/kesai-labs/py123d_garage}}
}

@article{Dauner2026ARXIV,
  title={123D: Unifying Multi-Modal Autonomous Driving Data at Scale},
  author={Dauner, Daniel and Charraut, Valentin and Berle, Bastian and Li, Tianyu and Nguyen, Long and Wang, Jiabao and Jing, Changhui and Igl, Maximilian and Caesar, Holger and Ivanovic, Boris and Geiger, Andreas and Chitta, Kashyap},
  journal={arXiv preprint arXiv:2605.08084},
  year={2026}
}
```

See [citation](docs/citation.md) for the baselines, benchmarks and datasets to cite alongside.
