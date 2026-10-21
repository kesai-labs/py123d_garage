<div align="center">

<img src="docs/assets/logo.png" alt="Py123d Garage" width="630">

[![python](https://img.shields.io/badge/python-3.10%20--%203.13-699cdb?logo=python&logoColor=white&labelColor=555555)](pyproject.toml)
[![Alpasim E2E](https://img.shields.io/badge/Alpasim%20E2E-2026-ffbe27?labelColor=555555)](https://huggingface.co/spaces/nvidia/AlpasimE2EClosedLoopChallenge2026)
[![CARLA](https://img.shields.io/badge/CARLA-2.0-de7061?labelColor=555555)](https://leaderboard.carla.org/challenge/)
[![NAVSIM](https://img.shields.io/badge/NAVSIM-v1-405b7d?labelColor=555555)](https://huggingface.co/spaces/AGC2024-P/e2e-driving-navtest)
[![license](https://img.shields.io/badge/license-Apache%202.0-b2b2b2?labelColor=555555)](LICENSE)

**Data. Training. Benchmarks. A starter kit for end-to-end driving research.**

[Setup](docs/index.md) | [Data](docs/data/index.md) | [Training](docs/training.md) | [Evaluation](docs/evaluation/index.md) | [Research logs](docs/research_logs.md) | [Checkpoints](https://huggingface.co/kesai-labs/py123d_garage_pretrained_checkpoints) | [Blog](https://kesai.eu/blog/2026-09-08-common-task-framework/) | [Leaderboard](https://huggingface.co/spaces/nvidia/AlpasimE2EClosedLoopChallenge2026)

</div>

<p align="center">
<img src="https://github.com/user-attachments/assets/fabc48e4-4de1-4072-ad22-4069ff1a9868" width="100%">
</p>

## News

- 🔥 [2026 Oct. 19](https://github.com/kesai-labs/py123d_garage/releases#release-v0.2.0) New [TransFuser](docs/policy/transfuser.md) checkpoints, optimized [training pipeline](docs/training.md) and [research logs](docs/research_logs.md).
- 🔥 [2026 Sept. 08](https://github.com/kesai-labs/py123d_garage/releases#release-v0.1.0) Initial code and checkpoint release.

## Overview

py123d_garage is an open framework for training and evaluating end-to-end driving policies. Built on 123D, it works toward one driving policy trained on every public dataset and evaluated in closed loop on every public benchmark. Training currently covers nuPlan, Physical AI AV and LEAD; evaluation covers AlpaSim, NAVSIM and CARLA Leaderboard 2.0. It is also the reference starting point for the [Alpasim E2E Challenge 2026](https://huggingface.co/spaces/nvidia/AlpasimE2EClosedLoopChallenge2026).

Everything needed to reproduce and build on our results:

- **Strong baseline.** Trained on 1800h of public data, our 68M model rivals frontier open models.
- **Efficient training pipeline.** From scratch in under a day on one 8 x H100 node, at 17-23 % MFU.
- **Open research logs.** The [research logs](docs/research_logs.md) document our experiments, including failed and untried directions.

Under active development: more datasets, benchmarks and baselines will follow. Contributions welcome.

## Results

Closed-loop AlpaSim results of [ltf-base-joint64-68M v0.2.0](https://huggingface.co/kesai-labs/py123d_garage_pretrained_checkpoints/tree/main/v0.2.0/ltf-base-joint64-68M) on both challenge tracks, mean ± std over three training seeds. The unconditional TransFuser drives without navigational intent to stay comparable with Alpamayo, while the conditional TransFuser consumes target points 40, 60 and 80 m ahead. [Alpamayo 1](https://huggingface.co/nvidia/Alpamayo-R1-10B) and [Alpamayo 1.5](https://huggingface.co/nvidia/Alpamayo-1.5-10B) run in their native setting, 10 Hz with four cameras and no route, over three sampled rollouts.

<div align="center">

| Physical AI   |        Score ↑ |   km/incident ↑ | Offroad ↓ | Collision ↓ | Corridor ↓ | Progress ↓ |
| :------------ | -------------: | --------------: | --------: | ----------: | ---------: | ---------: |
| Alpamayo 1    |     56.0 ± 1.0 |     0.96 ± 0.05 |    10.3 % |       5.4 % |     24.3 % |     10.2 % |
| Alpamayo 1.5  |     55.3 ± 0.7 |     0.94 ± 0.06 |    11.3 % |   **5.2 %** |     25.4 % |  **8.0 %** |
| Unconditional |     63.9 ± 1.5 |     1.25 ± 0.10 |     6.9 % |       6.8 % |     15.2 % |     16.4 % |
| Conditional   | **76.0 ± 0.8** | **1.34 ± 0.12** | **6.7 %** |       6.8 % |  **3.1 %** |     17.2 % |

| nuPlan        |        Score ↑ |   km/incident ↑ | Offroad ↓ | Collision ↓ | Corridor ↓ | Progress ↓ |
| :------------ | -------------: | --------------: | --------: | ----------: | ---------: | ---------: |
| Unconditional |     89.5 ± 0.9 |     0.63 ± 0.07 |     3.9 % |       0.3 % |      1.1 % |     11.5 % |
| Conditional   | **91.4 ± 1.0** | **0.75 ± 0.11** | **3.3 %** |   **0.2 %** |  **0.0 %** | **11.4 %** |

<sub>Score: AlpaSim score, 0 to 100. km/incident: km per at-fault incident. Offroad, Collision, Corridor: share of scenes ending in that failure. Progress: share of scenes under 80 % of the logged progress. The TransFuser policy replans at 2 Hz without post-processing and with the NuRec harmonizer on. AlpaSim is pinned to commit [e121e59](https://github.com/NVlabs/alpasim/commit/e121e59141dc391f6c295513d6dadc840a419149) </sub>

</div>

## License and citation

Everything in this repository is released under the Apache 2.0 license unless stated otherwise; the datasets keep their own licenses. If you use this software, please cite:

```bibtex
@article{Dauner2026ARXIV,
  title={123D: Unifying Multi-Modal Autonomous Driving Data at Scale},
  author={Dauner, Daniel and Charraut, Valentin and Berle, Bastian and Li, Tianyu and Nguyen, Long and Wang, Jiabao and Jing, Changhui and Igl, Maximilian and Caesar, Holger and Ivanovic, Boris and Geiger, Andreas and Chitta, Kashyap},
  journal={arXiv preprint arXiv:2605.08084},
  year={2026}
}
```

If you use the TransFuser baseline or its checkpoints, please also cite:

```bibtex
@inproceedings{Nguyen2026CVPR,
  title={LEAD: Minimizing Learner-Expert Asymmetry in End-to-End Driving},
  author={Long Nguyen and Micha Fauth and Bernhard Jaeger and Daniel Dauner and Maximilian Igl and Andreas Geiger and Kashyap Chitta},
  booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```

This repository builds on many other baselines, benchmarks and datasets listed [here](docs/citation.md); if your work uses them, please cite them as well.

## Acknowledgements

The TransFuser baseline and training pipeline descend from [LEAD](https://github.com/kesai-labs/lead). The training recipe draws on ideas from many works, above all [Alpamayo](https://github.com/NVlabs/alpamayo), the [2025 BEHAVIOR Challenge winner](https://arxiv.org/abs/2512.06951), [VaVAM](https://github.com/valeoai/VideoActionModel) and [Qwen-Drive](https://github.com/QwenLM/Qwen-Drive-1.0).
