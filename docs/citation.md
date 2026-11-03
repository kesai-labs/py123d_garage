<div align="center">

<img src="assets/logo.png" alt="Py123d Garage" width="630">

</div>

<h1 align="center">Citation</h1>

If you use this software, please cite it as follows:

```bibtex
@article{Dauner2026ARXIV,
  title={123D: Unifying Multi-Modal Autonomous Driving Data at Scale},
  author={Dauner, Daniel and Charraut, Valentin and Berle, Bastian and Li, Tianyu and Nguyen, Long and Wang, Jiabao and Jing, Changhui and Igl, Maximilian and Caesar, Holger and Ivanovic, Boris and Geiger, Andreas and Chitta, Kashyap},
  journal={arXiv preprint arXiv:2605.08084},
  year={2026}
}
```

## Baselines

The policy baselines build on [LEAD TransFuser v6](https://github.com/kesai-labs/lead) (TransFuser). If you use them, please also cite:

```bibtex
@inproceedings{Nguyen2026CVPR,
  title={LEAD: Minimizing Learner-Expert Asymmetry in End-to-End Driving},
  author={Long Nguyen and Micha Fauth and Bernhard Jaeger and Daniel Dauner and Maximilian Igl and Andreas Geiger and Kashyap Chitta},
  booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```

## Benchmarks

The closed-loop and open-loop evaluations build on [AlpaSim](https://github.com/NVlabs/alpasim), [CARLA](https://carla.org), [NAVSIM](https://github.com/autonomousvision/navsim) v1 and v2, [HUGSIM](https://github.com/hyzhou404/HUGSIM), [Odyssey](https://github.com/SNU-ADR/Odyssey) and [NavSafe-∞](https://github.com/VAIL-UCLA/NavSafe-infty); the AlpaSim nuPlan track renders its scenes with [World Engine](https://arxiv.org/abs/2606.19836), and HUGSIM runs through valeo.ai's fork [HUGSIM_garage](https://github.com/valeoai/HUGSIM_garage). If you report their scores, please also cite:

```bibtex
@software{alpasim2025,
  title={AlpaSim: A Modular, Lightweight, and Data-Driven Research Simulator for Autonomous Driving},
  author={NVIDIA and Yulong Cao and Riccardo de Lutio and Sanja Fidler and Guillermo Garcia Cobo and Zan Gojcic and Maximilian Igl and Boris Ivanovic and Peter Karkus and Janick Martinez Esturo and Marco Pavone and Aaron Smith and Ellie Tanimura and Michal Tyszkiewicz and Michael Watson and Qi Wu and Le Zhang},
  url={https://github.com/NVlabs/alpasim},
  year={2025}
}

@inproceedings{Dosovitskiy17,
  title={{CARLA}: {An} Open Urban Driving Simulator},
  author={Alexey Dosovitskiy and German Ros and Felipe Codevilla and Antonio Lopez and Vladlen Koltun},
  booktitle={Proceedings of the 1st Annual Conference on Robot Learning},
  pages={1--16},
  year={2017}
}

@inproceedings{Dauner2024NEURIPS,
  title={NAVSIM: Data-Driven Non-Reactive Autonomous Vehicle Simulation and Benchmarking},
  author={Daniel Dauner and Marcel Hallgarten and Tianyu Li and Xinshuo Weng and Zhiyu Huang and Zetong Yang and Hongyang Li and Igor Gilitschenski and Boris Ivanovic and Marco Pavone and Andreas Geiger and Kashyap Chitta},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2024}
}

@inproceedings{Cao2025CORL,
  title={Pseudo-Simulation for Autonomous Driving},
  author={Wei Cao and Marcel Hallgarten and Tianyu Li and Daniel Dauner and Xunjiang Gu and Caojun Wang and Yakov Miron and Marco Aiello and Hongyang Li and Igor Gilitschenski and Boris Ivanovic and Marco Pavone and Andreas Geiger and Kashyap Chitta},
  booktitle={Conference on Robot Learning (CoRL)},
  year={2025}
}

@article{zhou2024hugsim,
  title={HUGSIM: A Real-Time, Photo-Realistic and Closed-Loop Simulator for Autonomous Driving},
  author={Zhou, Hongyu and Lin, Longzhong and Wang, Jiabao and Lu, Yichong and Bai, Dongfeng and Liu, Bingbing and Wang, Yue and Geiger, Andreas and Liao, Yiyi},
  journal={arXiv preprint arXiv:2412.01718},
  year={2024}
}

@article{kim2026odyssey,
  title={Odyssey: A Closed-Loop Benchmark for Long-Horizon Real-World Driving with Explicit Navigation Routes},
  author={Kim, Jungho and Shin, Hongjae and Yu, Seunghoon and Yoo, Heecheol and Kim, Myeongjun and Oh, Jiyong and Kwak, Donghyuk and Nam, Seunghyeop and Oh, Haesung and Kim, Hyunju and Cho, Hyungchan and Park, Jaehyun and Seo, Soo Won and Choi, Jun Won},
  journal={arXiv preprint arXiv:2610.06469},
  year={2026}
}

@article{bao2026navsafe,
  title={{NavSafe-$\infty$: Benchmarking Closed-Loop Driving Safety in Photorealistic Environments}},
  author={Bao, Yuxin and Ruan, Hongwei and Wang, Luobin and Zhao, Seth Z. and Leng, Ziyang and Zhang, Zihan and Zeng, Yu and McAllister, Rowan and Christensen, Henrik and Zhou, Bolei},
  journal={arXiv preprint arXiv:2609.26618},
  year={2026}
}

@misc{li2026worldengine,
  title={World Engine: Towards the Era of Post-Training for Autonomous Driving},
  author={Tianyu Li and Li Chen and Caojun Wang and Haochen Liu and Kashyap Chitta and Zhenjie Yang and Yuhang Lu and Naisheng Ye and Yihang Qiu and Yufei Wang and Luoxi Zou and Jiaxin Peng and Jin Pan and Zhaoyu Su and Andrei Bursuc and Shengbo Eben Li and Andreas Geiger and Peng Su and Hongyang Li},
  year={2026},
  eprint={2606.19836},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2606.19836},
}
```

## Datasets

The pretrained checkpoints are trained on [nuPlan](https://www.nuplan.org) and [Physical AI AV](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles). If you use them, please also cite:

```bibtex
@misc{nvidia_physical_ai_av_2025,
  title={Physical AI Autonomous Vehicles Dataset},
  author={{NVIDIA Corporation}},
  publisher={Hugging Face},
  howpublished={\url{https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles}},
  year={2025}
}

@inproceedings{Karnchanachari2024ICRA,
  title={Towards learning-based planning: The nuPlan benchmark for real-world autonomous driving},
  author={Napat Karnchanachari and Dimitris Geromichalos and Kok Seang Tan and Nanxiang Li and Christopher Eriksen and Shakiba Yaghoubi and Noushin Mehdipour and Gianmarco Bernasconi and Whye Kit Fong and Yiluan Guo and Holger Caesar},
  booktitle={International Conference on Robotics and Automation (ICRA)},
  year={2024}
}
```
