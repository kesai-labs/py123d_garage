<div align="center">

<img src="docs/assets/logo.png" alt="Py123d Garage" width="630">

[![Odyssey](https://img.shields.io/badge/Odyssey-2026-8a6fb3?labelColor=555555)](https://github.com/SNU-ADR/Odyssey)
[![NavSafe-∞](https://img.shields.io/badge/NavSafe--%E2%88%9E-2026-c0603c?labelColor=555555)](https://github.com/VAIL-UCLA/NavSafe-infty)
[![Alpasim E2E](https://img.shields.io/badge/Alpasim%20E2E-2026-ffbe27?labelColor=555555)](https://huggingface.co/spaces/nvidia/AlpasimE2EClosedLoopChallenge2026)
[![NAVSIM v2](https://img.shields.io/badge/NAVSIM-v2-405b7d?labelColor=555555)](https://huggingface.co/spaces/AGC2025/e2e-driving-navhard)
[![HUGSIM](https://img.shields.io/badge/HUGSIM-2024-4f9a6e?labelColor=555555)](https://github.com/valeoai/HUGSIM_garage)
[![NAVSIM v1](https://img.shields.io/badge/NAVSIM-v1-405b7d?labelColor=555555)](https://huggingface.co/spaces/AGC2024-P/e2e-driving-navtest)
[![CARLA](https://img.shields.io/badge/CARLA-2.0-de7061?labelColor=555555)](https://leaderboard.carla.org/challenge/)

**Data. Training. Benchmarks. A starter kit for end-to-end driving research.**

[Setup](docs/index.md) | [Data](docs/data/index.md) | [Training](docs/training.md) | [Evaluation](docs/evaluation/index.md) | [Research logs](docs/research_logs.md) | [Checkpoints](https://huggingface.co/kesai-labs/py123d_garage_pretrained_checkpoints) | [Blog](https://kesai.eu/blog/2026-11-02-py123d-garage/) | [Leaderboard](https://huggingface.co/spaces/nvidia/AlpasimE2EClosedLoopChallenge2026)

</div>

<p align="center">
<img src="https://github.com/user-attachments/assets/fabc48e4-4de1-4072-ad22-4069ff1a9868" width="100%">
</p>

## News

- 🔥 [2026 Oct. 19](https://github.com/kesai-labs/py123d_garage/releases#release-v0.2.0) New [TransFuser-K1](docs/policy/transfuser.md) checkpoints, optimized [training pipeline](docs/training.md) and [research logs](docs/research_logs.md).
- 🔥 [2026 Sept. 08](https://github.com/kesai-labs/py123d_garage/releases#release-v0.1.0) Initial code and checkpoint release.

## Contents

- [Overview](#overview)
- [Results](#results)
  - Closed loop: [AlpaSim](#alpasim), [Odyssey](#odyssey), [HUGSIM](#hugsim), [NavSafe-∞](#navsafe-)
  - Open loop: [NAVSIM v1](#navsim-v1), [NAVSIM v2](#navsim-v2)
- [License and citation](#license-and-citation)
- [Acknowledgements](#acknowledgements)

## Overview

py123d_garage is an open framework for training and evaluating end-to-end driving policies. Built on 123D, it works toward one driving policy trained on every public dataset and evaluated in closed loop on every public benchmark. Training currently covers nuPlan, Physical AI AV and LEAD; evaluation covers AlpaSim, HUGSIM, Odyssey, NavSafe-∞, NAVSIM v1 and v2, and CARLA Leaderboard 2.0. It is also the reference starting point for the [Alpasim E2E Challenge 2026](https://huggingface.co/spaces/nvidia/AlpasimE2EClosedLoopChallenge2026).

Everything needed to reproduce and build on our results:

- **Strong baseline.** Trained on 1800h of public data, our 68M model rivals frontier open models.
- **Efficient training pipeline.** From scratch in under a day on one 8 x H100 node, at 17-23 % MFU.
- **Open research logs.** The [research logs](docs/research_logs.md) document our experiments, including failed and untried directions.

Under active development: more datasets, benchmarks and baselines will follow. Contributions welcome.

## Results

We report the results of TransFuser-K1 (version [ltf-base-joint32-68M v0.2.0](https://huggingface.co/kesai-labs/py123d_garage_pretrained_checkpoints/tree/main/v0.2.0/ltf-base-joint32-68M/posttrain)) on closed-loop and open-loop benchmarks.

Each benchmark is only an approximation of driving in real world. If a baseline is trained on the quantities that a benchmark measures, a higher score says less about how well the capability would generalize. The tables show two types of such baselines in italics:

- **Metric training.** The baseline is trained to predict the metric of the benchmark, or it uses such predictions to select its plan. This mainly affects NAVSIM, where many baselines learn to predict the PDM score or its sub-scores. We mark such a baseline only on the benchmark whose metric it is trained on.
- **Distillation from reinforcement learning.** The baseline imitates a teacher that is trained with reinforcement learning in closed-loop simulation, with rewards designed for the benchmark. We mark such a baseline only on the benchmark that its teacher is trained for. This mainly affects HUGSIM, where GigaPixel, DriveZero, DriveZero-Scale and TerraTransfer++ are trained in this way.

### Closed Loop Benchmarks

#### Alpasim

Closed-loop AlpaSim results on both challenge tracks, mean ± std over three training seeds. The unconditional TransFuser-K1 drives without navigational intent to stay comparable with Alpamayo, while the conditional TransFuser-K1 consumes target points 40, 60 and 80 m ahead. [Alpamayo 1](https://huggingface.co/nvidia/Alpamayo-R1-10B) and [Alpamayo 1.5](https://huggingface.co/nvidia/Alpamayo-1.5-10B) run in their native setting, 10 Hz with four cameras and no navigational intent input, over three sampled rollouts. On nuPlan, the [SimScale](https://arxiv.org/abs/2511.23369) baselines LTF, GTRS-Dense and DiffusionDrive read the driving command.

<div align="center">

<table>
  <thead>
    <tr><th nowrap>Physical AI</th><th nowrap>Navigation Input</th><th nowrap>Score ↑</th><th nowrap>km/incident ↑</th><th nowrap>Offroad ↓</th><th nowrap>Collision ↓</th><th nowrap>Corridor ↓</th><th nowrap>Progress ↑</th></tr>
  </thead>
  <tbody>
    <tr><td nowrap>Alpamayo 1</td><td nowrap>×</td><td nowrap>56.0 ± 1.0</td><td nowrap>0.96 ± 0.05</td><td nowrap>10.3 %</td><td nowrap>5.4 %</td><td nowrap>24.3 %</td><td nowrap>75.3</td></tr>
    <tr><td nowrap>Alpamayo 1.5</td><td nowrap>×</td><td nowrap>55.3 ± 0.7</td><td nowrap>0.94 ± 0.06</td><td nowrap>11.3 %</td><td nowrap><b>5.2 %</b></td><td nowrap>25.4 %</td><td nowrap>76.3</td></tr>
    <tr><td rowspan="2" nowrap><b>TransFuser-K1 (Ours)</b></td><td nowrap>×</td><td nowrap>64.2 ± 0.4</td><td nowrap>1.28</td><td nowrap>6.3 %</td><td nowrap>7.3 %</td><td nowrap>15.2 %</td><td nowrap>79.3</td></tr>
    <tr><td nowrap>✓</td><td nowrap><b>78.4 ± 0.7</b></td><td nowrap><b>1.44</b></td><td nowrap><b>5.8 %</b></td><td nowrap>7.0 %</td><td nowrap><b>3.3 %</b></td><td nowrap><b>84.6</b></td></tr>
  </tbody>
</table>

<table>
  <thead>
    <tr><th nowrap>nuPlan</th><th nowrap>Navigation Input</th><th nowrap>Score ↑</th><th nowrap>km/incident ↑</th><th nowrap>Offroad ↓</th><th nowrap>Collision ↓</th><th nowrap>Corridor ↓</th><th nowrap>Progress ↑</th></tr>
  </thead>
  <tbody>
    <tr><td nowrap>SimScale DiffusionDrive</td><td nowrap>Command</td><td nowrap>83.4</td><td nowrap>0.25</td><td nowrap>7.7 %</td><td nowrap>0.7 %</td><td nowrap>4.9 %</td><td nowrap>81.2</td></tr>
    <tr><td nowrap>SimScale LTF</td><td nowrap>Command</td><td nowrap>86.3</td><td nowrap>0.32</td><td nowrap>5.7 %</td><td nowrap>1.4 %</td><td nowrap>3.4 %</td><td nowrap>82.6</td></tr>
    <tr><td nowrap>SimScale GTRS-Dense</td><td nowrap>Command</td><td nowrap>88.5</td><td nowrap>0.35</td><td nowrap>3.7 %</td><td nowrap>0.9 %</td><td nowrap>2.5 %</td><td nowrap>82.9</td></tr>
    <tr><td rowspan="2" nowrap><b>TransFuser-K1 (Ours)</b></td><td nowrap>×</td><td nowrap>89.6 ± 0.3</td><td nowrap>0.63</td><td nowrap>3.8 %</td><td nowrap><b>0.4 %</b></td><td nowrap>1.0 %</td><td nowrap>79.3</td></tr>
    <tr><td nowrap>✓</td><td nowrap><b>90.3 ± 0.5</b></td><td nowrap><b>0.72</b></td><td nowrap><b>3.3 %</b></td><td nowrap>0.5 %</td><td nowrap><b>0.8 %</b></td><td nowrap><b>79.6</b></td></tr>
  </tbody>
</table>

<sub>Score: AlpaSim score, 0 to 100. km/incident: km per at-fault incident. Offroad, Collision, Corridor: share of scenes ending in that failure. Progress: mean progress along the logged drive, 0 to 100. Navigation Input: target points 40, 60 and 80 m ahead (✓), the driving command, or none (×). TransFuser-K1 replans at 2 Hz without post-processing and with the NuRec harmonizer on. AlpaSim is pinned to commit [e121e59](https://github.com/NVlabs/alpasim/commit/e121e59141dc391f6c295513d6dadc840a419149) </sub>

</div>

[AlpaSim E2E Challenge](https://huggingface.co/spaces/nvidia/AlpasimE2EClosedLoopChallenge2026) leaderboard on the hidden test sets, as of November 2, 2026: the reference entries of the host and of OpenDriveLab, and our submission (TransFuser-K1, seed 2).

<div align="center">

<table>
  <thead>
    <tr><th nowrap>Track</th><th nowrap>Entry</th><th nowrap>Score ↑</th><th nowrap>km/incident ↑</th><th nowrap>Dist. to GT ↓</th></tr>
  </thead>
  <tbody>
    <tr><td rowspan="3" nowrap>Physical AI</td><td nowrap>Host: go straight with delay</td><td nowrap>17.5</td><td nowrap>0.14</td><td nowrap>2.16</td></tr>
    <tr><td nowrap>Host: Alpamayo 1</td><td nowrap>56.0</td><td nowrap>0.57</td><td nowrap>2.29</td></tr>
    <tr><td nowrap><b>TransFuser-K1 (Ours)</b></td><td nowrap><b>63.2</b></td><td nowrap><b>0.65</b></td><td nowrap><b>1.09</b></td></tr>
    <tr><td rowspan="4" nowrap>nuPlan</td><td nowrap>OpenDriveLab: SimScale LTF</td><td nowrap>74.1</td><td nowrap>0.27</td><td nowrap>2.27</td></tr>
    <tr><td nowrap>OpenDriveLab: SimScale GTRS-Dense</td><td nowrap>76.2</td><td nowrap>0.41</td><td nowrap>2.17</td></tr>
    <tr><td nowrap>OpenDriveLab: SimScale DiffusionDrive</td><td nowrap>77.8</td><td nowrap>0.27</td><td nowrap>2.16</td></tr>
    <tr><td nowrap><b>TransFuser-K1 (Ours)</b></td><td nowrap><b>92.2</b></td><td nowrap><b>0.68</b></td><td nowrap><b>0.63</b></td></tr>
  </tbody>
</table>

<sub>Score: AlpaSim scene score, 0 to 100. km/incident: km per at-fault incident. Dist. to GT: mean distance to the recorded trajectory, in m. Physical AI: 161 entries from 47 teams; nuPlan: 153 entries from 49 teams. </sub>

</div>

#### Odyssey

Closed-loop [Odyssey](https://github.com/SNU-ADR/Odyssey) results on all 100 scenes in non-reactive and reactive traffic, mean over three training seeds. TransFuser-K1 consumes target points 40, 60 and 80 m ahead on Odyssey's SD route and replans at 2 Hz. The baselines are the published results of the Odyssey repository, with and without SD-route guidance.

<div align="center">

<table>
  <thead>
    <tr><th rowspan="2" nowrap>Odyssey</th><th rowspan="2" nowrap>SD route</th><th colspan="6" nowrap>Non-reactive</th><th colspan="6" nowrap>Reactive</th></tr>
    <tr><th nowrap>RouteDS ↑</th><th nowrap>SDC ↑</th><th nowrap>PLCA ↑</th><th nowrap>PLCS ↑</th><th nowrap>Eff.</th><th nowrap>Comf. ↑</th><th nowrap>RouteDS ↑</th><th nowrap>SDC ↑</th><th nowrap>PLCA ↑</th><th nowrap>PLCS ↑</th><th nowrap>Eff.</th><th nowrap>Comf. ↑</th></tr>
  </thead>
  <tbody>
    <tr><td rowspan="2" nowrap>LTF</td><td nowrap>×</td><td nowrap>16.2</td><td nowrap>51</td><td nowrap>66.0</td><td nowrap>50.3</td><td nowrap>63.9</td><td nowrap><b>100</b></td><td nowrap>16.3</td><td nowrap>54</td><td nowrap>63.5</td><td nowrap>49.1</td><td nowrap>76.1</td><td nowrap>99</td></tr>
    <tr><td nowrap>✓</td><td nowrap>20.1</td><td nowrap>95</td><td nowrap>57.1</td><td nowrap>56.1</td><td nowrap>68.2</td><td nowrap>96</td><td nowrap>23.0</td><td nowrap>93</td><td nowrap>53.6</td><td nowrap>52.6</td><td nowrap>78.0</td><td nowrap>93</td></tr>
    <tr><td rowspan="2" nowrap>ReCogDrive (VLA)</td><td nowrap>×</td><td nowrap>15.4</td><td nowrap>71</td><td nowrap>75.7</td><td nowrap>59.0</td><td nowrap>76.9</td><td nowrap><b>100</b></td><td nowrap>20.3</td><td nowrap>62</td><td nowrap>77.1</td><td nowrap>58.7</td><td nowrap>93.7</td><td nowrap>99</td></tr>
    <tr><td nowrap>✓</td><td nowrap>21.5</td><td nowrap>83</td><td nowrap>74.4</td><td nowrap>55.8</td><td nowrap>59.3</td><td nowrap>98</td><td nowrap>26.3</td><td nowrap>84</td><td nowrap>75.6</td><td nowrap>56.7</td><td nowrap>75.5</td><td nowrap>98</td></tr>
    <tr><td rowspan="2" nowrap>DiffusionDrive</td><td nowrap>×</td><td nowrap>25.6</td><td nowrap>59</td><td nowrap>78.8</td><td nowrap>63.7</td><td nowrap>61.4</td><td nowrap><b>100</b></td><td nowrap>24.2</td><td nowrap>59</td><td nowrap>77.0</td><td nowrap>63.1</td><td nowrap>75.7</td><td nowrap><b>100</b></td></tr>
    <tr><td nowrap>✓</td><td nowrap>35.9</td><td nowrap><b>100</b></td><td nowrap>82.0</td><td nowrap><b>80.5</b></td><td nowrap>63.4</td><td nowrap>99</td><td nowrap>41.0</td><td nowrap><b>100</b></td><td nowrap>81.2</td><td nowrap><b>80.2</b></td><td nowrap>71.5</td><td nowrap>99</td></tr>
    <tr><td rowspan="2" nowrap>DrivoR</td><td nowrap>×</td><td nowrap>20.6</td><td nowrap>50</td><td nowrap>75.5</td><td nowrap>62.8</td><td nowrap>104.7</td><td nowrap>44</td><td nowrap>25.5</td><td nowrap>53</td><td nowrap>76.6</td><td nowrap>62.8</td><td nowrap>133.4</td><td nowrap>47</td></tr>
    <tr><td nowrap>✓</td><td nowrap>36.5</td><td nowrap>90</td><td nowrap>72.8</td><td nowrap>66.0</td><td nowrap>84.5</td><td nowrap>60</td><td nowrap>41.4</td><td nowrap>89</td><td nowrap>71.8</td><td nowrap>66.0</td><td nowrap>107.4</td><td nowrap>64</td></tr>
    <tr><td nowrap><b>TransFuser-K1 (Ours)</b></td><td nowrap>✓</td><td nowrap>43.3 ± 3.5</td><td nowrap>87</td><td nowrap><b>83.7</b></td><td nowrap>67.5</td><td nowrap>78.0</td><td nowrap>67</td><td nowrap>43.3 ± 3.6</td><td nowrap>86</td><td nowrap><b>84.3</b></td><td nowrap>67.8</td><td nowrap>87.4</td><td nowrap>71</td></tr>
    <tr><td rowspan="2" nowrap>SafeDrive</td><td nowrap>×</td><td nowrap>27.5</td><td nowrap>41</td><td nowrap>79.3</td><td nowrap>63.7</td><td nowrap>69.4</td><td nowrap>47</td><td nowrap>30.4</td><td nowrap>45</td><td nowrap>78.8</td><td nowrap>62.8</td><td nowrap>86.3</td><td nowrap>40</td></tr>
    <tr><td nowrap>✓</td><td nowrap><b>44.4</b></td><td nowrap>82</td><td nowrap>75.0</td><td nowrap>69.8</td><td nowrap>68.3</td><td nowrap>44</td><td nowrap><b>48.9</b></td><td nowrap>79</td><td nowrap>73.8</td><td nowrap>69.5</td><td nowrap>81.3</td><td nowrap>49</td></tr>
  </tbody>
</table>

<sub>RouteDS: route driving score, 0 to 100. SDC: SD-route compliance. PLCA, PLCS: pre-lane-change accuracy and score. Eff.: median efficiency, 100 = the speed of the surrounding traffic. Comf.: share of episodes within every comfort limit. SD route: with (✓) or without (×) SD-route guidance. ±: standard deviation over the three seeds, for our RouteDS only. Odyssey is pinned to commit [8c4a902](https://github.com/SNU-ADR/Odyssey/commit/8c4a902ebfe242feb624d00e756c6c1ccc95555e). </sub>

</div>

#### HUGSIM

Closed-loop [HUGSIM](https://github.com/hyzhou404/HUGSIM) results on its 345 pre-challenge scenarios of nuScenes, PandaSet, KITTI-360 and Waymo, by difficulty, mean over three training seeds. TransFuser-K1 drives without navigational intent and replans at 2 Hz. VAD, LTF, UniAD and DrivoR are the results of the [DrivoR](https://arxiv.org/abs/2601.05083) paper, rerun with corrected HUGSIM code; DrivoR-Scale, GigaPixel, DriveZero and DriveZero-Scale are taken from the [DriveZero](https://arxiv.org/abs/2609.06055) paper, GTRS-Dense and ZTRS from version 2 of the [ZTRS](https://arxiv.org/abs/2510.24108) paper, and TerraTransfer++, which reports only the average HD-Score, from the [TerraTransfer++](https://arxiv.org/abs/2610.30319) paper.

<div align="center">

<table>
  <thead>
    <tr><th rowspan="2" nowrap>HUGSIM</th><th rowspan="2" nowrap>Navigation Input</th><th rowspan="2" nowrap>RL teacher</th><th colspan="5" nowrap>HD-Score ↑</th><th colspan="5" nowrap>RC ↑</th></tr>
    <tr><th nowrap>Avg.</th><th nowrap>E</th><th nowrap>M</th><th nowrap>H</th><th nowrap>X</th><th nowrap>Avg.</th><th nowrap>E</th><th nowrap>M</th><th nowrap>H</th><th nowrap>X</th></tr>
  </thead>
  <tbody>
    <tr><td nowrap>VAD</td><td nowrap>Command</td><td nowrap>×</td><td nowrap>13.4</td><td nowrap>36.3</td><td nowrap>9.5</td><td nowrap>8.0</td><td nowrap>11.5</td><td nowrap>31.4</td><td nowrap>51.3</td><td nowrap>31.1</td><td nowrap>25.3</td><td nowrap>26.5</td></tr>
    <tr><td nowrap>LTF</td><td nowrap>×</td><td nowrap>×</td><td nowrap>23.7</td><td nowrap>58.9</td><td nowrap>18.0</td><td nowrap>9.8</td><td nowrap>25.9</td><td nowrap>38.9</td><td nowrap>67.8</td><td nowrap>35.1</td><td nowrap>26.2</td><td nowrap>40.5</td></tr>
    <tr><td nowrap>GTRS-Dense</td><td nowrap>Command</td><td nowrap>×</td><td nowrap>28.2</td><td nowrap>57.8</td><td nowrap>35.2</td><td nowrap>16.2</td><td nowrap>5.8</td><td nowrap>34.5</td><td nowrap>62.0</td><td nowrap>40.5</td><td nowrap>23.9</td><td nowrap>13.7</td></tr>
    <tr><td nowrap>UniAD</td><td nowrap>Command</td><td nowrap>×</td><td nowrap>32.7</td><td nowrap>64.9</td><td nowrap>45.8</td><td nowrap>20.6</td><td nowrap>6.6</td><td nowrap>45.9</td><td nowrap>78.4</td><td nowrap>60.5</td><td nowrap>33.6</td><td nowrap>17.8</td></tr>
    <tr><td nowrap>ZTRS</td><td nowrap>Command</td><td nowrap>×</td><td nowrap>32.9</td><td nowrap>66.9</td><td nowrap>37.9</td><td nowrap>21.1</td><td nowrap>9.8</td><td nowrap>42.0</td><td nowrap>74.8</td><td nowrap>46.1</td><td nowrap>31.6</td><td nowrap>20.0</td></tr>
    <tr><td nowrap>DrivoR</td><td nowrap>×</td><td nowrap>×</td><td nowrap>35.7</td><td nowrap>73.3</td><td nowrap>34.6</td><td nowrap>18.8</td><td nowrap><b>32.5</b></td><td nowrap>49.8</td><td nowrap>80.9</td><td nowrap>50.5</td><td nowrap>33.8</td><td nowrap><b>47.1</b></td></tr>
    <tr><td nowrap>DrivoR-Scale</td><td nowrap>×</td><td nowrap>×</td><td nowrap>38.1</td><td nowrap>64.8</td><td nowrap>50.3</td><td nowrap>16.8</td><td nowrap>25.8</td><td nowrap>46.4</td><td nowrap>70.4</td><td nowrap>54.4</td><td nowrap>28.2</td><td nowrap>39.2</td></tr>
    <tr><td nowrap><i>GigaPixel</i></td><td nowrap><i>Command</i></td><td nowrap><i>✓</i></td><td nowrap><i>38.5</i></td><td nowrap><i>67.4</i></td><td nowrap><i>51.9</i></td><td nowrap><i>19.1</i></td><td nowrap><i>21.6</i></td><td nowrap><i>50.1</i></td><td nowrap><i>78.6</i></td><td nowrap><i>60.7</i></td><td nowrap><i>32.7</i></td><td nowrap><i>35.4</i></td></tr>
    <tr><td nowrap><b>TransFuser-K1 (Ours)</b></td><td nowrap>×</td><td nowrap>×</td><td nowrap>38.5 ± 1.3</td><td nowrap>73.1</td><td nowrap>54.6</td><td nowrap>23.3</td><td nowrap>9.7</td><td nowrap>46.7</td><td nowrap>79.8</td><td nowrap>61.8</td><td nowrap>31.4</td><td nowrap>20.6</td></tr>
    <tr><td nowrap><i>DriveZero</i></td><td nowrap><i>Command</i></td><td nowrap><i>✓</i></td><td nowrap><i>39.4</i></td><td nowrap><i>78.5</i></td><td nowrap><i>47.2</i></td><td nowrap><i>16.0</i></td><td nowrap><i>28.7</i></td><td nowrap><i>53.9</i></td><td nowrap><i>85.5</i></td><td nowrap><i>63.7</i></td><td nowrap><i>31.7</i></td><td nowrap><i>43.3</i></td></tr>
    <tr><td nowrap><i>DriveZero-Scale</i></td><td nowrap><i>Command</i></td><td nowrap><i>✓</i></td><td nowrap><i>46.6</i></td><td nowrap><i><b>81.9</b></i></td><td nowrap><i><b>60.5</b></i></td><td nowrap><i><b>24.7</b></i></td><td nowrap><i>27.6</i></td><td nowrap><i><b>54.5</b></i></td><td nowrap><i><b>87.5</b></i></td><td nowrap><i><b>64.9</b></i></td><td nowrap><i><b>34.6</b></i></td><td nowrap><i>39.8</i></td></tr>
    <tr><td nowrap><i>TerraTransfer++</i></td><td nowrap><i>Goal</i></td><td nowrap><i>✓</i></td><td nowrap><i><b>51.8</b></i></td><td nowrap><i>–</i></td><td nowrap><i>–</i></td><td nowrap><i>–</i></td><td nowrap><i>–</i></td><td nowrap><i>–</i></td><td nowrap><i>–</i></td><td nowrap><i>–</i></td><td nowrap><i>–</i></td><td nowrap><i>–</i></td></tr>
  </tbody>
</table>

<sub>HD-Score: HUGSIM's score of the planned trajectories times route completion, 0 to 100. ±: standard deviation over the three seeds, for our average HD-Score only. RC: route completion, 0 to 100. E, M, H, X: easy, medium, hard and extreme scenarios. Navigation Input: HUGSIM's driving command, a goal point, or none (×). RL teacher: imitates a teacher trained with reinforcement learning whose rewards target HUGSIM (✓, italic rows). TerraTransfer++ scores a braking stand-in plan instead of its own plan and trains on frames rendered from the evaluation scenes. HUGSIM runs through [HUGSIM_garage](https://github.com/valeoai/HUGSIM_garage), pinned to commit [c18814c](https://github.com/valeoai/HUGSIM_garage/commit/c18814c0ef2b1cfdeabe33506a5d345cfbb83cbc), with the protocol of the DrivoR paper. </sub>

</div>

#### NavSafe-∞

Closed-loop [NavSafe-∞](https://github.com/VAIL-UCLA/NavSafe-infty) results on its 280 safety scenarios of reconstructed nuPlan logs, 66 of them with inserted actors, mean over three training seeds. TransFuser-K1 drives without navigational intent and replans at 2 Hz, while the baselines read NavSafe's driving command. The baselines are the results of the [NavSafe-∞ leaderboard](https://navsafe-vail.github.io/#leaderboard), mean over three evaluation seeds.

<div align="center">

<table>
  <thead>
    <tr><th nowrap>NavSafe-∞</th><th nowrap>Navigation Input</th><th nowrap>DS ↑</th><th nowrap>SR ↑</th><th nowrap>DE</th><th nowrap>Comfort ↑</th><th nowrap>TC ↑</th><th nowrap>VRUC ↑</th><th nowrap>TV ↑</th><th nowrap>TI ↑</th></tr>
  </thead>
  <tbody>
    <tr><td nowrap>DiffusionDrive</td><td nowrap>Command</td><td nowrap>44.8 ± 0.8</td><td nowrap>21.0 ± 0.6</td><td nowrap>165.1 ± 1.8</td><td nowrap>97.7 ± 0.6</td><td nowrap>21.3 ± 1.2</td><td nowrap>11.7 ± 1.4</td><td nowrap>23.6 ± 0.0</td><td nowrap>22.2 ± 1.9</td></tr>
    <tr><td nowrap>DiffusionDrive (BeyondDrive)</td><td nowrap>Command</td><td nowrap>46.6 ± 0.1</td><td nowrap>21.8 ± 0.9</td><td nowrap>172.8 ± 1.3</td><td nowrap>97.2 ± 0.2</td><td nowrap>22.7 ± 1.5</td><td nowrap>10.0 ± 0.0</td><td nowrap>25.2 ± 1.4</td><td nowrap>22.2 ± 1.9</td></tr>
    <tr><td nowrap>RAP</td><td nowrap>Command</td><td nowrap>47.7 ± 0.7</td><td nowrap>27.3 ± 0.4</td><td nowrap>220.8 ± 3.8</td><td nowrap>48.4 ± 0.6</td><td nowrap>32.3 ± 1.2</td><td nowrap>14.2 ± 6.3</td><td nowrap>28.8 ± 0.5</td><td nowrap>22.2 ± 3.9</td></tr>
    <tr><td nowrap>MTDrive-SFT</td><td nowrap>Command</td><td nowrap>48.8 ± 0.9</td><td nowrap>31.8 ± 1.4</td><td nowrap>173.8 ± 0.8</td><td nowrap>94.5 ± 1.3</td><td nowrap>30.3 ± 1.2</td><td nowrap>15.0 ± 6.6</td><td nowrap>35.8 ± 1.4</td><td nowrap>44.4 ± 1.9</td></tr>
    <tr><td nowrap>ReCogDrive-2B-IL</td><td nowrap>Command</td><td nowrap>49.0 ± 1.0</td><td nowrap>28.2 ± 1.4</td><td nowrap>166.9 ± 3.2</td><td nowrap>97.6 ± 0.1</td><td nowrap>30.0 ± 1.0</td><td nowrap>16.7 ± 3.8</td><td nowrap>31.2 ± 0.5</td><td nowrap>26.7 ± 3.3</td></tr>
    <tr><td nowrap>AutoVLA</td><td nowrap>Command</td><td nowrap>51.1 ± 0.8</td><td nowrap>29.5 ± 2.4</td><td nowrap>186.0 ± 0.6</td><td nowrap>65.4 ± 0.5</td><td nowrap>33.0 ± 2.7</td><td nowrap>15.0 ± 4.3</td><td nowrap>29.7 ± 2.9</td><td nowrap>36.7 ± 8.8</td></tr>
    <tr><td nowrap>DiffusionDrive (SimScale)</td><td nowrap>Command</td><td nowrap>52.2 ± 2.1</td><td nowrap>35.2 ± 2.4</td><td nowrap>189.7 ± 4.3</td><td nowrap>96.9 ± 0.3</td><td nowrap>38.3 ± 1.2</td><td nowrap>18.3 ± 7.2</td><td nowrap>36.7 ± 5.0</td><td nowrap>42.2 ± 3.9</td></tr>
    <tr><td nowrap>LTF (SimScale)</td><td nowrap>Command</td><td nowrap>52.7 ± 0.9</td><td nowrap>34.1 ± 0.2</td><td nowrap>192.1 ± 0.9</td><td nowrap>97.4 ± 0.5</td><td nowrap>35.0 ± 1.7</td><td nowrap>17.5 ± 2.5</td><td nowrap>38.5 ± 0.5</td><td nowrap>36.7 ± 0.0</td></tr>
    <tr><td nowrap>LTF</td><td nowrap>Command</td><td nowrap>53.0 ± 1.9</td><td nowrap>27.7 ± 1.8</td><td nowrap>163.7 ± 0.1</td><td nowrap><b>97.8 ± 0.2</b></td><td nowrap>31.7 ± 0.6</td><td nowrap>25.0 ± 7.5</td><td nowrap>28.5 ± 1.9</td><td nowrap>15.6 ± 1.9</td></tr>
    <tr><td nowrap>ReCogDrive-2B-RL</td><td nowrap>Command</td><td nowrap>53.6 ± 1.4</td><td nowrap>31.0 ± 2.1</td><td nowrap>174.0 ± 1.9</td><td nowrap>85.1 ± 0.5</td><td nowrap>35.3 ± 2.3</td><td nowrap>18.3 ± 6.3</td><td nowrap>32.4 ± 1.4</td><td nowrap>27.8 ± 5.1</td></tr>
    <tr><td nowrap>MTDrive-mtGRPO</td><td nowrap>Command</td><td nowrap>55.9 ± 0.5</td><td nowrap>40.6 ± 0.9</td><td nowrap>277.6 ± 3.2</td><td nowrap>93.9 ± 0.4</td><td nowrap>40.3 ± 2.9</td><td nowrap>16.7 ± 1.4</td><td nowrap>46.4 ± 0.9</td><td nowrap><b>52.2 ± 3.9</b></td></tr>
    <tr><td nowrap>SimWAM</td><td nowrap>Command</td><td nowrap>57.5 ± 2.7</td><td nowrap>37.5 ± 1.6</td><td nowrap>160.5 ± 1.6</td><td nowrap>97.2 ± 0.3</td><td nowrap>41.3 ± 0.6</td><td nowrap>26.7 ± 7.2</td><td nowrap>40.6 ± 0.5</td><td nowrap>27.8 ± 5.1</td></tr>
    <tr><td nowrap>DrivoR (SimScale)</td><td nowrap>Command</td><td nowrap>57.7 ± 1.3</td><td nowrap>42.5 ± 0.4</td><td nowrap>176.7 ± 0.9</td><td nowrap>85.9 ± 0.4</td><td nowrap>44.0 ± 1.0</td><td nowrap>17.5 ± 2.5</td><td nowrap>48.2 ± 2.4</td><td nowrap>50.0 ± 0.0</td></tr>
    <tr><td nowrap>GTRS-Dense-V2-99</td><td nowrap>Command</td><td nowrap>64.3 ± 0.8</td><td nowrap>49.5 ± 1.2</td><td nowrap>163.9 ± 0.9</td><td nowrap>95.9 ± 0.6</td><td nowrap>52.0 ± 1.7</td><td nowrap>40.0 ± 2.5</td><td nowrap>54.2 ± 0.5</td><td nowrap>36.7 ± 0.0</td></tr>
    <tr><td nowrap>DrivoR</td><td nowrap>Command</td><td nowrap>64.7 ± 1.6</td><td nowrap>50.7 ± 0.7</td><td nowrap>188.0 ± 1.1</td><td nowrap>87.9 ± 1.3</td><td nowrap>58.7 ± 2.5</td><td nowrap>31.7 ± 5.8</td><td nowrap>52.4 ± 2.6</td><td nowrap>43.3 ± 0.0</td></tr>
    <tr><td nowrap>SparseDriveV2</td><td nowrap>Command</td><td nowrap>65.3 ± 1.7</td><td nowrap>48.8 ± 3.0</td><td nowrap>168.0 ± 1.0</td><td nowrap>93.4 ± 0.4</td><td nowrap>54.0 ± 0.0</td><td nowrap>43.3 ± 8.0</td><td nowrap>48.2 ± 4.2</td><td nowrap>41.1 ± 3.9</td></tr>
    <tr><td nowrap><b>TransFuser-K1 (Ours)</b></td><td nowrap>×</td><td nowrap>72.1 ± 1.1</td><td nowrap>54.0 ± 2.4</td><td nowrap>185.5 ± 1.8</td><td nowrap>93.3 ± 0.3</td><td nowrap>59.6 ± 6.8</td><td nowrap>42.5 ± 5.0</td><td nowrap>57.9 ± 1.4</td><td nowrap>35.6 ± 1.9</td></tr>
    <tr><td nowrap>GTRS-Dense-V2-99 (SimScale)</td><td nowrap>Command</td><td nowrap><b>73.5 ± 0.8</b></td><td nowrap><b>61.8 ± 1.6</b></td><td nowrap>141.5 ± 1.3</td><td nowrap>87.7 ± 1.7</td><td nowrap><b>69.3 ± 0.6</b></td><td nowrap><b>50.0 ± 2.5</b></td><td nowrap><b>64.2 ± 1.9</b></td><td nowrap>43.3 ± 3.3</td></tr>
  </tbody>
</table>

<sub>DS: driving score, 0 to 100. SR: success rate. DE: driving efficiency, the ego's speed relative to the surrounding traffic, in percent. Comfort: compliance with the acceleration, jerk, yaw-rate and yaw-acceleration bounds. TC, VRUC, TV, TI: success rate on traffic crashes, crashes with vulnerable road users, traffic violations and traffic incidents, the mean over their event types. ±: sample standard deviation over the three seeds. Navigation Input: NavSafe's driving command, or none (×). NavSafe-∞ is pinned to commit [bd68458](https://github.com/VAIL-UCLA/NavSafe-infty/commit/bd68458df1e2cc8e0f0dfcf62c06b71a86ae229c), with its benchmark protocol; the renderer draws the cameras only at the policy's 2 Hz. </sub>

</div>

### Open Loop Benchmarks

#### NAVSIM v1

[NAVSIM](https://github.com/autonomousvision/navsim) v1.1 results on navtest, scored with the official devkit, mean over three training seeds. TransFuser-K1 drives without navigational intent, while the baselines read NAVSIM's driving command. The baselines are the published results of the [NAVSIM](https://arxiv.org/abs/2406.15349), [DrivoR](https://arxiv.org/abs/2601.05083), [DriveZero](https://arxiv.org/abs/2609.06055), [TerraTransfer++](https://arxiv.org/abs/2610.30319), [4D-WAM](https://arxiv.org/abs/2608.10107), [ForeDrive](https://arxiv.org/abs/2609.26299), [SimWAM](https://arxiv.org/abs/2608.07468), [SUV](https://arxiv.org/abs/2608.03084) and [ReDrive](https://arxiv.org/abs/2609.33854) papers; LTFv6 is the [LEAD](https://github.com/kesai-labs/lead)-LTFv6 leaderboard entry (one seed).

<div align="center">

<table>
  <thead>
    <tr><th nowrap>navtest</th><th nowrap>Navigation Input</th><th nowrap>Metric training</th><th nowrap>PDMS ↑</th><th nowrap>NC ↑</th><th nowrap>DAC ↑</th><th nowrap>TTC ↑</th><th nowrap>Comf. ↑</th><th nowrap>EP ↑</th></tr>
  </thead>
  <tbody>
    <tr><td nowrap>UniAD</td><td nowrap>Command</td><td nowrap>×</td><td nowrap>83.4</td><td nowrap>97.8</td><td nowrap>91.9</td><td nowrap>92.9</td><td nowrap><b>100</b></td><td nowrap>78.8</td></tr>
    <tr><td nowrap>PARA-Drive</td><td nowrap>Command</td><td nowrap>×</td><td nowrap>84.0</td><td nowrap>97.9</td><td nowrap>92.4</td><td nowrap>93.0</td><td nowrap>99.8</td><td nowrap>79.3</td></tr>
    <tr><td nowrap>LTFv6</td><td nowrap>Command</td><td nowrap>×</td><td nowrap>86.4</td><td nowrap>97.5</td><td nowrap>95.4</td><td nowrap>93.8</td><td nowrap><b>100</b></td><td nowrap>80.9</td></tr>
    <tr><td nowrap><b>TransFuser-K1 (Ours)</b></td><td nowrap>×</td><td nowrap>×</td><td nowrap>87.9 ± 0.1</td><td nowrap>98.3</td><td nowrap>96.6</td><td nowrap>95.2</td><td nowrap><b>100</b></td><td nowrap>81.2</td></tr>
    <tr><td nowrap>DiffusionDrive</td><td nowrap>Command</td><td nowrap>×</td><td nowrap>88.1</td><td nowrap>98.2</td><td nowrap>96.2</td><td nowrap>94.7</td><td nowrap><b>100</b></td><td nowrap>82.2</td></tr>
    <tr><td nowrap>ForeDrive</td><td nowrap>Command</td><td nowrap>×</td><td nowrap>89.9</td><td nowrap>98.6</td><td nowrap>97.6</td><td nowrap>95.6</td><td nowrap><b>100</b></td><td nowrap>83.8</td></tr>
    <tr><td nowrap>SimWAM</td><td nowrap>Command</td><td nowrap>×</td><td nowrap>90.3</td><td nowrap>98.6</td><td nowrap>98.0</td><td nowrap>95.8</td><td nowrap>–</td><td nowrap>84.0</td></tr>
    <tr><td nowrap>SUV</td><td nowrap>Command</td><td nowrap>×</td><td nowrap>90.8</td><td nowrap>99.1</td><td nowrap>97.8</td><td nowrap>96.7</td><td nowrap><b>100</b></td><td nowrap>84.6</td></tr>
    <tr><td nowrap>4D-WAM</td><td nowrap>Command</td><td nowrap>×</td><td nowrap>90.9</td><td nowrap>99.1</td><td nowrap>97.9</td><td nowrap>96.6</td><td nowrap><b>100</b></td><td nowrap>84.6</td></tr>
    <tr><td nowrap>ReDrive</td><td nowrap>Command</td><td nowrap>×</td><td nowrap>91.0</td><td nowrap>99.1</td><td nowrap>97.9</td><td nowrap>97.2</td><td nowrap><b>100</b></td><td nowrap>84.3</td></tr>
    <tr><td nowrap><i>DrivoR</i></td><td nowrap><i>Command</i></td><td nowrap><i>✓</i></td><td nowrap><i>93.1</i></td><td nowrap><i>98.9</i></td><td nowrap><i>98.3</i></td><td nowrap><i>96.2</i></td><td nowrap><i><b>100</b></i></td><td nowrap><i>89.1</i></td></tr>
    <tr><td nowrap><i>DrivoR-Scale</i></td><td nowrap><i>Command</i></td><td nowrap><i>✓</i></td><td nowrap><i>94.6</i></td><td nowrap><i>99.1</i></td><td nowrap><i>99.2</i></td><td nowrap><i>96.9</i></td><td nowrap><i><b>100</b></i></td><td nowrap><i>91.6</i></td></tr>
    <tr><td nowrap><i>DriveZero-Scale</i></td><td nowrap><i>Command</i></td><td nowrap><i>✓</i></td><td nowrap><i>95.3</i></td><td nowrap><i><b>99.2</b></i></td><td nowrap><i>99.4</i></td><td nowrap><i><b>97.3</b></i></td><td nowrap><i>99.9</i></td><td nowrap><i>92.5</i></td></tr>
    <tr><td nowrap><i>TerraTransfer++</i></td><td nowrap><i>Goal</i></td><td nowrap><i>✓</i></td><td nowrap><i><b>95.5</b></i></td><td nowrap><i>98.9</i></td><td nowrap><i><b>99.5</b></i></td><td nowrap><i>92.2</i></td><td nowrap><i>99.9</i></td><td nowrap><i><b>96.9</b></i></td></tr>
  </tbody>
</table>

<sub>NC: no at-fault collision. DAC: drivable area compliance. TTC: time to collision. Comf.: comfort. EP: ego progress. PDMS: PDM score, 0 to 100. ±: standard deviation over the three seeds, for our PDMS only. Navigation Input: NAVSIM's driving command, a goal point, or none (×). Metric training: trained on the PDM score or its sub-scores (✓, italic rows); TerraTransfer++ selects its training targets by collision, TTC and progress. SimWAM: before its reinforcement learning stage; comfort not reported. SimWAM and SUV: 5B-parameter video models. </sub>

</div>

#### NAVSIM v2

[NAVSIM](https://github.com/autonomousvision/navsim) v2.2 results on navhard two-stage, scored with the official devkit, mean over three training seeds. TransFuser-K1 drives without navigational intent, while the baselines read NAVSIM's driving command. The baselines are the navhard leaderboard snapshot of 03/2026 in the [pseudo-simulation](https://arxiv.org/abs/2506.04218) paper; GigaPixel, DriveZero-Scale and TerraTransfer++ are the results published in the [TerraTransfer++](https://arxiv.org/abs/2610.30319) paper. DiffusionDrive and the SimScale models (from [SimScale](https://arxiv.org/abs/2511.23369)), [GTRS-Dense](https://arxiv.org/abs/2506.06664), [ReDrive](https://arxiv.org/abs/2609.33854), DiffusionDrive-ViRA (from [ViRA](https://arxiv.org/abs/2610.09695)), [4D-WAM](https://arxiv.org/abs/2608.10107), [SUV](https://arxiv.org/abs/2608.03084) and [SimWAM](https://arxiv.org/abs/2608.07468) are taken from their papers; these results predate a devkit fix that raises the other entries by 2 to 3 points. LTFv6 is the [LEAD](https://github.com/kesai-labs/lead)-LTFv6 leaderboard entry (one seed).

<div align="center">

<table>
  <thead>
    <tr><th rowspan="2" nowrap>navhard</th><th rowspan="2" nowrap>Navigation Input</th><th rowspan="2" nowrap>Metric training</th><th rowspan="2" nowrap>EPDMS ↑</th><th colspan="9" nowrap>Stage 1</th><th colspan="9" nowrap>Stage 2</th></tr>
    <tr><th nowrap>NC</th><th nowrap>DAC</th><th nowrap>DDC</th><th nowrap>TLC</th><th nowrap>EP</th><th nowrap>TTC</th><th nowrap>LK</th><th nowrap>HC</th><th nowrap>EC</th><th nowrap>NC</th><th nowrap>DAC</th><th nowrap>DDC</th><th nowrap>TLC</th><th nowrap>EP</th><th nowrap>TTC</th><th nowrap>LK</th><th nowrap>HC</th><th nowrap>EC</th></tr>
  </thead>
  <tbody>
    <tr><td nowrap>CV</td><td nowrap>×</td><td nowrap>×</td><td nowrap>11.4</td><td nowrap>88.8</td><td nowrap>42.8</td><td nowrap>70.6</td><td nowrap>99.3</td><td nowrap>77.5</td><td nowrap>87.3</td><td nowrap>78.6</td><td nowrap>97.1</td><td nowrap>60.4</td><td nowrap>83.2</td><td nowrap>59.1</td><td nowrap>76.5</td><td nowrap>98.0</td><td nowrap>71.3</td><td nowrap>81.1</td><td nowrap>47.9</td><td nowrap>97.1</td><td nowrap>61.9</td></tr>
    <tr><td nowrap>MLP</td><td nowrap>Command</td><td nowrap>×</td><td nowrap>14.1</td><td nowrap>93.2</td><td nowrap>55.7</td><td nowrap>86.6</td><td nowrap>99.3</td><td nowrap>81.2</td><td nowrap>92.2</td><td nowrap>83.5</td><td nowrap>97.5</td><td nowrap>77.7</td><td nowrap>77.2</td><td nowrap>51.9</td><td nowrap>74.4</td><td nowrap>98.2</td><td nowrap>77.1</td><td nowrap>75.0</td><td nowrap>40.8</td><td nowrap>97.8</td><td nowrap><b>79.8</b></td></tr>
    <tr><td nowrap>LTF</td><td nowrap>Command</td><td nowrap>×</td><td nowrap>25.1</td><td nowrap>96.2</td><td nowrap>79.5</td><td nowrap>99.1</td><td nowrap>99.5</td><td nowrap>84.1</td><td nowrap>95.1</td><td nowrap>94.2</td><td nowrap>97.5</td><td nowrap>79.1</td><td nowrap>77.7</td><td nowrap>70.2</td><td nowrap>84.2</td><td nowrap>98.0</td><td nowrap>85.1</td><td nowrap>75.6</td><td nowrap>45.4</td><td nowrap>95.7</td><td nowrap>75.9</td></tr>
    <tr><td nowrap>DiffusionDrive</td><td nowrap>Command</td><td nowrap>×</td><td nowrap>27.5</td><td nowrap>96.8</td><td nowrap>86.0</td><td nowrap>98.8</td><td nowrap>99.3</td><td nowrap>84.0</td><td nowrap>95.8</td><td nowrap>96.7</td><td nowrap>97.6</td><td nowrap>79.6</td><td nowrap>80.1</td><td nowrap>72.8</td><td nowrap>84.4</td><td nowrap>98.4</td><td nowrap>85.9</td><td nowrap>76.6</td><td nowrap>46.4</td><td nowrap>96.3</td><td nowrap>72.8</td></tr>
    <tr><td nowrap><i>SimScale LTF</i></td><td nowrap><i>Command</i></td><td nowrap><i>✓</i></td><td nowrap><i>30.2</i></td><td nowrap><i>96.1</i></td><td nowrap><i>85.3</i></td><td nowrap><i>99.4</i></td><td nowrap><i>99.3</i></td><td nowrap><i>84.7</i></td><td nowrap><i>94.7</i></td><td nowrap><i>93.6</i></td><td nowrap><i>97.6</i></td><td nowrap><i>77.3</i></td><td nowrap><i>85.5</i></td><td nowrap><i>66.9</i></td><td nowrap><i>91.6</i></td><td nowrap><i>99.1</i></td><td nowrap><i>93.0</i></td><td nowrap><i>81.1</i></td><td nowrap><i>58.3</i></td><td nowrap><i>95.1</i></td><td nowrap><i>42.9</i></td></tr>
    <tr><td nowrap>LTFv6</td><td nowrap>Command</td><td nowrap>×</td><td nowrap>31.9</td><td nowrap>96.5</td><td nowrap>86.6</td><td nowrap>99.2</td><td nowrap>99.5</td><td nowrap>84.4</td><td nowrap>95.1</td><td nowrap>94.4</td><td nowrap>97.7</td><td nowrap>76.4</td><td nowrap>79.8</td><td nowrap>75.5</td><td nowrap>86.2</td><td nowrap>97.8</td><td nowrap>89.5</td><td nowrap>76.0</td><td nowrap>50.0</td><td nowrap>95.2</td><td nowrap>66.7</td></tr>
    <tr><td nowrap><i>SimScale DiffusionDrive</i></td><td nowrap><i>Command</i></td><td nowrap><i>✓</i></td><td nowrap><i>32.6</i></td><td nowrap><i>97.2</i></td><td nowrap><i>88.7</i></td><td nowrap><i>99.3</i></td><td nowrap><i>99.3</i></td><td nowrap><i>82.8</i></td><td nowrap><i>96.9</i></td><td nowrap><i><b>98.0</b></i></td><td nowrap><i>97.3</i></td><td nowrap><i>59.6</i></td><td nowrap><i>82.4</i></td><td nowrap><i>72.1</i></td><td nowrap><i>92.9</i></td><td nowrap><i>98.5</i></td><td nowrap><i>92.1</i></td><td nowrap><i>80.6</i></td><td nowrap><i>60.8</i></td><td nowrap><i>95.4</i></td><td nowrap><i>31.9</i></td></tr>
    <tr><td nowrap>ReDrive</td><td nowrap>Command</td><td nowrap>×</td><td nowrap>34.4</td><td nowrap>97.9</td><td nowrap>93.6</td><td nowrap>99.4</td><td nowrap>99.6</td><td nowrap>84.2</td><td nowrap>96.9</td><td nowrap>97.8</td><td nowrap><b>97.8</b></td><td nowrap>73.3</td><td nowrap>82.2</td><td nowrap>75.8</td><td nowrap>84.3</td><td nowrap>98.2</td><td nowrap>87.3</td><td nowrap>77.9</td><td nowrap>47.8</td><td nowrap>95.9</td><td nowrap>54.2</td></tr>
    <tr><td nowrap><b>TransFuser-K1 (Ours)</b></td><td nowrap>×</td><td nowrap>×</td><td nowrap>34.9 ± 0.4</td><td nowrap>97.2</td><td nowrap>90.7</td><td nowrap>98.3</td><td nowrap>99.6</td><td nowrap>83.1</td><td nowrap>96.3</td><td nowrap>96.6</td><td nowrap>97.6</td><td nowrap>72.7</td><td nowrap>82.6</td><td nowrap>77.7</td><td nowrap>87.4</td><td nowrap>98.4</td><td nowrap>86.9</td><td nowrap>78.8</td><td nowrap>49.3</td><td nowrap>96.5</td><td nowrap>58.7</td></tr>
    <tr><td nowrap>DiffusionDrive-ViRA</td><td nowrap>Command</td><td nowrap>×</td><td nowrap>35.8</td><td nowrap>96.4</td><td nowrap>87.8</td><td nowrap>99.2</td><td nowrap>99.3</td><td nowrap>98.5</td><td nowrap>95.3</td><td nowrap>96.4</td><td nowrap>97.6</td><td nowrap>78.7</td><td nowrap>80.5</td><td nowrap>74.1</td><td nowrap>86.4</td><td nowrap>98.4</td><td nowrap>97.5</td><td nowrap>77.2</td><td nowrap>47.6</td><td nowrap>95.8</td><td nowrap>71.3</td></tr>
    <tr><td nowrap>4D-WAM</td><td nowrap>Command</td><td nowrap>×</td><td nowrap>35.9</td><td nowrap>92.8</td><td nowrap>89.6</td><td nowrap>99.7</td><td nowrap>99.3</td><td nowrap><b>98.6</b></td><td nowrap>91.6</td><td nowrap>86.9</td><td nowrap>97.1</td><td nowrap><b>81.3</b></td><td nowrap>82.6</td><td nowrap>71.8</td><td nowrap>86.9</td><td nowrap>98.2</td><td nowrap><b>97.6</b></td><td nowrap>78.8</td><td nowrap>49.0</td><td nowrap>95.8</td><td nowrap>71.6</td></tr>
    <tr><td nowrap>SUV</td><td nowrap>Command</td><td nowrap>×</td><td nowrap>36.9</td><td nowrap>96.9</td><td nowrap>94.2</td><td nowrap>99.3</td><td nowrap>99.6</td><td nowrap>84.1</td><td nowrap>95.6</td><td nowrap>96.7</td><td nowrap><b>97.8</b></td><td nowrap>79.6</td><td nowrap>82.7</td><td nowrap>74.2</td><td nowrap>85.9</td><td nowrap>98.4</td><td nowrap>86.0</td><td nowrap>78.9</td><td nowrap>47.2</td><td nowrap>95.9</td><td nowrap>69.1</td></tr>
    <tr><td nowrap>SimWAM</td><td nowrap>Command</td><td nowrap>×</td><td nowrap>37.6</td><td nowrap>98.0</td><td nowrap>92.0</td><td nowrap>99.7</td><td nowrap>99.6</td><td nowrap>83.8</td><td nowrap>96.2</td><td nowrap>97.3</td><td nowrap><b>97.8</b></td><td nowrap>71.6</td><td nowrap>81.8</td><td nowrap>78.6</td><td nowrap>87.3</td><td nowrap>98.4</td><td nowrap>86.3</td><td nowrap>78.6</td><td nowrap>49.5</td><td nowrap>96.3</td><td nowrap>69.9</td></tr>
    <tr><td nowrap><i>RAP</i></td><td nowrap><i>Command</i></td><td nowrap><i>✓</i></td><td nowrap><i>39.6</i></td><td nowrap><i>97.1</i></td><td nowrap><i>94.4</i></td><td nowrap><i>98.7</i></td><td nowrap><i>99.7</i></td><td nowrap><i>83.8</i></td><td nowrap><i>96.8</i></td><td nowrap><i>94.6</i></td><td nowrap><i>96.4</i></td><td nowrap><i>66.2</i></td><td nowrap><i>83.2</i></td><td nowrap><i>83.8</i></td><td nowrap><i>87.3</i></td><td nowrap><i>98.0</i></td><td nowrap><i>86.8</i></td><td nowrap><i>80.3</i></td><td nowrap><i>52.2</i></td><td nowrap><i>95.1</i></td><td nowrap><i>52.4</i></td></tr>
    <tr><td nowrap><i>GTRS-Dense</i></td><td nowrap><i>Command</i></td><td nowrap><i>✓</i></td><td nowrap><i>45.3</i></td><td nowrap><i>98.9</i></td><td nowrap><i>98.2</i></td><td nowrap><i>99.8</i></td><td nowrap><i>99.6</i></td><td nowrap><i>73.9</i></td><td nowrap><i>98.9</i></td><td nowrap><i>95.3</i></td><td nowrap><i>97.3</i></td><td nowrap><i>40.0</i></td><td nowrap><i>91.5</i></td><td nowrap><i>90.8</i></td><td nowrap><i>94.7</i></td><td nowrap><i>98.5</i></td><td nowrap><i>70.8</i></td><td nowrap><i>90.1</i></td><td nowrap><i>55.4</i></td><td nowrap><i>97.2</i></td><td nowrap><i>54.2</i></td></tr>
    <tr><td nowrap><i>SimScale GTRS-Dense</i></td><td nowrap><i>Command</i></td><td nowrap><i>✓</i></td><td nowrap><i>48.0</i></td><td nowrap><i>99.3</i></td><td nowrap><i>97.6</i></td><td nowrap><i>99.4</i></td><td nowrap><i><b>100.0</b></i></td><td nowrap><i>71.5</i></td><td nowrap><i><b>99.6</b></i></td><td nowrap><i>96.0</i></td><td nowrap><i>95.8</i></td><td nowrap><i>30.7</i></td><td nowrap><i><b>94.9</b></i></td><td nowrap><i>94.3</i></td><td nowrap><i>94.5</i></td><td nowrap><i><b>99.3</b></i></td><td nowrap><i>79.0</i></td><td nowrap><i>92.6</i></td><td nowrap><i>57.8</i></td><td nowrap><i>93.4</i></td><td nowrap><i>30.9</i></td></tr>
    <tr><td nowrap><i>ZTRS</i></td><td nowrap><i>Command</i></td><td nowrap><i>✓</i></td><td nowrap><i>48.1</i></td><td nowrap><i>98.8</i></td><td nowrap><i>97.5</i></td><td nowrap><i><b>100.0</b></i></td><td nowrap><i><b>100.0</b></i></td><td nowrap><i>66.6</i></td><td nowrap><i>98.8</i></td><td nowrap><i>96.2</i></td><td nowrap><i>96.6</i></td><td nowrap><i>44.0</i></td><td nowrap><i>91.1</i></td><td nowrap><i>90.4</i></td><td nowrap><i>95.7</i></td><td nowrap><i>99.0</i></td><td nowrap><i>63.5</i></td><td nowrap><i>89.7</i></td><td nowrap><i>60.4</i></td><td nowrap><i>97.6</i></td><td nowrap><i>66.0</i></td></tr>
    <tr><td nowrap><i>GigaPixel</i></td><td nowrap><i>Command</i></td><td nowrap><i>✓</i></td><td nowrap><i>50.1</i></td><td nowrap><i>99.4</i></td><td nowrap><i>95.8</i></td><td nowrap><i>99.4</i></td><td nowrap><i>99.8</i></td><td nowrap><i>68.1</i></td><td nowrap><i><b>99.6</b></i></td><td nowrap><i>91.8</i></td><td nowrap><i>97.6</i></td><td nowrap><i>49.8</i></td><td nowrap><i>93.7</i></td><td nowrap><i>92.9</i></td><td nowrap><i>96.0</i></td><td nowrap><i>98.9</i></td><td nowrap><i>62.6</i></td><td nowrap><i>90.7</i></td><td nowrap><i>60.2</i></td><td nowrap><i>98.2</i></td><td nowrap><i>58.8</i></td></tr>
    <tr><td nowrap><i>SimScale</i></td><td nowrap><i>Command</i></td><td nowrap><i>✓</i></td><td nowrap><i>53.2</i></td><td nowrap><i><b>99.5</b></i></td><td nowrap><i>99.1</i></td><td nowrap><i>99.8</i></td><td nowrap><i><b>100.0</b></i></td><td nowrap><i>69.6</i></td><td nowrap><i>99.5</i></td><td nowrap><i>95.7</i></td><td nowrap><i>95.5</i></td><td nowrap><i>28.4</i></td><td nowrap><i>94.5</i></td><td nowrap><i>94.2</i></td><td nowrap><i>95.7</i></td><td nowrap><i>99.2</i></td><td nowrap><i>75.8</i></td><td nowrap><i><b>92.8</b></i></td><td nowrap><i>60.0</i></td><td nowrap><i>96.0</i></td><td nowrap><i>43.2</i></td></tr>
    <tr><td nowrap><i>DrivoR</i></td><td nowrap><i>Command</i></td><td nowrap><i>✓</i></td><td nowrap><i>54.5</i></td><td nowrap><i>99.1</i></td><td nowrap><i>98.2</i></td><td nowrap><i>99.3</i></td><td nowrap><i>99.7</i></td><td nowrap><i>75.4</i></td><td nowrap><i>98.6</i></td><td nowrap><i>94.8</i></td><td nowrap><i>97.5</i></td><td nowrap><i>70.2</i></td><td nowrap><i>92.3</i></td><td nowrap><i>91.6</i></td><td nowrap><i>97.3</i></td><td nowrap><i>99.0</i></td><td nowrap><i>75.7</i></td><td nowrap><i>90.5</i></td><td nowrap><i>56.0</i></td><td nowrap><i><b>98.4</b></i></td><td nowrap><i>44.7</i></td></tr>
    <tr><td nowrap><i>DriveZero-Scale</i></td><td nowrap><i>Command</i></td><td nowrap><i>✓</i></td><td nowrap><i>57.1</i></td><td nowrap><i>98.7</i></td><td nowrap><i>99.3</i></td><td nowrap><i><b>100.0</b></i></td><td nowrap><i>99.3</i></td><td nowrap><i>76.5</i></td><td nowrap><i>98.4</i></td><td nowrap><i>95.3</i></td><td nowrap><i>97.1</i></td><td nowrap><i>44.4</i></td><td nowrap><i>94.6</i></td><td nowrap><i><b>95.6</b></i></td><td nowrap><i><b>97.7</b></i></td><td nowrap><i>98.4</i></td><td nowrap><i>76.2</i></td><td nowrap><i>92.7</i></td><td nowrap><i>59.9</i></td><td nowrap><i>96.1</i></td><td nowrap><i>44.0</i></td></tr>
    <tr><td nowrap><i>TerraTransfer++</i></td><td nowrap><i>Goal</i></td><td nowrap><i>✓</i></td><td nowrap><i><b>59.4</b></i></td><td nowrap><i>98.9</i></td><td nowrap><i><b>100.0</b></i></td><td nowrap><i><b>100.0</b></i></td><td nowrap><i>99.6</i></td><td nowrap><i>84.9</i></td><td nowrap><i>98.4</i></td><td nowrap><i>97.8</i></td><td nowrap><i><b>97.8</b></i></td><td nowrap><i>70.2</i></td><td nowrap><i>94.0</i></td><td nowrap><i>90.2</i></td><td nowrap><i>93.6</i></td><td nowrap><i>98.0</i></td><td nowrap><i>85.2</i></td><td nowrap><i>87.2</i></td><td nowrap><i><b>62.5</b></i></td><td nowrap><i>95.4</i></td><td nowrap><i>53.8</i></td></tr>
  </tbody>
</table>

<sub>CV: constant velocity. MLP: ego status MLP. ±: standard deviation over the three seeds, for our EPDMS only. EPDMS: extended PDM score, 0 to 100, over the real frames (stage 1) and the synthetic frames that follow them (stage 2). NC: no at-fault collision. DAC: drivable area compliance. DDC: driving direction compliance. TLC: traffic light compliance. EP: ego progress. TTC: time to collision. LK: lane keeping. HC: history comfort. EC: extended comfort. Navigation Input: NAVSIM's driving command, a goal point, or none (×). Metric training: trained on the PDM score or its sub-scores (✓, italic rows); TerraTransfer++ selects its training targets by collision, TTC and progress. GigaPixel: the variant whose scoring head is trained on the PDM score. GTRS-Dense: the ViT-L model. SimScale models: trained on simulated scenes whose PDM-Closed expert trajectories are selected with the EPDMS sub-scores, which we count as metric training. DiffusionDrive-ViRA: DiffusionDrive with ViRA representation alignment. SimWAM: before its reinforcement learning stage. </sub>

</div>

## License and citation

The code in this repository is released under the Apache 2.0 license unless stated otherwise. The [pretrained checkpoints](https://huggingface.co/kesai-labs/py123d_garage_pretrained_checkpoints) are released under CC BY-NC-SA 4.0, because their training data allow only non-commercial use; the datasets keep their own licenses. If you use this software, please cite:

```bibtex
@article{Dauner2026ARXIV,
  title={123D: Unifying Multi-Modal Autonomous Driving Data at Scale},
  author={Dauner, Daniel and Charraut, Valentin and Berle, Bastian and Li, Tianyu and Nguyen, Long and Wang, Jiabao and Jing, Changhui and Igl, Maximilian and Caesar, Holger and Ivanovic, Boris and Geiger, Andreas and Chitta, Kashyap},
  journal={arXiv preprint arXiv:2605.08084},
  year={2026}
}
```

If you use TransFuser-K1 or its checkpoints, please also cite:

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

TransFuser-K1 and the training pipeline descend from [LEAD](https://github.com/kesai-labs/lead). The training recipe draws on ideas from many works, above all [Alpamayo](https://github.com/NVlabs/alpamayo), the [2025 BEHAVIOR Challenge winner](https://arxiv.org/abs/2512.06951), [VaVAM](https://github.com/valeoai/VideoActionModel) and [Qwen-Drive](https://github.com/QwenLM/Qwen-Drive-1.0).
