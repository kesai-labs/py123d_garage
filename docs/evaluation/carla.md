<div align="center">

<img src="../assets/logo.png" alt="Py123d Garage" width="630">

</div>

<h1 align="center">CARLA evaluation</h1>

> \[!CAUTION\]
> This is not the official CARLA leaderboard. The policy is conditioned on sparse target points along the route instead of the leaderboard's dense GPS target points. Scores are not comparable to the policies in CARLA research literature.

Policies for CARLA train on the [LEAD data](../data/carla.md).

Install the simulator once into `lib/carla/simulator/`, with the Bench2Drive and CARLA leaderboards cloned at pinned commits into `lib/carla/benchmark/` and patched, then start a headless server:

```bash
lib/carla/tools/setup_carla.sh
lib/carla/simulator/0915/CarlaUE4.sh -world-port=2000 -nosound -RenderOffScreen &
```

Drive one Bench2Drive route:

```bash
python -m py123d_garage.evaluation.carla.evaluate \
    policy_config.evaluation_checkpoint_files=[<checkpoint.pth>] \
    policy_config.evaluation_sensor_rig_file=<run dir>/sensor_rig_<dataset>.yaml \
    routes_file=lib/carla/scenes/bench2drive/1711.xml \
    benchmark=bench2drive
```
