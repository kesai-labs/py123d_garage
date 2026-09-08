<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../assets/logo_dark.png">
  <img src="../assets/logo_light.png" alt="Py123d Garage" width="575">
</picture>

</div>

<h1 align="center">CARLA evaluation</h1>

> \[!CAUTION\]
> This is not the official CARLA leaderboard. The policy is conditioned on sparse target points along the route instead of the leaderboard's dense GPS waypoints. Scores of baseline policies are not comparable to the policies in CARLA research literature.

### Setup

Install the simulator once into `lib/carla/simulator/`:

```bash
lib/carla/tools/setup_carla.sh
```

Start a headless server and then drive one Bench2Drive route:

```bash
lib/carla/simulator/0915/CarlaUE4.sh -world-port=2000 -nosound -RenderOffScreen &
scripts/evaluation/carla/transfuser.sh
```

The script reads `CHECKPOINT_FILE`, `SENSOR_RIG_FILE`, `ROUTE_FILE`, `CARLA_PORT` and `CARLA_TM_PORT`.
