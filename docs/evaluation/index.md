<div align="center">

<img src="../assets/logo.png" alt="Py123d Garage" width="630">

</div>

<h1 align="center">Evaluation Overview</h1>

Our target is the [AlpaSim](alpasim.md) simulator and, after it, real-world deployment. The other benchmarks remain for testing and development.

<div align="center">

|         Benchmark         |  Feedback   |                     Data source                      |
| :-----------------------: | :---------: | :--------------------------------------------------: |
|   [AlpaSim](alpasim.md)   | closed loop | AlpaSim on NuRec (Physical-AI-AV) and MTGS (nuPlan)  |
|    [NAVSIM](navsim.md)    |  open loop  | nuPlan, or any 123D dataset with boxes and an HD map |
| [Open loop](open_loop.md) |  open loop  |                   any 123D dataset                   |
|     [CARLA](carla.md)     | closed loop |                   CARLA simulator                    |

</div>

> \[!CAUTION\]
> NAVSIM and CARLA here are not the official protocols. Official NAVSIM conditions on a driving command, and the CARLA leaderboard on dense GPS waypoints that need an HD map. Here both use sparse target points that an SD map can give. Scores are not comparable to their leaderboards.

## Checks before a run

Each entry point first checks that the policy fits the benchmark: that it plans far enough ahead, reads no more history than the benchmark serves, and asks only for sensors and target points that exist there. A mismatch fails before any data loads or the simulator starts.

These checks catch mistakes; they are not leaderboard rules. Widen a limit if your policy needs it. The exception is AlpaSim: it serves the route only from 40 m to 80 m ahead, and that is not ours to change.

## Ensembles

All entry points accept several checkpoints of one training run and average their plans.
