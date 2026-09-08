<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../assets/logo_dark.png">
  <img src="../assets/logo_light.png" alt="Py123d Garage" width="575">
</picture>

</div>

<h1 align="center">Evaluation Overview</h1>

> \[!CAUTION\]
> NAVSIM and CARLA here are not the official protocols. They exist in this repository for testing and development purpose; our main target is the Alpasim simulator and real-world deployment. Official NAVSIM conditions on a driving command (left, straight, right), the CARLA leaderboard conditions on much denser GPS target points that require HD-Map. In Py123D Garage, both are replaced by very sparse target points that are possible to be obtained with SD-Map. Therefore, the scores obtained with Py123D Garage are not comparable to the leaderboards.

See documentation of each benchmark for further details.

<div align="center">

|         Benchmark         |  Feedback   |                            Evaluation data source                            |
| :-----------------------: | :---------: | :--------------------------------------------------------------------------: |
| [Open loop](open_loop.md) |  open loop  |                               any 123D dataset                               |
|    [NAVSIM](navsim.md)    |  open loop  | any dataset that has 3D bounding boxes + HD map (we tested only with nuPlan) |
|   [Alpasim](alpasim.md)   | closed loop |                     Alpasim Simulator on nuRec and MTGS                      |
|     [CARLA](carla.md)     | closed loop |                               CARLA Simulator                                |

</div>

## How to check if a policy can be evaluated on an offline dataset or a simulator

An open-loop benchmark is a scoring protocol plus one or more offline datasets. The benchmark declares in `py123d_garage.api.abstract_benchmark_config.AbstractBenchmarkConfig` what its protocol scores and serves, the trajectory horizon and the history it allows; each dataset declares in `py123d_garage.api.abstract_offline_data_source_config.AbstractOfflineDataSourceConfig` what its logs record, the sensor intervals and the target point range. The policy declares what it consumes in `py123d_garage.api.abstract_policy_config.AbstractPolicyConfig`. A policy, in order to be evaluated on an open-loop benchmark, need to pass both benchmark's requirements as well as the dataset's requirements.

A closed-loop benchmark produces the data itself and only needs to declare everything in its `AbstractBenchmarkConfig`.

Every benchmark entrypoint in `py123d_garage.evaluation.<benchmark>.evaluate` calls `py123d_garage.api.abstract_policy.AbstractPolicy.verify_contract` before loading data or starting the simulator. It fails if the policy can not be evaluated on the dataset or the simulator.

For examples

| Benchmark or dataset declares                 | Policy must                                                                                                   |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `required_trajectory_horizon_us`              | plan at least that far (`trajectory_horizon_us`)                                                              |
| `max_history_duration_us`                     | read at most that much past (`required_history_duration_us`)                                                  |
| `min/max_served_target_point_distance_m`      | keep every `required_target_point_distances_m` inside                                                         |
| `served_{ego_state,camera,lidar}_interval_us` | ask for past frames at a whole multiple (`required_past_*_interval_us`); a `None` modality cannot be required |

## Concrete examples

`None` has different meaning differently per column. Under horizon, history and target points range None means no limit. For the ego / camera / lidar past intervals, None means the modality is not available (for example, LiDAR is not available on Alpasim).

### Supported benchmarks

| Benchmark | Min planning horizon | Target points range | Max history | Ego / camera / lidar past interval |
| --------- | -------------------- | ------------------- | ----------- | ---------------------------------- |
| Open loop | None                 | by dataset          | None        | by dataset                         |
| NAVSIM    | 4.0 s                | by dataset          | 1.5 s       | by dataset                         |
| Alpasim   | 2.5 s                | 20 to 80 m          | None        | 100 / 100 / None ms                |
| CARLA     | 2.0 s                | None                | None        | 50 / 50 / 50 ms                    |

Some details on the numbers of Min planning horizon:

- NAVSIM's min planning horizon comes from the original paper, where we observed that policy with too short planning horizon are prone to short-cut learning. If needed, this min planning horizon of this benchmark can be reduced.
- Alpasim's and CARLA's numbers come from the controllers that track the trajectory, not from the scoring. If really needed, rewrite the controller and a shorter horizon should work.

Similarly, the Max history of NAVSIM is an artifact of the original paper. This can be extended if needed.

### Supported datasets

| Dataset        | Target points range | Ego / camera / lidar past interval |
| -------------- | ------------------- | ---------------------------------- |
| nuPlan         | None                | 100 / 100 / 100 ms                 |
| Physical AI AV | None                | 100 / 100 / 100 ms                 |
| LEAD (CARLA)   | None                | 50 / 250 / 50 ms                   |

## Some notes on restrictions

Most of the detail on this page exists to catch mistakes and coding bugs early, not to restrict participants the way a leaderboard does.

The code imposes as few restrictions as it can beyond what Alpasim needs. If the target points range is too small for your policy, for example, widen it.
