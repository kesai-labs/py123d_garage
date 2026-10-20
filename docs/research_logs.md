<div align="center">

<img src="assets/logo.png" alt="Py123d Garage" width="630">

</div>

<h1 align="center">Research logs</h1>

This page lists what we tried while we built the TransFuser recipe and what we did not try yet.

## How to use this page

Most entries are one or two trainings in our setup (data mix, schedule, local AlpaSim scenes), so a dropped idea does not mean a dead end.

- Where a row gives a number, it is the difference in local score points against the row's baseline, on the AlpaSim nuPlan track (**nuPlan**) or the Physical-AI-AV track on the 441 `nurec_curated_val` scenes (**NuRec**), both 0 to 100. Absolute scores are left out: the scoring setup changed during the project. Most results are three-seed means.
- Each row compares against the baseline of its time. Compare within a row, not across rows.
- Run-to-run noise is about 0.5 on nuPlan and 3 on NuRec.
- A change counts as a win only if it helps on both tracks. Single-track gains did not transfer to the leaderboard.
- **Adopted**: in the release recipe. **Dropped**: removed after the test. **Neutral**: no clear effect. **Open**: no clean answer yet.

## Release v0.2.0

### Current weaknesses of TransFuser

- The policy still leans on the target points more than the scene warrants. Adding 1 m of target point noise in training can degrade driving rather than regularize it.
- Speed control, braking in particular, is weaker than Alpamayo's.

### Camera input

| Change                                                                  | Result                                                                                                                                  | Verdict |
| :---------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------- | :------ |
| Front camera only, instead of three cameras                             | Better on nuPlan, worse on NuRec, with 25 % shorter epochs. With a 1 s past frame for both, the front camera alone leads on both tracks | Adopted |
| A past front-camera frame from 1 s ago                                  | nuPlan +1.8, NuRec +7.3: at-fault collisions on NuRec fall from 61 to 25 per seed                                                       | Adopted |
| Two past frames, from 0.5 s and 1 s ago                                 | Midtraining: a tie on nuPlan, slightly worse on NuRec, more scenes where the car never starts. Posttraining: a tie                      | Dropped |
| Tele crop: a magnified crop of the front camera, not a real tele camera | Alone: NuRec −6.7, the car drives faster and collides more. Together with the past frame: a tie, at 29 % more epoch time                | Dropped |
| Photometric augmentation                                                | No clean test. The only run had a shorter pretraining                                                                                   | Open    |
| Other car hoods in the rendered front camera                            | An offline swap of the hood moves the plans by 0.5 m at most. Not a cause of failures                                                   | Neutral |

### Speed input

| Change                                                              | Result                                                                                                                                                                                        | Verdict |
| :------------------------------------------------------------------ | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------ |
| A token with the past ego motion, instead of only the current speed | NuRec −7.4 with 0.1 s and −17.3 with 0.5 s of history, against the speed alone. Lateral failures grow with the history length                                                                 | Dropped |
| Noise on the speed input in training, 10 to 30 % of the speed       | NuRec −28 to −21, nuPlan unchanged                                                                                                                                                            | Dropped |
| Speed dropout, 25 to 75 % of samples                                | First with the regression planner, then with the flow planner. Front collisions on NuRec stay at about 22 per seed at every rate. More scenes where the car never starts: nuPlan −2.0 at 75 % | Dropped |
| No speed input at all                                               | NuRec −23.3 (one seed). After the takeover, the plans slow down: the camera alone cannot hold the speed                                                                                       | Dropped |

### Encoder

| Change                                                                      | Result                                                                              | Verdict |
| :-------------------------------------------------------------------------- | :---------------------------------------------------------------------------------- | :------ |
| ConvNeXt-T with DINOv3 weights, instead of ResNet34                         | Slightly worse on nuPlan                                                            | Dropped |
| ConvNeXt-T with ImageNet-12k weights, instead of ResNet34                   | Perception losses 10 to 24 % lower in pretraining, but closed loop is a tie         | Neutral |
| RegNetY-3.2GF for the image and BEV branches, as in the original TransFuser | A tie on NuRec, more at-fault collisions, 53 % slower                               | Dropped |
| Eight learned register tokens in the fusion                                 | A tie on both tracks, almost no extra cost                                          | Adopted |
| Nearest instead of bilinear up-sampling in the fusion                       | A tie on nuPlan, but twice the at-fault collisions. Posttraining 24 % faster        | Dropped |
| Learned BEV tokens in place of the BEV ResNet                               | Higher pretraining losses on box velocity and heatmaps. Never tested in closed loop | Open    |
| Streaming backbone (see [TransFuser](policy/transfuser.md))                 | Midtraining: a tie on both tracks. Slightly behind after posttraining               | Option  |

### Motion planner

| Change                                                                                                                                                | Result                                                                                                                                                                     | Verdict          |
| :---------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------------- |
| Flow matching instead of one-step L1 regression                                                                                                       | A tie on nuPlan, NuRec +5.4: regression drives more assertively and collides more                                                                                          | Adopted          |
| Flow loss: L1 on the clean trajectory instead of MSE on the velocity (noise) target, with uniform noise levels and time conditioning by adaptive norm | One run with all three changes: NuRec +4.9, nuPlan a tie                                                                                                                   | Adopted          |
| Sampling from zero noise, instead of random or fixed noise                                                                                            | A fixed noise vector made every plan accelerate (NuRec about 20 points down). Zero noise was best. A mean of 16 samples, more Euler steps and the Heun solver did not help | Adopted          |
| Horizon of 6 s instead of 4 s                                                                                                                         | A small gain on both tracks                                                                                                                                                | Adopted          |
| QK-norm in the planner attention                                                                                                                      | Score neutral. It stops the attention-logit growth that made some posttraining seeds diverge                                                                               | Adopted          |
| Unicycle actions (acceleration and curvature, as in Alpamayo), three variants                                                                         | NuRec −7.5 to −5.8, or a tie                                                                                                                                               | Dropped          |
| Bounded acceleration and lateral-acceleration actions, with a pose loss on the rollout                                                                | nuPlan +2.0, but NuRec −7.3 from lateral drift                                                                                                                             | Auxiliary output |
| Path added in posttraining, together with target points                                                                                               | nuPlan −8.4: the car cuts sharp turns toward the target points                                                                                                             | Dropped          |
| Path learned in midtraining, without target points                                                                                                    | After posttraining: slightly better on both tracks                                                                                                                         | Adopted          |
| Critic head: predicts collision and lane-exit labels for a fixed set of candidate trajectories, as an auxiliary loss                                  | Midtraining a tie. Posttraining worse on NuRec, more exits from the lane corridor                                                                                          | Dropped          |
| Waypoints and actions do not attend to each other                                                                                                     | Slightly worse on both tracks                                                                                                                                              | Dropped          |

### Route target points

| Change                                                                    | Result                                                                                                                                            | Verdict  |
| :------------------------------------------------------------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------ | :------- |
| One target point at 45 m, always present                                  | The policy steers to the point. Rotate the point and the plan rotates too, through parked cars. nuPlan −9.1 against the run without target points | Replaced |
| Midtraining without target points, target points only in posttraining     | Part of every result below                                                                                                                        | Adopted  |
| Target point dropout 0 / 25 / 50 / 75 %                                   | Monotonic: nuPlan +9.5 and NuRec +3.1 from dropout 0 to 75 %. Offroad scenes fall from 175 to 68 per seed                                         | Adopted  |
| Three target points at 40, 60 and 80 m, each dropped on its own           | On or slightly above the one-point curve at the same share of samples without any point. At 75 %: slightly above the one-point run on both tracks | Adopted  |
| Share of samples without any point: 75 / 85 / 95 %, with the path         | nuPlan +2.9 and NuRec +6.3 from 75 to 95 %; 85 and 95 % tie. At 75 %, offroad doubles                                                             | Adopted  |
| Drop the near points more often than the far points                       | A tie with uniform dropout, both at 75 % and at 95 %                                                                                              | Dropped  |
| Lateral noise on the target points: 1 to 8 m, or 0.5 m on the 95 % recipe | 1 to 8 m: nuPlan up to +8.8, but NuRec worse by 2 to 6. 0.5 m: slightly worse on both                                                             | Dropped  |
| Target points as adaptive-norm conditioning instead of tokens             | Worse on NuRec                                                                                                                                    | Dropped  |
| Test time: fewer target points                                            | No point: NuRec −13.5. Only the 80 m point: slightly worse. 60 and 80 m: a tie                                                                    | Info     |

### Data and curation

| Change                                                                     | Result                                                                                                                                       | Verdict |
| :------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------- | :------ |
| Reweight hard frames (sharp turns, stops, close leads) against easy frames | Every weight from 0.3 to 1 is within noise on both tracks                                                                                    | Dropped |
| Train mostly on frames where a constant speed is wrong                     | NuRec level with or below two uncurated runs, and falling as the easy-frame weight drops from 0.25 to 0.025. More overspeed at lower weights | Dropped |
| Upweight the NAVSIM navtrain scenes in nuPlan training                     | A tie at weight 2, slightly worse at weight 4                                                                                                | Dropped |
| Remove nuPlan from the training mix                                        | A tie                                                                                                                                        | Neutral |
| Offline distillation from Alpamayo trajectories                            | Not run. The estimated gain was inside the noise                                                                                             | Dropped |

### Training

| Change                                    | Result                                                                                                                                              | Verdict |
| :---------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------- | :------ |
| Freeze the encoder in posttraining        | No score cost, 1.6 times faster                                                                                                                     | Adopted |
| Half of the epochs                        | Midtraining: nuPlan −1.8, more offroad. Posttraining with 21 instead of 45 epochs: a tie, but the recipe keeps 45 to match the released checkpoints | Dropped |
| AMSGrad and a post-LayerNorm planner      | Gradient spikes froze training or made single seeds diverge                                                                                         | Dropped |
| Weight decay 0.03 or 0.05 instead of 0.01 | 0.03 in posttraining: slightly worse on nuPlan, a tie on NuRec. 0.05 shrinks the speed encoder, and some seeds never drive faster than about 15 m/s | Dropped |
| One training instead of three stages      | Not finished                                                                                                                                        | Open    |

### Inference and evaluation

| Change                                            | Result                                                                                                                                 | Verdict |
| :------------------------------------------------ | :------------------------------------------------------------------------------------------------------------------------------------- | :------ |
| Replanning at 10 Hz instead of 2 Hz               | Older models: 2.5 times more lateral collisions, because the car follows every flicker of the plan. Released model: 1 to 10 Hz all tie | 2 Hz    |
| Drive the action rollout instead of the waypoints | Better below 8 m/s with fewer stalls, worse above 8 m/s from lateral drift. a tie on nuPlan, slightly worse on NuRec                   | Open    |

### Training speed

| Change                                                                 | Result                                                                                          | Verdict |
| :--------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------- | :------ |
| Three seeds in one job with one data loader and CUDA graphs            | About the cost of one seed (see [Training](training.md))                                        | Adopted |
| One CUDA stream per seed                                               | No gain: the step was limited by kernel launches                                                | Dropped |
| `torch.vmap` over seeds                                                | Two times slower                                                                                | Dropped |
| Triton kernels: implicit-GEMM convolution, depthwise 7x7, RegNet block | Slower than cuDNN or a tie. The CUDA RegNet block cut the step by 16 % against the Triton block | Dropped |
| CUDA kernels for Hopper                                                | 17 to 35 % shorter training steps (see [Training](training.md))                                 | Adopted |

### Not tried yet

- Scaling: train on all frames instead of sparse anchor frames, and on fractions of the Physical AI AV data, to measure how the policy scales with data.
- Test-time choice of the driven representation: for example, waypoints timed along the predicted path, or actions at low speed and waypoints at high speed. Also flow temperature tuning.
- An input resolution sweep on the full front image.
- A lower learning rate for pretrained encoders such as DINOv3.
- A ViT encoder with joint attention over image and BEV tokens. A BEVFormer encoder.
- The Muon optimizer, weight EMA and a warmup-stable-decay schedule.
- Logit-normal noise levels for flow matching (see the motion planner notes).
- Hood swap as a training augmentation.
- Closed-loop training. The streaming backbone is made for it.
- Rule-based logic on perception outputs, such as emergency braking from detected boxes, and trajectory scorers. We left them out by choice, to keep the policy end-to-end.
