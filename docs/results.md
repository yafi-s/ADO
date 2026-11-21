# Town01_Opt driving results

The v0.4 Town01 policy passed all 16 targeted assessment drives and both additional interactive trips. Its matched clear-road control took 15.1 s. The v0.4 Town03 film separately passed 11/11 encounters and 11/11 junctions across 3,354 exposures. Both CARLA-calibrated Waymo research arms passed all four cases on one development seed, including the arm using only CARLA inputs. [Town01 assessment design and scores](../results/town01-yield-gated-v04/README.md) · [Town03 release review](../results/urban-pace/README.md) · [Watch the current film](https://yafi-s.github.io/ADO/?run=urban-pace) · [Waymo calibration results](../waymo_training/CARLA_SPEED_CALIBRATION.md)

## Archived Town01 evaluations

The following is the archived v0.1 evaluation. The v0.2 comparison reports 47/50 clean finishes against 32/50 on fresh matched cases, and 97/100 against the historical 70/100. The complete comparison archive includes every saved frame trace. [Read the v0.2 results](../results/behavior-v2/README.md) · [View comparisons by case](https://yafi-s.github.io/ADO/benchmarks.html)


The fusion policy finished **70/100** primary trials cleanly in CARLA 1.0 on Town01_Opt. The primary suite exercised 10 cases across 10 seeds (101, 102, 103, 104, 105, 106, 107, 108, 109, 110). The scored ego vehicle made 6.35 km of route progress across 7.64 km of planned routes.

The runtime recorded CARLA server version `1.0` and client version `1.0`. The training archive records server version `1.0`, client version `1.0` and map `Town01_Opt`.

[Primary trials](../results/primary.json) · [Evaluation protocol, environment and hashes](../results/manifest.json) · [CARLA 1.0 release](https://github.com/carla-simulator/carla/releases/tag/1.0)

## Primary suite

| Primary case | Trials | Route reached | Clean finishes | Crashes | Red-light outcomes | Departures | Timeouts | Yield breaches | Triggered / cleared | Route layouts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pedestrian | 10 | 9 | 9 | 0 | 0 | 0 | 1 | 0 | 9/10 / 8/10 | 9 |
| pedestrian-delayed | 10 | 7 | 7 | 2 | 0 | 0 | 1 | 0 | 9/10 / 6/10 | 9 |
| lead | 10 | 9 | 9 | 0 | 0 | 0 | 1 | 0 | 10/10 / 10/10 | 9 |
| braking | 10 | 10 | 10 | 0 | 0 | 0 | 0 | 0 | 10/10 / 10/10 | 9 |
| occluded | 10 | 2 | 2 | 0 | 0 | 0 | 8 | 0 | 9/10 / 8/10 | 10 |
| signal | 10 | 10 | 10 | 0 | 0 | 0 | 0 | 0 | 10/10 / 10/10 | 9 |
| left | 10 | 5 | 3 | 0 | 0 | 5 | 0 | 7 | 10/10 / 9/10 | 7 |
| turn | 10 | 5 | 5 | 0 | 0 | 1 | 4 | 0 | — | 10 |
| clear | 10 | 9 | 9 | 0 | 0 | 0 | 1 | 0 | — | 9 |
| trip | 10 | 6 | 6 | 0 | 0 | 3 | 1 | 0 | 9/10 / 9/10 | 2 |
| **Total** | 100 | 72 | 70 | 2 | 0 | 9 | 17 | 7 | 76/80 / 70/80 | 39 |

The runner assigns one terminal outcome per trial. Collision takes precedence over a red-light outcome, departure and completion. Route reaches are the runner's completion outcomes. A clean finish also requires zero recorded collisions, red-light violations and early entries into the oncoming turn. In total 0 primary trials recorded a red-light violation. The yield column counts trials with a breach even when another outcome ends the trial. The trigger and clearance fractions use the eligible trials in each row as the denominator.

Primary contact actors by number of trials: `walker.pedestrian.0016` (1), `walker.pedestrian.0024` (1). The collision sensor logged 2 contact events.

Weather by primary trial: `ClearNoon` (60), `ClearSunset` (30), `WetNoon` (10). Weather presets select the simulated appearance. The driving study used stock tire friction.

## Geographic holdout

The demonstration seeds (11, 12, 13, 14, 15, 16) and primary seeds (101, 102, 103, 104, 105, 106, 107, 108, 109, 110) share 0 seed values. The primary trials used 39 unique route IDs across all cases. Route IDs describe the recorded start and end waypoints. A route can recur under several cases and seeds, including the two trip destinations.

| Recorded route IDs by type | Demonstrations | Primary tests | Shared IDs |
| --- | --- | --- | --- |
| OpenDRIVE roads outside junctions | 0, 1, 2, 9, 10, 16, 25 | 5, 6, 7, 12, 19, 24 | ∅ |
| OpenDRIVE junctions | 26, 54, 222, 332 | 87, 143, 194, 255 | ∅ |

These sets use the planned ego routes stored in the episode records. Junction IDs identify the intersection traversals. The holdout covers a separate district of the same CARLA town, under the same sensor and vehicle setup with ideal supplied route localization. `braking` is the protocol's withheld encounter family.

## Paired camera comparison

Camera and fusion trials were joined by the exact seed and case ID. 20/20 trials also matched the recorded route, destination, weather, initial pose perturbations and crossing delay. Actions can change when the distance-triggered encounters start. Counts summarize the sampled pairs.

| Case | Pairs | Both clean | Fusion only clean | Camera only clean | Neither clean | Fusion / camera crashes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pedestrian | 2 | 1 | 1 | 0 | 0 | 0 / 0 |
| pedestrian-delayed | 2 | 0 | 1 | 1 | 0 | 1 / 0 |
| lead | 2 | 1 | 1 | 0 | 0 | 0 / 0 |
| braking | 2 | 1 | 1 | 0 | 0 | 0 / 1 |
| occluded | 2 | 0 | 0 | 1 | 1 | 0 / 0 |
| signal | 2 | 2 | 0 | 0 | 0 | 0 / 0 |
| left | 2 | 0 | 1 | 0 | 1 | 0 / 0 |
| turn | 2 | 0 | 0 | 0 | 2 | 0 / 0 |
| clear | 2 | 2 | 0 | 0 | 0 | 0 / 0 |
| trip | 2 | 0 | 1 | 0 | 1 | 0 / 0 |
| **Total** | 20 | 7 | 6 | 2 | 5 | 1 / 1 |

[Camera cohort](../results/camera_ablation.json) · [Primary cohort](../results/primary.json)

## City trips and crossing timing

- **Market** had 4/5 clean finishes on 184.7 m routes. Planned road sequence `6 → 24 → 12`, junction sequence `87 → 255`.
- **Station** had 2/5 clean finishes on 185.7 m routes. Planned road sequence `12 → 19 → 6`, junction sequence `194 → 143`.

Seed 101 uses the Market destination. Recorded setup matches across both trials. Crossing start and clearance are the first scored frames where the respective trace flags are set.

| Trial ID | Route ID | Configured delay | Crossing start | Crossing cleared | Outcome | Route progress |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| `101-trip-fusion` | `heldout:6123726579003061302:4225981044877295467` | 0 s | — | — | timeout | 1.3/184.7 m |
| `101-trip-delayed-fusion` | `heldout:6123726579003061302:4225981044877295467` | 1 s | — | — | timeout | 1.2/184.7 m |

[Supplemental trial](../results/supplemental.json) · [Per-tick traces](../results/traces.jsonl.gz)

## Observed failures

- `101-occluded-fusion` ended with timeout at 24.2/64.0 m after 36.0 simulated seconds. Rerun the settings with `python -m crosswalk.drive --scenario occluded --seed 101 --policy fusion`.
- `101-turn-fusion` ended with lane departure at 21.9/60.9 m after 13.0 simulated seconds. Rerun the settings with `python -m crosswalk.drive --scenario turn --seed 101 --policy fusion`.
- `102-pedestrian-delayed-fusion` ended with collision with walker.pedestrian.0024 at 33.0/64.0 m after 7.2 simulated seconds. Rerun the settings with `python -m crosswalk.drive --scenario pedestrian --seed 102 --policy fusion --crossing-delay 1`.
- `104-pedestrian-delayed-fusion` ended with collision with walker.pedestrian.0016 at 34.4/64.0 m after 8.2 simulated seconds. Rerun the settings with `python -m crosswalk.drive --scenario pedestrian --seed 104 --policy fusion --crossing-delay 1`.
- `107-left-fusion` ended with completion with a violation at 66.9/67.1 m after 15.3 simulated seconds. Rerun the settings with `python -m crosswalk.drive --scenario left --seed 107 --policy fusion`.
- `101-braking-camera` ended with collision with vehicle.tesla.model3 at 27.7/64.0 m after 8.4 simulated seconds. Rerun the settings with `python -m crosswalk.drive --scenario braking --seed 101 --policy camera`.

[All scored trial summaries](../results/episodes.json) · [Per-tick traces keyed by trial ID](../results/traces.jsonl.gz)

## Timing

The fixed control timestep is 0.1 s (10 simulated Hz).
The 100 primary trial loops with wall times processed 13.82 scored frames per wall-clock second in aggregate. The timer includes per-tick simulation, perception, control, any enabled recording and asynchronous collision draining after the final scored frame.

## Camera rendering

The benchmark used a native-resolution front RGB view and requested DLSS at 50% screen percentage on auxiliary RGB views. Recorded benchmark episodes have five synchronized views: front, left, right, chase and modeled infrared.
The depth and semantic cameras used a `native` profile.
Spawned actors returned matching settings for 205/205 DLSS-requested camera instances. CARLA can fall back to another upscaling path.
GPU DLSS activation was unmeasured in these trials.

[Camera configuration and readback per trial](../results/primary.json) · [CARLA 1.0 RGB camera reference](https://github.com/carla-simulator/carla/blob/1.0/Docs/ref_sensors.md#rgb-camera)

A separate local timing profile selected `mixed` RGB rendering.

[Hardware, timings and rendering profile](../results/rendering.json)

Showcase `showcase-211-pedestrian-fusion` is a separate closed-loop drive. Its chase recording is 1920×1080 with requested `High` quality. The policy's front camera video is 640×360. [Showcase capture manifest](../web/media/showcase/pedestrian/manifest.json)

## Development comparison

The initial and updated checkpoints matched on 36 drives by seed, exact case ID, policy and recorded setup. Development used seed(s) 31, 32 in the training district for iteration ahead of the held-out evaluation.

| Policy | Paired drives | Clean finishes before → after | Timeouts before → after | Contact trials before → after |
| --- | ---: | ---: | ---: | ---: |
| Fusion | 18 | 4 → 12 | 10 → 3 | 1 → 0 |
| Camera | 18 | 8 → 13 | 2 → 2 | 3 → 0 |

[Initial development drives](../results/development-baseline/episodes.json) · [Updated development drives](../results/dev/episodes.json) · [Updated development protocol and model hashes](../results/dev/manifest.json)

## Demonstrations and offline checks

48 Traffic Manager demonstration drives covered 6 seeds. The collection outcomes were `complete` (44), `lane_departure` (4).

- **Fusion** trained on 10,086 synchronized demonstration rows with 27 input features. The pre-refit comparison on 1,636 demonstration frames from group(s) 11 measured mean absolute actuator errors of steer 0.153, throttle 0.123, brake 0.059. [Fit log](../results/fit-fusion.json)
- **Camera** trained on 10,086 synchronized demonstration rows with 22 input features. The pre-refit comparison on 1,636 demonstration frames from group(s) 11 measured mean absolute actuator errors of steer 0.153, throttle 0.176, brake 0.125. [Fit log](../results/fit-camera.json)

The compact pedal heads used 21 fusion inputs and 16 camera inputs.

Both policies use the same mapped-RGB signal specialist. Static map geometry locates the lamps and the RGB image supplies their color. The manifest records 8 driving lessons, 486 demonstrated controls and 16 static color pairs for the specialist. [Signal lessons and model](../data/signals/README.md)

[Demonstration outcomes](../results/collection.json) · [Training features and labels](../data/training.npz) · [Methods and paper references](research.md)

## Separate tire-friction trials

9 straight-line trials commanded full braking from near 8 m/s in CARLA 1.0 on Town01_Opt. The trials scaled the Chaos Vehicles wheel field `WheelPhysicsControl.friction_force_multiplier` relative to stock. The stopping distance is the along-route displacement when speed first reaches 0.15 m/s or below. CARLA samples the threshold at 0.05 s ticks.

| Stock friction multiplier | Stops / trials | Median distance | Distance range | Median time |
| --- | ---: | ---: | ---: | ---: |
| 1× | 3/3 | 5.04 m | 5.04–5.04 m | 1.20 s |
| 0.5× | 3/3 | 5.04 m | 5.04–5.04 m | 1.20 s |
| 0.25× | 3/3 | 7.58 m | 7.58–7.58 m | 1.85 s |

Brake-entry speeds ranged from 8.07 to 8.08 m/s.
Final wheel friction readback matched the requested setting in 9/9 trials. 9/9 trials matched the stock setup for their seed. These simulator settings describe one vehicle on straight routes.

[Tire-friction trial data](../results/traction.json) · [CARLA Chaos Vehicles vehicle physics controls](https://github.com/carla-simulator/carla/blob/1.0/Docs/tuto_vehicle_physics.md)

Regenerate this report with `python -m crosswalk.report --root .` using the published results.
