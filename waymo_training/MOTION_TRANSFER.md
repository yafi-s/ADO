# Waymo motion speed forecasts in CARLA

The hybrid completed the clear route using ADO for every driving command after warmup. It collided with the stopped lead and both crossing pedestrians. The Waymo model controlled speed immediately before each collision. These four seed-31 development runs used the same 64 m Town01_Opt route on CARLA 1.0. Maximum lateral error was 0.38 m in all four. [Machine-readable outcomes and checkpoint hashes](motion_transfer_development.json)

| Development case | Outcome | Progress | Waymo pedal-control frames | ADO pedal-control frames |
| --- | --- | ---: | ---: | ---: |
| Clear street | Complete | 63.67 m | 0 | 133 |
| Stopped lead | Contact with vehicle | 31.30 m | 63 | 34 |
| Pedestrian crossing | Contact with pedestrian | 33.76 m | 23 | 72 |
| Occluded crossing | Contact with pedestrian | 33.85 m | 38 | 62 |

Each case also had ten stationary-hold ticks to acquire a full second of measured speed history. The controller used each synchronized observation to choose the command for the following tick. On the final tick **before the first recorded contact**, the model requested acceleration in all three hazard cases.

- **Stopped lead.** From 3.51 m/s it forecast 6.04, 7.04 and 8.29 m/s at one, two and three seconds. The servo requested 0.706 throttle and zero brake.
- **Crossing.** From 4.13 m/s it forecast 6.51, 8.48 and 9.92 m/s. The servo requested 0.625 throttle and zero brake.
- **Occluded crossing.** From 5.08 m/s it forecast 7.67, 9.85 and 11.43 m/s. The servo requested 0.582 throttle and zero brake.

The stopped-lead collision frame itself exposed a separate actuator issue. After the impact reduced measured speed to 1.41 m/s, the model forecast zero at all three horizons and the servo still requested 0.188 throttle. Its acceleration feedback retained a positive pedal correction. Across the four traces, the collision frame was the only instance of positive throttle alongside an all-zero forecast. The preceding command had already advanced the simulator into contact. The measured forecasts were poor close to all three hazards. Changing the servo would require another closed-loop test to establish an effect.

## What controlled the car

The ridge model trained on 1,503 real Waymo Open Motion Dataset v1.3.1 scenarios. It estimated ego speed at one, two and three seconds from measured ego history, lane bend and annotated road users. It achieved mean absolute error of 0.501 m/s on 582 separate validation scenarios. History alone achieved 0.500 m/s, constant velocity 1.052 m/s and constant acceleration 0.557 m/s. The paired 95% scenario-bootstrap interval for interaction minus history error was −0.011 to +0.012 m/s. The logged experiment leaves the improvement from explicit interaction features unresolved. [Training and evaluation](interaction.py) · [Waymo Open Motion Dataset](https://waymo.com/open/data/motion/)

In CARLA, the frozen ADO behavior-v2 model supplied steering. NanoDet front-camera boxes and roof-lidar returns established track positions, with radar corroboration when available. Successive measured positions and ego localization provided ground velocities. The transfer used its Waymo forecast when there was a qualifying lead or vulnerable road user, including short tracked coasts. Otherwise it used behavior-v2's learned sensor-based pedals. The signed servo converted the three speed forecasts to one acceleration request. Throttle response used an earlier Lincoln calibration and the brake scale was nominal. Logged future speed reflects the human driver's earlier decisions and circumstances. Treating it as a desired closed-loop speed failed in these development encounters. [Transfer and feedback implementation](interaction_transfer.py) · [Sensor tracker](../crosswalk/tracking.py) · [ADO behavior model](../crosswalk/behavior_v2.py)

## Run the private development trial

Start from the repository root with the CARLA Python environment, the private Waymo speed checkpoint and its adjacent notice/agreement, the frozen ADO behavior-v2 checkpoint, and an available CARLA 1.0 Town01_Opt server. Use a new private output directory. The CLI saves the model and source hashes, actual server/map identity, per-tick controls and forecast ownership, scenario outcomes and trace hashes. This seed-31 trial used the four default development cases. No official benchmark or test-district cases were run in this trial.

```powershell
python -m waymo_training.interaction_transfer --motion "A:\private\motion-speed.npz" --steering "A:\private\behavior-v2.npz" --output "A:\private\new-town01-trial" --port 2868 --attach --seed 31
```

## Dataset attribution

The Waymo-trained model, associated materials and downstream use or modification are governed by the Waymo Dataset License Agreement for Non-Commercial Use. This repository summary contains aggregate CARLA outcomes and source code. The Waymo records, extracted examples and trained weights stay private. [Full agreement](WAYMO_DATASET_AGREEMENT.txt) · [Waymo terms](https://waymo.com/open/terms/)

```bibtex
@misc{waymo_open_dataset,
  title = {Waymo Open Dataset: An autonomous driving dataset},
  website = {\url{https://www.waymo.com/open}},
  year = {2019-2025}
}
```
