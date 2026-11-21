# CARLA calibration of the Waymo motion speed forecast

**Both newly fitted controllers completed all four seed-31 Town01 scenarios.** Each stopped for the lead vehicle, the visible crossing pedestrian and the occluded pedestrian, then continued after the hazard cleared. All eight drives recorded zero collisions, yield violations and red-light violations. The original Waymo transfer collided on all three hazards. Adding the Waymo forecast to the new CARLA calibration supplied no additional successful outcome in this comparison. [Paired native evidence](carla_speed_native_development.json)

The frozen controller accelerated immediately before three seed-31 contacts. Replaying its saved feature vectors reproduces all three forecasts within 0.00000001 m/s. Removing only the tracked road users gives the following one-second predictions immediately before contact. [Original transfer](MOTION_TRANSFER.md)

- Stopped lead, traveling at 3.51 m/s. The forecast was 6.04 m/s with the tracked lead and 6.59 m/s without it.
- Pedestrian, traveling at 4.13 m/s. The forecast was 6.51 m/s with the tracked pedestrian and 6.55 m/s without.
- Occluded pedestrian, traveling at 5.08 m/s. The forecast was 7.67 m/s with the tracked pedestrian and 7.71 m/s without.

At the three-second horizon the pedestrian inputs raised the forecast by 0.39 m/s at the visible crossing and 0.51 m/s at the occluded one. The model extrapolated acceleration into both contacts. The last pre-contact commands across the three hazards applied 0.58–0.71 throttle. The ablation covers all 121 pre-impact decisions under Waymo control. [Per-horizon measurements](carla_speed_development.json)

Keeping the tracks and current speed fixed while setting both measured half-second acceleration features to zero reduced the same one-second forecasts to **3.29, 4.04 and 5.02 m/s**, respectively. The difference isolates how recent ego acceleration reinforced the original go command in each encounter. The counterfactual histories were evaluated offline. [Feature definitions and frozen model inference](interaction.py)

## Learning a CARLA target speed

Twenty native demonstrations covered the clear street, stopped lead, pedestrian, delayed pedestrian and occluded crossing on seeds 43, 44, 47 and 48. Each comparison fitted on three seeds and evaluated on the fourth. There were 3,180 scored frames after a one-second history warmup per episode. Both planners used the same 40-tree regression settings, route-relative lidar and radar, front-camera detector summaries and measured ego motion. One planner also received the frozen Waymo one-, two- and three-second forecasts. The CARLA teacher targets used the existing demonstration speed labels and privileged clearance supervision. Driving inputs contain only the sensors, ego motion and planned route. [Calibration and drive code](carla_speed_calibration.py) · [Demonstration provenance](../data/behavior-v2/README.md)

The lessons store summarized ranges and camera detections without raw clouds or persistent track IDs. A straight-road adapter matches camera bearings and apparent heights to lidar returns, estimates velocity from recent positions and provides the **same reconstructed measurements to both planners**, during fitting and driving. These tracks supported 304 vehicle and 370 pedestrian feature rows. The original collision probe above uses the archived full sensor tracker. [Per-episode results](carla_speed_development.json)

Mean absolute target-speed error, weighted equally across the 20 held-out episodes, fell from **0.0830 to 0.0747 m/s** with the forecasts. Three of the four held-out seeds improved. On all 457 requested-stop frames (teacher target below 1 m/s), frame-weighted error increased from **0.2068 to 0.2109 m/s**. Cyclically shifting forecast triplets within each held-out lesson moved the fitted target by 0.488 m/s on average across episodes. This perturbation tests dependence on the feature and does not represent an observed physical scene. These runs use closely related routes and fixtures in one town. [Paired fold evidence](carla_speed_development.json)

## Paired native Town01 drives

The paired protocol was frozen before the first native trial. The same CARLA 1.0 Low server ran both controllers on the same 64 m route, with a 10-tick stationary history warmup, learned steering and learned pedal feedback. Each controller trained on the same four teacher seeds and evaluation used the original failure seed, 31. Arm order alternated by scenario. Checks on all eight trace hashes, 1,445 contiguous scored frames, and forecast presence by arm matched the protocol. [Native run summaries and trace hashes](carla_speed_native_development.json)

- **Clear street.** Completion took 14.1 s with Waymo and 14.7 s with CARLA alone.
- **Stopped lead.** Both controllers stopped for 13 ticks. Minimum bumper gaps were 2.731 m with Waymo and 2.750 m with CARLA alone. The runs completed in 18.3 and 18.2 s.
- **Pedestrian.** Minimum gaps were 2.642 and 2.656 m. The cars registered 22 and 23 stopped ticks and completed in 19.5 and 19.6 s.
- **Occluded pedestrian.** Minimum gaps were 2.785 and 2.720 m. Both stopped for 28 ticks and completed in 20.1 and 20.0 s.

One tick represents 0.1 s. A stopped tick required measured speed below 0.35 m/s within 12 m of the scored hazard and after 2 m of route progress. The three paired minimum-gap differences were all under 7 cm. One pair per scenario on one known development seed cannot establish equivalence, generalization, or a Waymo safety advantage. The research controller remains outside the public city driver.

## Separate Waymo camera predictor

The original E2E camera checkpoint drove a different system. For each applied decision in its two rolling hazard trials, we replaced the three decoded camera views with the clear trial's simultaneous views and kept the hazard run's motion history and route intention fixed. The CARLA rear-axle positions agreed to within 1.1 cm. Each trial contributed 34 applied decisions before lane departure.

The swap changed the path follower's target speed by a mean absolute **0.00163 m/s** for the stopped lead and **0.00080 m/s** for the pedestrian. Replaying the compressed video changed the target by roughly 0.00021 m/s relative to the original sensor frames in each trial. The pedestrian was never measured in-path within 12 m in this recording. The two tests establish weak response to these particular scene changes. [Camera probe](carla_camera_probe.py) · [Measured camera effects](carla_camera_development.json)

## Research terms

The Waymo models, associated materials, and downstream use or modification are governed by the [Waymo Dataset License Agreement for Non-Commercial Use](WAYMO_DATASET_AGREEMENT.txt). Model checkpoints, full traces and training records remain private. Attribution to the Waymo Open Dataset, 2019–2025. [Waymo Open Dataset](https://waymo.com/open/) · [Dataset notice](NOTICE.md)

To repeat the paired fit in the project Python environment, use NumPy 2.5.5 and scikit-learn 1.9.1. Supply the private Colab checkpoint and hash-matched seed-31 traces. The native command refits on all four teacher seeds and attaches to an idle CARLA 1.0 Town01_Opt server. Use new private output paths with the dataset agreement and notice alongside them.

```powershell
python -m waymo_training.carla_speed_calibration --motion $motion --archived $originalSeed31 --output $newOfflineSummary
python -m waymo_training.carla_speed_calibration --motion $motion --native-port 2868 --output $newNativeSummary
python -m waymo_training.carla_camera_probe $originalVisionRuns $visionCheckpoint $newCameraSummary
```
