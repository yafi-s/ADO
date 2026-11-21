# Native CARLA surround perception

A crossing pedestrian remained track 5 as the ego car passed it. On the seed 211 Town03 run, the same ID appears on 112 consecutive frames from 4.2 to 15.3 s. Of those, 97 frames have fresh range support and 15 are coasts. Front, left and rear-left RGB observations contributed to the same track. Against CARLA's separately recorded pedestrian position, the median horizontal error on those 97 measured frames was 0.061 m and the largest was 0.144 m. This is one selected handoff. Earlier, sparser sightings of this walker had separate IDs.

| Snapshot | Native observation | Reconstructed evidence |
|---|---|---|
| [Crossing, 9.6 s](crossing-left/) | Person 5, left RGB | 16 object lidar returns and 24 front-left radar returns, median Doppler −0.937 m/s |
| [Crossing, 12.0 s](crossing-rear-left/) | Person 5, rear-left RGB | 21 object lidar returns and 8 rear-left radar returns, median +1.623 m/s |
| [Roadside vegetation, 24.5 s](rear-right-edge/) | Seed 212 rear-right RGB assigned a car label to a sliver at the edge of the image | 10 lidar and 22 rear-right radar returns, median +0.234 m/s. A simultaneous right-camera view shows the hedge. The revised tracker retains the 2D candidate and rejects it as a new 3D car. |

The seed 211 preflight stopped at its planned 155-frame limit. Each selected snapshot contains the original five lossless CARLA RGB images, raw roof lidar and five radar arrays, ego pose, camera boxes and time-stamped track state. The registry and image review of the false positive are saved separately. [Semantic review evidence](../perception/seed212-hedge-source.zip)

## Rig and measurement

The front image is 640×360. The other four are 320×180. The five level cameras have 90° horizontal fields of view and yaw angles of 0°, −70°, +70°, −145° and +145°. They capture at 10 Hz. YOLOX-S runs at 5 Hz per camera, alternating same-tick auxiliary camera pairs. Optical centres are displaced from one another, so near-field parallax, occlusion and image-edge truncation can still hide a road user. The roof lidar makes one 360° revolution per 0.1 s tick. A 36° forward radar and four overlapping 100° corner radars report signed line-of-sight speed. The native exposure poses of all eleven actors matched the configured mounts within 0.012 mm and 0.018° on the preflight. [Rig calibration](calibration.json)

A new classed 3D object needs an RGB detection and a compatible, unambiguous lidar surface. A unique new lidar surface can extend a recent track between RGB updates. A new unclassified moving object requires positive lidar rays through both newly occupied and vacated space in successive ego-compensated sweeps. Missing range leaves a camera bearing and box. Established tracks can coast for 0.8 s, and one-off tracks for 0.3 s. A lidar sweep passing through the expected surface can remove the displayed track earlier. Short camera histories allow a later depth fix to recover an ID while it is hidden. Dense clutter and a distant pedestrian returning only a few points can still cause missed tracks or ID changes. The retained seed 211 samples precede the image-history improvement. The seed 212 sample precedes the thin-edge safeguard.

In the city driver, the full lidar sweep influences the tactical passing lane preview and speed control. The learned cruise controller uses its trained front RGB and forward radar/range features. The five-view semantic tracks and corner Doppler associations feed the rider display and telemetry. Actor IDs and world positions in the handoff extract are scorer-only references.

## Verify

From the repository root with its Python dependencies installed, run

```sh
python results/surround-rig/verify.py
```

The verifier checks retained sample and model hashes, PNG sizes and raw arrays, finds the recorded surface samples in the native lidar cloud, and independently projects them into the recorded optical boxes. It reruns the spatial association on the raw radar returns to check sensor identity, hit counts and median signed Doppler. It then checks how a fresh tracker handles the same three observations and recomputes handoff position error from ego odometry and the separate actor reference. It requires no running simulator. The original full drives are outside this small evidence set. [Source manifest](manifest.json) · [Machine-readable audit](audit.json) · [Handoff frames](crossing-handoff.json)
