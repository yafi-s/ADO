# Surround sensing and urban overtaking

The rider display and camera movies share the same CARLA frame clock. Select **Front**, **Left**, **Right**, **Rear left** or **Rear right** to see the measurements that supplied a track. **Chase** is an independent spectator camera. The center screen renders the revised I-PACE from Blender; the camera movies show CARLA's Lincoln carrying the simulated sensors.

## Current rider recording

The v0.4 replay pairs all 3,354 native exposures with the rider display and vehicle pose. At frame 3460 the front camera shows the approach to the striped barricades. The full drive scored eleven encounters and eleven junctions. Separate work-zone films compare stopping for a board across the lane with continuing alongside a board at the curb. [Watch the v0.4 drive](https://yafi-s.github.io/ADO/?run=urban-pace) · [Inspect the barricade exposure](https://yafi-s.github.io/ADO/inspect.html?run=urban-pace&frame=3460&camera=front) · [Work-zone films](../results/workzone-stop/README.md) · [Current perception and motion review](../results/urban-pace/README.md)

## Two moving overtakes from v0.3

[Open-lane pass](https://yafi-s.github.io/ADO/?run=overtake-clear) · [Wait for adjacent traffic](https://yafi-s.github.io/ADO/?run=overtake-yield) · [Physical measurements and no-lead control](../results/tactical-pass/paired-4fppmp4k/summary.json)

Both recordings place a slower moving car in the original lane on a Town03 boulevard with a broken lane line. The tactical model measures the lead's distance and motion and the front and rear gap in the passing lane using the roof lidar. A supervised model selects keep, yield or pass. Its labels come from synthetic counterfactual traffic rollouts that score progress and collision risk. The car steers along the chosen lane preview. Ordinary steering and pedal feedback are fitted from demonstrations. When the tactical classifier selects yield, a brake hold keeps the car in its source lane until it accepts the gap.

- With a clear adjacent lane, ADO completed the overtake at 7.8 seconds with 1.535 m minimum lateral clearance.
- With a car alongside, ADO yielded for 4.7 seconds. It committed at 8.6 seconds after that car's rear was 33.0 m ahead, and completed its pass at 15.9 seconds with 1.535 m minimum clearance.
- In the separate empty-road control, it drove 45.5 m with 0.001 m maximum displacement from the original lane center.

These are paired development scenarios on one legal two-lane segment. The viewer's path comes from the recorded tactical commitment and corridor geometry. Its vehicle outlines are estimates from the cameras and range sensors. CARLA supplies independent actor footprints for scoring. [Tactical training and inference](../crosswalk/overtake.py) · [Training objective](../crosswalk/overtake_fit.py)

## Inspect a single exposure

[Inspect the moving-lead pass one frame at a time](https://yafi-s.github.io/ADO/inspect.html?run=overtake-yield&frame=8057&camera=chase). The inspector opens paused. Step forward or backward by one recorded exposure and switch between cameras at that instant. The overhead view independently toggles the rider drawing, sensor measurements and CARLA actor footprints. Each uses the same metric coordinate frame. This reference overlay is available only for inspection and scoring.

For example, on frame 8057 the native chase image shows the ego beside the white lead car. The inspector reports the actor's body center, the track estimate, and the displayed center in metres ahead and right of the ego. The box uses the native blueprint's measured offset from the actor origin. These positions allow a reviewer to separate an actual moving vehicle from a wobbling estimate. [Actor body calibration](../web/media/map-features/Town03_Opt-bodies.json)

## What the screen measures

Five overlapping RGB cameras cover the horizontal circle. A full-sweep lidar and five Doppler radars supply range and relative-motion measurements. The display can follow the same object across views, briefly extrapolate an occluded track and show a camera box while depth is unresolved. An object behind the vehicle can therefore remain visible even when the front camera loses sight of it.

In a pose-aligned pedestrian crossing, the new tracker matched the person in 120 of 155 frames, compared with 38 before. In the rear sector the counts were 21 of 22 and zero of 22. The rider display matched the pedestrian in 113 frames compared with 59. The audit includes false classifications, identity interruptions and unobserved road users. [Full audit and original frames](../results/perception/README.md) · [Reproject the original range measurements](../results/surround-rig/README.md) · [YOLOX and NanoDet camera comparison](../results/surround-recognition/README.md)

Traffic-light posts and heads have surveyed locations. Camera pixels determine the displayed color, with brief retention between consistent readings. A mapped head with insufficient evidence is drawn neutral. Scoring records the simulator's actual light state separately. [Camera measurement](../crosswalk/signal_observer.py) · [Stationary calibration measurements](../models/surround-signals/evaluation.json)

The learned urban cruise head was fitted on Colab. Its original front-camera policy inputs and route-conditioned ranges govern ordinary speed and steering. Full-sweep lidar also informs tactical passing. The surround detections and corner-radar associations feed the rider display. [Speed measurements](../results/city-cruise/README.md) · [Research sources](research.md) · [Updated front and rear vehicle renders](../assets/waymo-ipace/front-rear.png)
