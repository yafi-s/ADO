# City Progress driver

The City Progress checkpoint extends the learned Town03 City Clearance driver. A second speed proposal uses current pavement visibility and the curvature of the actual navigation path. An approaching lateral surface can lower the final request to a complete stop. The driver keeps the original learned steering and pedal calibration.

## What was fitted

The retained Town03 instructor completed 3,141 native simulator frames on seed 75. The fit estimates a 10.64 m/s free-road speed from 165 settled cruise observations, a 1.73 m/s² lateral acceleration from 125 settled sharp-turn observations, and a 2.59 m/s² recovery rate from 313 accelerating observations. The instructor had a uniform low request at intersections, so the new fit uses measured motion and path geometry to estimate these quantities. The inherited stopping response was calibrated with four native braking trials and separately checked at three other speeds. The largest distance underestimate on those three checks was 0.039 m. Extra speed from the new proposal is limited to the highest tested braking speed of 8.08 m/s. [Fit report](fit.json) · [Town03 lesson](../../data/city-cruise/lessons.zip) · [Braking trials](../../results/roadworks-pace/stopping-calibration.json)

## Which observations change a request

- Three consecutive synchronized scans must show pavement through the swept lane and toward the conflict area. The observer checks camera usability, odometry and time, recent body returns, vulnerable road users, opposing traffic, and proximity to a controlling lamp. Reduced sight shortens the stopping budget. Route curvature, the posted limit, signed speed previews and tactical holds bound the proposal.
- Revision 4 also checks distant apparent body returns on a smooth uphill grade. It fits longitudinal road height from laser measurements on both sides of the swept lane, checks separately measured center returns and forward radar, and requires positive pavement beyond the surface. Surface residuals, motion, missing cross-sections and an unexplained return in the nine-scan history can withhold permission. This fit applies only to the additional progress proposal. The original obstacle readings stay in the inherited policy, actuator and telemetry. [Pavement fit](../../crosswalk/city_grade.py)
- A committed pass on a separate lane can use the measured continuous side of a stationary obstacle. Its far end limits how much stopping distance is available before a person could emerge. Ego-motion compensation tracks that end while the nearest side beam is clipped by the sensor's near range. The driver needs fresh evidence of the adjacent lane and positive clearance from the side surface. The blind end itself can still require the original low pace.
- A lateral object approaching from either side can impose a stopping constraint on both speed proposals and on pedal selection. The observer follows measured laser surfaces in world coordinates, compares their motion across multiple scans, and checks projected path entry, body extent and timing. A confirmed conflict persists through temporary loss. Recovery requires fresh pavement across the previously occupied region after the crossing interval, or having passed it.
- An established tactical wait requires consecutive measurements to release into ordinary following, or repeated positive pavement evidence where the previous lead was last observed.

Inference uses ego odometry, navigation geometry, RGB measurements, lidar and radar. Evaluation-only actor references are recorded separately. The 53 feature values and the selected model trace record the measured visibility, curvature, blind corner, lateral conflict, proposed speed and selected request. A trace's base request evaluates the inherited head using the actual previous speed request from that run. Matched v0.5 drives provide the independent baseline. [Observer and speed model](../../crosswalk/city_progress.py) · [Lateral motion guard](../../crosswalk/city_crossing.py) · [Tactical confirmation](../../crosswalk/overtake.py)

## Scope of the grade result

At the J1221 intersection, the old plane classified asphalt on the upcoming hill as an obstacle. On the same recorded input, revision 4 verified 28.98 m of pavement and raised the request from 13.01 to 29.07 km/h. In closed loop the car crossed the center at 17.54 km/h, compared with 13.61 km/h before the change. A peripheral camera detection on the empty fixture still caused a later 29.07 to 13.01 km/h request drop and 2.70 m/s² deceleration. On the separate 11° hill, the grade fit made no direct speed additions. Across all 3,320 range sweeps in the original City Life recording it also made none. The perpendicular crossing and parked hill obstacle have their own scored drives. [Recorded-input comparison](../../results/city-progress/driver/grade-replay.json) · [Remaining camera fallback](../../results/city-progress/driver/j1221-remaining-fallback.json) · [Archived City Life sweep results](../../results/city-progress/driver/grade-city-shadow.json) · [Independent hill and junction trials](../../results/city-progress/junction-transfer/README.md)

## Reproduction

From the repository root, reproduce the checkpoint and its fit report into a fresh output directory.

```powershell
.venv-city/Scripts/python.exe -m crosswalk.city_progress_fit --grade --output out/reproduced-city-progress
```

The checkpoint SHA-256 is `bfd49efe5525f9672ed34cb68c7d2e51c0a338acc33579fe536ae18cb437e5f0`. Its embedded parent is the v0.5 City Clearance checkpoint with SHA-256 `40273f7c760cb1a9406b0845377f461b7b8808b3bbeb6027f183f6849ebae01e`. The fit report identifies the training inputs and the execution sources used for inference. [Calibration code](../../crosswalk/city_progress_fit.py) · [Driving comparisons and retained failures](../../results/city-progress/README.md)

[Revision 3 checkpoint and execution sources](../../results/city-progress/trials/matched-candidate-v3/) retain the exact pre-grade validation artifact, SHA-256 `62db5184c31e257b99788ae9968f6e0a1cf254f2aec2333c9c0074de44242755`.
