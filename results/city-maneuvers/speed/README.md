# Native Town03 higher-speed trials

CARLA 1.0 ran the Lincoln MKZ at 10 Hz on Town03_Opt. The indexed traces contain measured speed after each physics tick. The checkpoint's speed request and the next-tick pedals are recorded separately.

## Vehicle response

Seven launches measured plan-view distance along the lane from a cue to the first horizontal speed sample at or below 0.05 m/s. Native driving grades use the full velocity vector. The car coasted for 0.5 s before braking. With a 0.45 brake command, measured cue speeds of 41.43, 50.72 and 60.08 km/h required 21.87, 32.17 and 42.48 m. The corresponding launches at 0.68 throttle took 3.5, 4.4 and 5.4 s. The 55.51 km/h held-out trial started at a different station and stopped in 37.46 m. Halving the four wheel friction multipliers produced a 38.04 m stop from 55.00 km/h. All wheel settings were read back from the native vehicle.

The fitted distance in metres is `0.5v + 0.12363544v²` for speed `v` in m/s. The controller adds 2.2585 m for the largest positive training residual plus one full exposure at the highest fitting speed. The unguarded fit underestimated the independent dry and reduced-grip trials by 0.36 m and 1.54 m. Including the guard left 1.90 m and 0.72 m. The controller reserves another 4.2576 m for body clearance, yielding sight requirements of 37.31 m at 50 km/h and 49.19 m at 60 km/h. The gentler 0.26 service brake needed 45.66 m from 50.81 km/h and 60.12 m from 60.17 km/h. The single held-out speed and single reduced-grip exposure do not establish a distribution of stopping errors across vehicles, grades or wet pavement.

[Calibration and trace index](stopping-envelope.json)

## Closed-loop driving

The 285.5 m primary drive follows native road 67 lane −2. The 454.3 m comparison travels road 69 lane +3 through two green junctions onto road 67 lane +3. Native 90, 60 and 30 km/h zones apply. The primary 45–165 m comparison spans the bend. The longer route's 148–268 m comparison follows the straighter opposing carriageway. Each driver starts from rest. The normal driver sees front and side RGB, lidar, radar, ego motion, navigation, and signed speed limits. Stationary-queue cases add a native vehicle in each lane. Rotated native body footprints enter scoring after the driver has selected its controls.

The published primary reference peaked at 39.93 km/h. Median speed in the bend was 38.53 km/h, with maximum 0.5 s braking of 0.31 m/s². The first higher-speed candidate peaked at 45.37 km/h. Its longest uninterrupted period above 40 km/h was 2.6 s and it spent 0.4 s above 45. Median speed in the same bend was 38.07 km/h and maximum 0.5 s braking was 1.22 m/s². The 30 km/h sign preview began 84.13 m before native zone entry, where the vehicle measured 30.10 km/h.

On the initial longer-route clear trial, speed peaked at 45.16 km/h, with a longest period of 2.8 s over 40. Straight-section median speed was 40.13 km/h and five single-exposure requests dropped by at least 5 km/h. That implementation reached 48.09 km/h earlier on the occupied approach, measured 42.71 km/h at the first body gap below 50 m, and held a full stop for two seconds in its original lane. The measured bumper gap was 4.13 m. On the primary queue, the earlier planner entered the adjacent lane and stopped behind the second queued car. Both queue arms had zero native collision events.

- [Published primary reference](published-40/comfort.json)
- [Initial primary clear and sign trial](higher-speed/summary.json)
- [Initial primary queue, including lane-change failure](stationary-queue/summary.json)
- [Initial longer clear trial](long-clear/summary.json)
- [Initial longer queue](long-queue/summary.json)

Eight one-Hz raw laser and radar sweeps from the initial longer clear trial were truncated to 22 m and replayed through the long-range head. Every one of those otherwise higher-speed requests returned to the ordinary regime. Maximum shortened request was 39.09 km/h. All other recorded features, including original camera detections, stayed aligned to the native exposure. This test measured requests with fixed motion. The original sampled sweeps and numerical results are preserved.

[Visibility stress results](limited-sight-replay.json) · [Source sweeps](limited-sight-samples)

## Frozen Town03 comparison

With identical model bytes, sources, initial conditions and route, enabling the long-range head raised median actual speed on the fixed 120 m straight from 38.49 to 40.95 km/h. Peak speed rose from 38.66 to 43.92 km/h. The enabled driver spent 7.4 s above 40 km/h, including 5.5 s continuously, and completed the 454.34 m route in 58.8 s against 62.8 s. Both arms had zero contacts and zero exposures above the native limit plus 1 km/h. Neither arm reached 45 km/h. Worst half-second braking in the straight changed from 2.53 to 1.75 m/s². The enabled arm had five one-exposure speed-request drops of at least 5 km/h on the straight. The capped arm had zero. Three drops coincided with potential vulnerable-user image detections on this actor-free fixture. The native 30 km/h transition was previewed 87.39 m ahead and braking of at least 0.05 while following that cap began 23.36 m before entry. Actual entry speed was 29.60 km/h.

The corrected primary queue produced a full 2 s stop in the source lane with a 3.89 m longitudinal bumper gap. The same frozen driver on the longer queue reached 46.39 km/h earlier on approach, measured 44.92 km/h at the first longitudinal approach with less than 50 m of measured footprint clearance and 42.70 km/h at the first gap below 35 m. It stopped 4.12 m behind the native lead and stayed in the source lane. The nine final recorded higher-speed proposals all withdrew when replayed with the 22 m sensor cutoff.

[Compact scorecard with frame and model hashes](comparison.json) · [Final longer clear run](final-long-clear/summary.json) · [Matched cap-40 run](final-long-matched-40/summary.json) · [Original queue recheck](final-primary-queue/summary.json) · [Longer queue](final-long-queue/summary.json) · [Final limited-visibility replay](final-limited-sight.json)

The release-source queue reruns used commit `9bd6dafa0348d37e163a3acf8de553f642bebf45`. Both held a two-second stop without changing lane or contacting an actor. The primary finished 3.856 m behind its lead. The longer arm reached 50.37 km/h, with **0.2 s above 50** and 3.5 s above 45. Speed measured 46.58 km/h at the first reference-body gap below 50 m and 43.63 km/h at the first gap below 35 m. Final clearance was 4.090 m. The complete film has all 433 consecutive source exposures, including approach and standstill. These queue results supplement the separate clear-road comparison.

[Native front film](../../../web/media/city-maneuvers-tests/boulevard.mp4) · [Every native frame with speed, target and posted limit](../../../web/media/city-maneuvers-tests/boulevard.json) · [Release-source verification](qualification-source.json)
