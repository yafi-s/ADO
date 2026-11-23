# Matched native driving controls

Both drivers ran the same registered Town03 fixtures at seed 212 with native vehicle controls, camera observations, lidar and radar. Each case retains its 10 Hz forward film, frame-aligned control trace, native participant geometry and checkpoint identity. Road s measures distance along the surveyed OpenDRIVE road. The split screen uses elapsed time from each matched start and ends each chapter on the final candidate exposure. [Watch the rigid board and moving lead](../../../web/media/city-maneuvers-tests/comparison.mp4) · [Movie frame index](../../../web/media/city-maneuvers-tests/comparison.json)

| Case | 0.6 City Progress | 0.7 City Maneuvers |
| --- | --- | --- |
| [Three in-lane cones](candidate/cones/summary.json) | Kept source lane. Final speed 0.0 km/h at road s 62.0 m. | Passed and returned. 1.66 m closest. |
| [Rigid in-lane board](candidate/board/summary.json) | Kept source lane. Final speed 0.0 km/h at road s 62.1 m. | Passed and returned. 1.96 m closest. |
| [Moving lead and closing adjacent car](candidate/slow/summary.json) | Followed in source lane. 0 source-boundary crossing frames. | 2.7 s of measured yield. Passed and returned with 1.70 m minimum lead clearance. |
| [Sustained occupied passing lane](candidate/blocked/summary.json) | Finished in source lane. 0 unsafe crossing or commitment frames. | Finished in source lane. 0 unsafe crossing or commitment frames. |
| [No lead vehicle](candidate/no-lead/summary.json) | Finished with 0 source-boundary crossing frames. | Finished with 0 source-boundary crossing frames. |
| [Departing lead](candidate/departing/summary.json) | Finished with 0 source-boundary crossing frames. | Finished with 0 source-boundary crossing frames. |
| [Clear bend](candidate/clear-bend/summary.json) | 28.6 km/h median at road s 79–89 m. | 28.6 km/h median at road s 79–89 m. |
| [Shoulder side of the curb](candidate/inside-curb/summary.json) | 28.6 km/h median at road s 79–89 m. Minimum board clearance 2.87 m. | 28.6 km/h median at road s 79–89 m. Minimum board clearance 2.87 m. |
| [Sidewalk side of the curb](candidate/outside-curb/summary.json) | 28.6 km/h median at road s 79–89 m. Minimum board clearance 4.07 m. | 28.6 km/h median at road s 79–89 m. Minimum board clearance 4.07 m. |
| [Occluded pedestrian](candidate/person/summary.json) | Walker occupied route for 5.2 s. Minimum body clearance 3.72 m. | Walker occupied route for 5.2 s. Minimum body clearance 3.72 m. |
| [Single blocked lane reopens](candidate/reopened/summary.json) | 7.9 s longest stop before reopening. 1.80 m passing clearance. | 7.7 s longest stop before reopening. 1.80 m passing clearance. |
| [Source car closes from behind](candidate/closing-rejoin/summary.json) | Stayed in the source lane. The post-pass threat was unexposed. | 64 threatened frames. 0 early source crossings. |
| [Fast rear car during return](candidate/uncooperative-rear/summary.json) | Return threat was not exposed. No positive safety grade. | Measured closing threat. 1.507 m closest to the moving adjacent car. |

The blocked-lane test requires the ego to finish the geographic window while two slower vehicles run in parallel. A downstream sedan departs on an independent 25-second clock. In the separate reopening control, a rigid board moves continuously out of a single legal lane. The driver must hold for the physical obstruction and resume after the measured clearance appears.

The original isolated moving-lead cut failed to expose a moving leader during the adjacent-car wait. Its paired replacement starts the lead and front neighbor rolling under native tire forces. The curb placements are independently attested at their first scored camera frame. Both corrected fixture pairs were fixed before running either model arm. [Original isolated evidence](../attempts/isolated-5510f55c) · [Original curb evidence](../attempts/curb-5510f55c) · [Moving-lead and blocked-lane registration](candidate/isolated-fixture-amendment.json) · [Curb registration](candidate/curb-fixture-amendment.json)

A held-out signal-led variation delayed the rear launch by six native road metres and raised the slow lead's requested speed from 1.65 to 1.90 m/s. The ego returned after passing the cones, yielded, then overtook the moving car. It remained in the adjacent lane and timed out at 584.724 m. No collision, red-light violation or missed yield was recorded. [Variation grade and source](../qualification/timing-variation/grade.json) · [Full-course qualification](../qualification/full-market/readiness.json)

The actual six-camera High run on the original spacing also ended without its second return, at 584.550 m. The full primary-camera rehearsal and that six-camera run have distinct native records. [High incident clip and independent gap audit](../attempts/high01/README.md)

The model, tactical classifier, execution archive and per-file checksums are available for each arm. A trace outcome of `restarted` marks the planned end of a test window. Missing maneuver or threat evidence stays unconfirmed. [Baseline protocol](baseline/protocol.json) · [Candidate protocol](candidate/protocol.json)
