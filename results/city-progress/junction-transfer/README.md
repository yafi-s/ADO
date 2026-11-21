# Native straight-junction transfer

The two vehicles drive physically across one another at a real, signal-controlled Town03 junction. The ego faces a frozen green light. A Mini enters from the side under throttle and steering control. Its state is available to the fixture and independent scorer. The student uses its normal cameras, lidar, radar, odometry and navigation.

## First attempt at junction 356

Junction 356 is outside the Market itinerary. The westbound ego follows connector 384 and the northbound Mini approaches from the left on connector 434. The same departure at 5.3 s applies to both checkpoints. An independent run with the Mini alone measured 7.0 s from release to conflict center. The baseline ego reached the center at 12.3 s with an empty intersection.

With the intersection clear, the v0.5 baseline crossed the center at 13.0 km/h (13.0 requested) and candidate R2 at 22.1 km/h (23.4 requested). The 80 m trips took 19.2 s and 16.4 s, respectively. The clear ego's exact timed body polygons would have overlapped the measured cross traffic for both policies.

In the occupied run the baseline stopped about 4 m short of the center and let the Mini pass. It completed with zero contacts and at least 1.1 s post-encroachment time, but minimum body clearance was 0.459 m. Candidate R2 collided with the Mini at 11.3 s. The collision sensor reported one contact with vehicle.mini.cooper. At that exposure, the requested speed was zero and the actual speed was 3.9 km/h. Both arms failed the prespecified combination of at least 1 m and 1 s clearance.

The Mini was visible in the side and forward cameras before impact. Candidate R2 withdrew the extra through-junction speed, but continued requesting 13 km/h. At 10 s the observed side-conflict distance was 3.8 m and the visibility-based proposal had reached zero, while the legacy base request remained 13.0 km/h. At 11.0 s the candidate slowed to 10.1 km/h with 0.3 m between the actual bodies. This is one geometry, signal phase, vehicle type, direction and release time.

[Clear and occupied speed/target pairs](attempt-01/paired-station-speeds.json) · [9 s left/front camera](attempt-01/candidate-crossing/cameras/0090-left-front.jpg) · [Baseline grade](attempt-01/baseline-crossing/grade.json) · [R2 grade](attempt-01/candidate-crossing/grade.json) · [Checkpoint and evaluator identities](attempt-01/protocol.json).

## Scoring and reproduction

Truth geometry is sampled at 10 Hz after the driver acts. The scorer projects the native world-space body boxes onto the two actual lane axes and checks polygon intersection and closest body separation. The collision sensor supplies the independent impact grade. Post-encroachment time subtracts two sample intervals from the time between the last occupied cross-traffic sample and the first occupied ego sample. Counterfactual overlap uses the clear ego trajectory and the occupied run's Mini polygon at identical simulated times. The collision case ends at contact.

The first attempt, its source and both frozen checkpoints are retained. Every arm has a per-frame trace, exact policy inputs, speed requests, contacts and synchronized camera samples. The instrumented R2 reproduction loads the archived production modules and checkpoint. Its sensor ZIP contains native 360-degree lidar and front radar arrays, plus high-quality JPEGs from the front, left and right at 10 Hz from 3 s until impact. Archive entries are identified by simulation step and world frame. The trace records the world transform for each timestamp.

Run the regular first pair with a free native simulator slot using `.venv-city/Scripts/python -m crosswalk.junction_progress_study --output <new-directory>`. The field study and production driver share a single native GPU slot.
## R2 reproduction and R3 development pair

The independently instrumented R2 run reproduced the collision at 11.3 s. Both runs lasted 114 exposures. The largest absolute change over matched ticks was 0.109 m in ego station and 0.035 m in Mini station. All 84 synchronized sensor packets passed archive validation. [Reproduction check](reproduction-v2-sensors/reproduction-check.json).

The R3 checkpoint stopped 8.3 m upstream of the conflict center. The car passed 4.788 m away. The first ego occupancy followed the Mini by at least 4.3 s. R3 then drove through and finished in 23.8 s. Its unimpeded trajectory would have overlapped the Mini for eight recorded exposures. With the intersection clear, R3 took 16.8 s and crossed the exact conflict center at 19.64 km/h with a 23.40 km/h applied target. The v0.5 baseline reached that same coordinate at 13.04 km/h with a 13.01 km/h target. [R3 grade](development-v3/candidate-crossing/grade.json) · [R3 exact-position speeds](development-v3/exact-station-speeds.json) · [Baseline and R2 positions](attempt-01/exact-station-speeds.json).

The first junction is a development case after its R2 failure. The subsequent transfer uses junction 1221, outside the Market itinerary. The Mini approaches from the opposite side of the ego and has a different speed and starting offset. In the additional nonintersecting control, the same Mini approaches then brakes at the cross-street entrance. Geometry, speed, grading and control variants were chosen before running that junction, and the production R3 source and checkpoint were frozen before its first native exposure.
