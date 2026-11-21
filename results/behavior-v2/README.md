# Behavior revision closed-loop results

Both cohorts used the same frozen speed planner, steering model and signal expert. The simulator ran CARLA 1.0 on Town01_Opt with synchronized front, left and right RGB, LiDAR and radar. A clean finish requires route completion and zero recorded collisions, yield breaches and red-light violations.

## Previously reserved seeds

Fifty seed–case pairs generated 100 new drives, with each policy running first on 25 pairs.

| Policy | Clean / drives | Completed | Collisions | Departures | Timeouts | Red outcomes | Yield breaches | Layouts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | 32 / 50 | 34 | 4 | 1 | 11 | 0 | 5 | 24 |
| Candidate | 47 / 50 | 50 | 0 | 0 | 0 | 0 | 3 | 24 |

Among 50 pairs with matching recorded setups, both finished cleanly on 32, only the candidate finished cleanly on 15, only the baseline finished cleanly on 0, and neither finished cleanly on 3. 0 pairs had differing recorded setups.

## Known regression seeds

The candidate drove the 100 published seed–case combinations. The baseline figures come from the v0.1 release.

| Policy | Clean / drives | Completed | Collisions | Departures | Timeouts | Red outcomes | Yield breaches | Layouts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | 70 / 100 | 72 | 2 | 9 | 17 | 0 | 7 | 39 |
| Candidate | 97 / 100 | 99 | 1 | 0 | 0 | 0 | 2 | 39 |

Among 100 pairs with matching recorded setups, both finished cleanly on 70, only the candidate finished cleanly on 27, only the baseline finished cleanly on 0, and neither finished cleanly on 3. 0 pairs had differing recorded setups.

## Clean finishes by case

| Case | Fresh baseline / 5 | Fresh candidate / 5 | Regression baseline / 10 | Regression candidate / 10 |
| --- | --- | --- | --- | --- |
| pedestrian | 4 | 5 | 9 | 10 |
| pedestrian-delayed | 3 | 5 | 7 | 9 |
| lead | 5 | 5 | 9 | 10 |
| braking | 5 | 5 | 10 | 10 |
| occluded | 0 | 5 | 2 | 10 |
| signal | 5 | 5 | 10 | 10 |
| left | 0 | 2 | 3 | 8 |
| turn | 3 | 5 | 5 | 10 |
| clear | 5 | 5 | 9 | 10 |
| trip | 2 | 5 | 6 | 10 |

## Candidate failures

| Cohort | Seed | Case | Baseline | Candidate |
| --- | --- | --- | --- | --- |
| fresh | 2111 | left | complete with yield breach | complete with yield breach |
| fresh | 2112 | left | collision with yield breach | complete with yield breach |
| fresh | 2113 | left | collision with yield breach | complete with yield breach |
| regression | 104 | pedestrian-delayed | collision | collision |
| regression | 105 | left | lane_departure with yield breach | complete with yield breach |
| regression | 107 | left | complete with yield breach | complete with yield breach |

Each trace was checked against its saved SHA-256 digest and its summary. The checks covered frame count, consecutive simulator frames, the 10 Hz timestep, sensor profile, terminal progress, collision events and violations. The two JSON result files include outcome records for every arm and case. The study manifest records source, model and result hashes and any recorded setup differences.

The fresh seeds sample a single district on one map and can reuse road geometry. The regression cases were already known during development. The simulator outcomes apply to these scenarios.

## Downloadable evidence

The [closed-loop evidence archive](closed-loop-evidence.zip) includes the 200 new per-drive summaries and compressed frame traces, frozen code and weights, and the original v0.1 traces. All 643 archive members were read back and checked. The archive is 6,115,585 bytes, with SHA-256 `08b4288f26c4aaefca5b0522bbc093418d778d9f65a0b905f97fcd32c76ab0b5`. [Study manifest](study.json)

To recompute the arm counts and matched pairs without CARLA, extract `verify_evidence.py` from the archive and run `python verify_evidence.py closed-loop-evidence.zip`. The published pairing uses the per-case JSON checkpoints for both policies.
