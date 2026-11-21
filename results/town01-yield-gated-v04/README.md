# Town01 gated interaction policy

The Town01 policy in `Watch-Live.cmd` finished **16/16** frozen assessment drives and **2/2** separately tested interactive launcher trips cleanly. The gate recovered **4.5 of the 6.1 seconds** the previous correction added on the matched clear-road control. The gated drive took **15.1 seconds**, 1.6 seconds longer than v0.2. The tests used CARLA 1.0, Town01_Opt, native 10 Hz closed-loop physics, three driving cameras, LiDAR and radar.

| Frozen assessment group | Clean | Distinct layouts |
| --- | --- | --- |
| All six v0.2 failures | 6/6 | 6 |
| Previously passing controls | 4/4 | 4 |
| New seed/case combinations at 5261 and 5262 | 6/6 | 4 |

All ten archived v0.2 comparisons matched on route, spawn, traffic, weather, sensor setup and producer. All four fresh layout identifiers were absent from the published v0.2 study. They belong to the same Town01 held-out district and can reuse roads. The prior 100-drive and 50-drive studies retain their published v0.2 results of 97/100 and 47/50.

## What changed

The approach to an unprotected left still uses observed opposing Doppler, mapped turn curvature and an instructor-calibrated stopping distance. On straights the gated driver slows for a curbside LiDAR return when a front-camera person detection or the base driver's requested deceleration corroborates the risk. The fitted gate retains that signal for up to 1.2 seconds. A separate short memory for in-path returns bridges intermittent measurements.

The gate thresholds were fitted on 15 native training-district episodes, with five episodes from seed 48 reserved for gate development. Seed 48 also contributed to the older base model and stopping calibration. All policies and test seeds were fixed in the [protocol](protocol.json) before this gated closed-loop study. On the eventual seed-104 pedestrian contact from v0.2, the gated driver finished in **22.1 s** with **2.55 m minimum scored clearance**. On each of the five previously breached left turns it held at least **2.63 m** of scored clearance before the opposing traffic cleared.

| Previously clean control | v0.2 | First correction | Gated |
| --- | ---: | ---: | ---: |
| Clear road, seed 104 | 13.5 s | 19.6 s | 15.1 s |
| Ordinary turn, seed 104 | 17.0 s | 17.1 s | 17.1 s |
| Opposing traffic, seed 104 | 23.1 s | 25.2 s | 25.2 s |
| Delayed pedestrian, seed 103 | 19.5 s | 25.5 s | 20.6 s |

The clear road had 29 curb-related target reductions, down from 118 with the first correction. The matched seed-103 pedestrian drive passed with 0.31 m minimum scored clearance. At route completion its pedestrian was still in the scripted “leaving lane” phase, the same final phase recorded by v0.2. The clean criterion requires route completion and zero recorded contacts, yield breaches and red-light violations.

## Interactive launcher trips

The two additional trips ran the normal live, interactive driver and produced a final frame in the local console. Both matched their archived route and starting setup. The Windows launcher selects the gated policy for the initial Market trip and subsequent user-selected drives.

| Trip | Historical fusion | v0.2 | Gated |
| --- | --- | --- | --- |
| Station, seed 32, training district | Timed out at 31.25/247.26 m after 100.0 s | Clean, 57.9 s | Clean, 60.1 s |
| Market, seed 103, held-out district | Clean, 35.8 s | Clean, 46.1 s | Clean, 46.9 s |

The six targeted failures cover five unprotected left turns and one delayed pedestrian. The fresh challenge adds new seeds in the same four case families. The measurements establish behavior on these simulator cases. The Town03 city launcher uses its own policy.

## Evidence and reproduction

- [Frozen assessment audit](audit.json), [full source snapshot](study-source.zip), [source and input hashes](manifest.json), and [interactive trip comparison](launcher/comparison.json). Each drive directory contains its scored summary and compressed per-frame trace.
- `python -m crosswalk.town01_interaction_gate` fits the gate from the training archive. [Fitted gate](../../models/town01-yield-gated-v04/calibration.json)
- `python -m crosswalk.town01_interaction_study --gated --output out/town01-gated-repeat` repeats the frozen case schedule on a new offscreen Town01 host. Use the CARLA-enabled Python environment.
- `python -m crosswalk.town01_interaction_audit --output results/town01-yield-gated-v04` independently checks summary and trace hashes, consecutive frame IDs, scores, timing and matched setups.
- `python -m tools.verify_town01_launcher --port 2992 --output out/town01-interactive-repeat` repeats the two launcher trips on an idle Town01 host at that port. The harness records the normal interactive driving loop's output, confirms delivery of a final live frame, and closes the console after each route.
