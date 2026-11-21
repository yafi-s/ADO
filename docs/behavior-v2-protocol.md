# Behavior revision driving benchmark

The primary comparison uses 50 previously reserved seed–case pairs in the held-out district. Each pair runs the published v0.1 fusion policy and one frozen candidate, for 100 newly scored drives. The main endpoint is the number of clean finishes in each arm. The paired difference is the number of candidate-only clean finishes minus the number of baseline-only clean finishes. Report those two counts separately so improvements and regressions remain visible.

## Cohorts

- Development uses seeds 31 and 32 in the training district. Both policies can run all ten cases for 40 drives. A single seed, case or arm can run as a smoke test. Development results may guide fitting.
- Fresh evaluation uses seeds 2111 through 2115 in the held-out district. Each policy takes the first position for exactly 25 pairs. Model selection ends before either final stage starts. Any change prompted by inspecting final results requires a newly reserved cohort to support another prospective claim.
- Known regression uses seeds 101 through 110 in the held-out district. The candidate runs 100 drives. The historical comparison uses the published 100 v0.1 fusion summaries. Those records contain 70 clean finishes, 72 route completions, 17 timeouts, nine lane departures and two collision outcomes. Identify the baseline as historical when presenting this comparison.

The ten cases are pedestrian, pedestrian-delayed, lead, braking, occluded, signal, left, turn, clear and trip. The delayed pedestrian crosses with the existing one-second delay. Odd trip seeds select Market and even seeds select Station. Each drive uses the original Scenario, episode time limit, seeded weather, stock grip and three-camera/LiDAR/radar observation pipeline. Seeds can share route geometry. Publish distinct route IDs by arm and case alongside the seed counts.

## Scoring and evidence

A clean finish requires the episode to return `complete` with zero collision events, red-light violations and yield violations. Full episodes retain their original terminal outcome, including a completed route with a yield breach. Training loss, development runs and observation-only proxies have separate denominators.

Each arm reports trial counts, clean finishes, terminal outcome counts, contacts, red-light and yield breaches, and distinct route IDs. Pairing requires matching seed and case plus recorded route, weather, spawn, district, delay, traffic counts, sensor configuration, producer and requested render profile. The report lists every setup mismatch and keeps both outcomes in arm totals. Interpret paired counts using their own denominator. Additional drives on these seeds reveal performance only on this CARLA suite. They do not establish performance on new cities or real roads.

## Running the comparison

Run from the checkout with the CARLA 1.0 project Python environment. The normal entry point loads a single candidate artifact with `BehaviorModel.load`. For a separate adapter and configuration use `--adapter crosswalk.module:function --config path/to/config.json`. The adapter receives the frozen checkpoint path and parsed configuration. The frozen v0.1 model uses the existing baseline loader.

```powershell
python -m crosswalk.revision_benchmark --stage dev --candidate models/behavior-v2.npz --seeds 31 --cases pedestrian --arms candidate
python -m crosswalk.revision_benchmark --stage fresh --candidate models/behavior-v2.npz
python -m crosswalk.revision_benchmark --stage regression --candidate models/behavior-v2.npz
```

Add `--attach --port 2468` to use an idle offscreen CARLA 1.0 Town01_Opt session. Use the release Low-quality server launch settings when attaching. CARLA exposes the requested camera attributes and active world settings through the client. The report does not independently confirm the graphics driver applied DLSS.

The default output is `out/behavior-v2`. Under `final`, both stages share one manifest with exact checkpoint, configuration and source hashes. Its `inputs` directory preserves their contents. The manifest also records the OpenDRIVE digest, simulator versions, synchronous physics settings, render profile, package versions and trial schedules. Development runs use separate directories for each input digest. Each stage maintains its own `report.json`, per-case `summary.json`, compressed frame trace and checksum marker.

Resume by repeating the same command and output directory. Each completed case, including a failed drive, is reused after checksum verification. An interrupted case without a completion marker runs again. A simulator or recording fault interrupts the batch and stays outside the completed count. If input files change during scoring the affected drive cannot be checkpointed. The report identifies partial cohorts by completed and expected drive counts. A completed cohort with setup mismatches keeps those pairs flagged.
