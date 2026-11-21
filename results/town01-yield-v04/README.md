# Town01 interaction correction

The optional `town01-yield-v04` policy finished **16/16** targeted CARLA Town01_Opt drives with zero collisions, red-light violations, or yield breaches. Ten archived v0.2 setups matched on route geometry, seed, weather, spawn, crossing delay, traffic, producer, and rendered sensor profile. These are selected regression and challenge drives. The published v0.2 results remain **97/100 and 47/50**.

| Frozen cohort | Clean | What was exercised |
| --- | --- | --- |
| Prior failures | 6/6 | All five unprotected-left yield breaches, plus delayed pedestrian seed 104. |
| Previously passing controls | 4/4 | Opposing traffic, delayed pedestrian, ordinary turn, clear road. |
| New seed/case combinations | 6/6 | Two left turns, two delayed pedestrians, one ordinary turn and one clear road on four distinct layouts, seeds 4901 and 4902. Two of the four layout IDs already occurred in the v0.2 study. |

## What the traces show

All five v0.2 yield breaches occurred at 1.37â€“1.55 m/s, 0.1â€“2.5 seconds before opposing traffic cleared. Seed 105 advanced across the line at 1.55 m/s while applying throttle. Seed 104's delayed pedestrian began crossing at 6.6 seconds. At 7.4 seconds the ego still traveled at 5.11 m/s with 3.55 m of scored clearance and a throttle command. The old driver briefly braked and released several times before contact at 8.5 seconds.

Twelve native instructor episodes supplied the fitted stopping envelope, including four left turns and eight pedestrian encounters. Stationary teacher observations yielded a 4.93 m stand-off from the mapped junction entrance and a 2.64 m stand-off from a measured crossing obstacle. Approach braking samples yielded 2.48 m/sÂ². The policy combines those fitted quantities with causal camera/LiDAR/radar features, signed route curvature, Doppler persistence, and short memory for intermittent in-path returns. The archived v0.2 traces contain actions and scoring, without raw detections. The new traces also record both target speeds and the cue that changed the command.

Both policies completed the seed-103 pedestrian control while its final scripted phase still read “leaving lane.” The published clean criterion requires completion and zero recorded violations or contacts. The revised driver retained 2.59 m minimum scored clearance from the delayed pedestrian in seed 104. It was also slower on passing controls. The clear road took **19.6 s versus 13.5 s** with v0.2. The roadside-return cap reduced the target on 118 of 196 frames in that drive. The ordinary turn took 17.1 s versus 17.0 s, the previously passing delayed pedestrian took 25.5 s versus 19.5 s, and the previously passing unprotected turn took 25.2 s versus 23.1 s. The Town03 urban driver has a separate policy.

## Reproduction

- Fit from the native archived demonstrations with `python -m crosswalk.town01_interaction`.
- Drive an individual encounter with `python -m ado drive --policy town01-yield-v04 --scenario pedestrian --seed 104 --crossing-delay 1`.
- Run a new set with `python -m crosswalk.town01_interaction_study --output out/town01-repeat`.
- Verify the recorded drives and historical pairings with `python -m crosswalk.town01_interaction_audit`.

Run the commands in the CARLA-enabled Python environment. The [manifest](manifest.json) records the frozen schedule, model and source hashes. [Study source](study-source.zip) preserves the exact 32 Python files used to identify the run. For byte-identical source inputs, unpack those files into a separate copy of this checkout before repeating the study. The default working-tree launcher supports an explicit fresh output path. Each drive directory contains its frame trace, scored summary, and checksums. The [audit](audit.json) rechecks them and compares the ten older cases with the [published v0.2 evidence](../behavior-v2/closed-loop-evidence.zip).
