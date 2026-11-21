# Tactical yield and restart

The earlier occupied-lane physics trial first yielded at 0.3 s. The ego then applied throttle on 33 frames before committing at 8.7 s. Three accelerations ended in renewed braking at 0.516, 0.830 and 1.139 m/s. The published six-camera replay had 36 such throttle frames and four renewed brake applications, the fastest at 0.998 m/s. In each episode, a missing lead return cleared the longitudinal history. The first three reacquisition scans supplied the classifier's fallback lead speed of 30 m/s. Both clear-lane trials committed at 0.5 s. The 86-frame no-lead trial stayed in its lane.

The v0.4 PACE runtime keeps the last measured source surface and its robust world velocity through five missed scans (0.5 s). A return inconsistent with the projected surface starts a new velocity estimate. After an established wait, the driver holds for three consecutive pass decisions and a current lead measurement. A late adjacent obstruction can cancel the pass while the ego is still centered. If the source lead accelerates away, the fitted tree can select ordinary following. If it leaves the lane, source pavement observed beyond its former surface can end the wait after the prediction expires.

During the wait the requested speed is zero. The existing PACE actuator selects throttle or brake, including its learned standstill response. The normal rise limit also governs release from zero. The checkpoint, gap classifier and earlier city cruise runtime behavior are preserved. The native validator uses reference actor footprints only after the ego has issued its control.

## High-quality Town03 closed-loop check

All three CARLA 1.0 trials passed with zero collisions and every footprint inside the legal lanes. Each 960 × 540 chase recording contains exactly one exposure for each 0.1 s telemetry row. The three runs share the same PACE checkpoint and frozen tactical tree. Their start geometry and NPC commands match the earlier physics tests.

- **Occupied adjacent lane.** Ego yielded from 0.3 through the commitment at 7.8 s. It applied zero throttle during those 75 frames, including lead dropouts of one and two scans. The fitted controller selected brake values from 0.6961 to 0.9507. Speed fell from 0.575 to 0.005 m/s in 0.2 s and stayed at or below 0.005 m/s until release. The adjacent car had opened a 30.427 m bumper gap at commitment. Ego cleared the moving lead at 17.8 s with 1.537 m minimum lateral separation. [Occupied-lane recording](paired-kiu_oqoh/occupied.mp4)
- **Clear adjacent lane.** Ego selected the pass at 0.5 s without yielding, then cleared the moving lead at 6.8 s with 1.512 m minimum lateral separation. [Clear-lane recording](paired-kiu_oqoh/clear.mp4)
- **No lead.** Ego drove 45.801 m in its source lane. The maximum centerline displacement was 0.001 m, with zero yield or pass commitments. [No-lead recording](paired-kiu_oqoh/no-lead.mp4)

The [native manifest](paired-kiu_oqoh/summary.json) records all three scores and the runtime source and checkpoint hashes. The older trials used the previous city cruise model. Changes in absolute passing time between versions therefore include longitudinal-model differences. The fixed-trajectory check below isolates the tactical decision difference.

## Archived measurement checks

Reapplying the revised lead estimate and the frozen gap tree to the five archived native measurement streams removed every premature release. The occupied physics and filmed traces produced first pass commitments at 8.0 and 8.1 s respectively. Each clear-lane case still selected the pass at 0.5 s and the no-lead case followed its source lane for all 86 frames. These are open-loop decisions on prior trajectories. The compact traces omit raw pavement returns, which the rescore treated as unobserved.

Across 4,648 archived v0.4 observations from seed 211 and the two late-entry challenges, the runtime with its tactical cap inactive produced byte-identical requested speeds and steer/throttle/brake values to the archived Colab source. Requested speeds also matched the saved native values exactly. The isolated city-pace source delta from the fit is an optional speed cap, its per-episode reset, and its application before the existing rise limit. Source hashes and per-case details are in [the replay comparison](replay-comparison.json).
