# Moving-pass transfer to a second Town03 road

The frozen policy passed all three native road 67 cases on its first run. All 308 scored footprints remained inside the legal lanes. No collisions occurred. The three 960 × 540 chase videos decoded at 10 fps and matched the corresponding telemetry counts exactly. [Native scores, geometry and source hashes](paired-jo9p4yw0/summary.json)

- **Clear adjacent lane.** The driver committed at 0.5 s and completed the moving pass at 7.1 s with 1.519 m minimum lateral clearance. The lead traveled 9.145 m. [72-frame recording](paired-jo9p4yw0/clear.mp4)
- **Occupied adjacent lane.** The driver held continuously from 0.3 s until commitment at 7.8 s, including 16 scans without a direct lead return. It commanded zero throttle on all 75 held frames and remained at or below 0.013 m/s after stopping. At commitment the adjacent vehicle had opened a 31.620 m bumper gap. The ego cleared the moving lead at 16.4 s with 1.535 m minimum lateral clearance. [165-frame recording](paired-jo9p4yw0/occupied.mp4)
- **No lead.** The ego traveled 45.804 m without a yield or pass. Maximum centerline displacement rounded to 0.000 m. [71-frame recording](paired-jo9p4yw0/no-lead.mp4)

## Fixture and scope

This test moves the road 4 pass fixture to road 67, with the source in lane −2 and the adjacent target in lane −1. The native map permits the crossing of a broken lane marking at all 121 one-metre samples from OpenDRIVE station 180 to 300. Both lanes are 3.50 m wide, run in the same direction and remain outside junctions. The maximum lateral deviation from the straight parallel-lane assumption is 0.000015 m. The new corridor descends 2.36 m over 120 m. The earlier part of road 67 curves and is excluded.

The ego starts at station 210, 207.24 m from the road 4 start and facing 89.99° away from its heading. Each moving-lead case places the same Tesla 24 m ahead. In the occupied case the Mini begins 2 m ahead in the adjacent lane. The source lane direction increases in OpenDRIVE station on this road and decreases on road 4. All actor offsets are measured along the direction of travel.

The [fixture](../../../crosswalk/overtake_transfer.py) supplies native lane geometry and spawn coordinates to the [existing evaluator](../../../crosswalk/overtake_trials.py). The frozen tactical tree, PACE checkpoint, range sensors, actor commands, control loop, pass-completion definition and all score thresholds are shared. The study explicitly configures the second route segment. Automatic discovery of passing opportunities across a whole map is outside this test.

Run `python -m crosswalk.overtake_transfer` to record the three cases on one offscreen Low Town03 host. The runner closes the host after the cases and retains each frame-aligned trace and chase recording.
