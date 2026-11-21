# Waymo-trained model transfer in CARLA

**Non-commercial research illustration.** This CARLA research illustration was made using a model trained on the Waymo Open Dataset provided by Waymo LLC. It is released under the Waymo Dataset License Agreement for Non-Commercial Use. Downstream use and modification are subject to that agreement, including the non-commercial restrictions in Section 4.
[Attribution](NOTICE.md) | [Agreement](WAYMO_DATASET_AGREEMENT.txt)

## Stopped lead vehicle, lane departure

Lane departure. At the only frame with an in-path gap below 12 m, the lead-vehicle gap was 11.98 m, ego speed was 5.55 m/s, and the controller commanded zero brake for the following tick. This trial did not demonstrate a safe response to the stopped lead vehicle.

The waypoint model took control at a measured 2.99 m/s following 4.8 seconds of measured rolling history. It advanced 8.22 m along the route after handoff and departed the lane at t=3.4 s. The simulator recorded zero contacts.

The model used three simulated RGB camera views, measured motion history, and a route command. A calibrated path follower translated the predicted trajectory into steering, throttle, and brake. The CARLA cameras use a different rig from Waymo's recordings. LiDAR, radar, camera detector tracks, and simulator ground truth were captured for viewing and scoring.

The two renders contain 35 frames at 10 fps. The scored timestamps span 0.0 to 3.4 seconds. The final frame remains on screen through 3.5 seconds.

[Chase footage](chase.mp4) | [Front footage](front.mp4) | [Synchronized telemetry](replay.json) | [Controller and trial report](report.json) | [SHA-256 checksums](SHA256SUMS.txt)
