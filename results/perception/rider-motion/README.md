# Rider display motion check

The comparison plays 91 consecutive acquisitions from Town03_Opt, seed 211, at their native 10 Hz. Frame 1062 starts at 33.5 s of simulator time and frame 1152 ends at 42.5 s. The clip runs for 9.1 s. [Play the High replay comparison](high-occluded-pedestrian-and-van.mp4)

- The **left panel** shows the rider display from commit `35b0af102a9d4ebbf9e0c1619df19f0b6b88a1fc`.
- The **center panel** shows the updated rider display. [Geometry checks and source hashes](high-replay-geometry.json)
- The **right panel** shows the simulator’s front, right and rear-right camera images from the same acquisition. Frame 1063 shows the van ahead. At frame 1095 the pedestrian is in front and the van is beside the ego vehicle. At frame 1141 the van appears in the rear-right camera after the ego vehicle passes. The updated display shows one van where the former display drew overlapping vehicle shapes.

Both rider panels use the production `PerceptionScene.updateReplay`, `sample`, `placeCamera` and renderer, with the captured town and the same i-Pace GLB. The harness uses a 400 × 225 day-mode Follow view. It fixes orbit, pitch, zoom and the viewport offset for the comparison and sets camera yaw to the sampled ego yaw. Video panels use the exact acquisition poses and content at fraction 0.

The geometry probe samples every acquisition and its midpoint in order. It then compares geometry after a pause, after eleven forward and backward seeks, and when advancing by every third acquisition. It evaluates the scene directly at recorded frame fractions and compares ego and sensor acquisition pose, object identity, class, position, shortest-arc heading, scale, footprint scale and opacity.

Both rendered versions matched exactly across 182 pause samples, 22 seek samples and 62 skipped samples. A follow-up browser probe ran the same inputs through the later production modules. All 182 acquisition and midpoint samples matched the frozen clip’s geometry byte for byte. The cyclist association module returned zero rows for these acquisitions. [Verification results and module hashes](high-replay-geometry.json)

## Reproduce from the release

Use a checkout of the v0.3 repository with `web/media/surround-city/replay.json`, `town.json` and the six published MP4s. Install Playwright and Pillow for Python, and have Chrome, FFmpeg and FFprobe available. Run the geometry proof from the repository root.

```powershell
python tools/capture_rider_motion.py verify-public
```

The public replay carries `video_index`, `video_time` and `camera_frame` on each synchronized acquisition. The proof checks the first 700 rows and matches the recorded geometry digest for frames 1062–1152 at acquisition and midpoint. Use `--public-dir PATH` for a separately downloaded release directory with the same structure.

To render a comparison from the public videos, run these from the repository root. The reference panel uses `front.mp4`, `right.mp4` and `rear_right.mp4` at indices 335–425. The output goes to `.local/rider-surround-qa/motion-clip/public/comparison`.

```powershell
$surroundMedia = 'web/media/surround-city'
python tools/capture_rider_motion.py before --public-dir $surroundMedia
python tools/capture_rider_motion.py after --public-dir $surroundMedia
python tools/capture_rider_motion.py assemble --public-dir $surroundMedia
```

The public build uses the released rider source and recompresses the published camera footage. Its video checksum will differ from the archival clip. The geometry digest verifies the scene independently of video encoding.

## Archival inputs

Use the immutable `.local/rider-surround-qa/high-prefix-700/perception.jsonl` snapshot and the capture under `out/journey-captures/long-city-drive-mqt_d1ge/raw`. The snapshot matches the first 700 rows of the native capture byte for byte. The native camera streams and perception rows append once per acquisition, so frame `n` uses zero-based camera index `n − 727`. The clip uses indices 335–425 inclusive. The JSON includes SHA-256 digests of the snapshot, town, vehicle and selected camera JPEGs.

With Playwright, Pillow, Chrome, FFmpeg and FFprobe available, run these from the repository root. `ADO_CHROME_PATH` can point to a different Chrome installation. The local candidate snapshot contains the exact renderer modules used for the clip.

```powershell
$frozenRiderSources = '.local/rider-surround-qa/motion-clip/candidate-frozen'
python tools/capture_rider_motion.py before
python tools/capture_rider_motion.py after --candidate-sources $frozenRiderSources
python tools/capture_rider_motion.py assemble
python tools/capture_rider_motion.py verify-working --candidate-sources $frozenRiderSources
```

The capture uses separate headless Chrome processes at below-normal priority on two logical CPUs on Windows. It checks the WebGL renderer for SwiftShader and disables hardware video acceleration and GPU compositing. FFmpeg uses one encoding thread, and FFprobe verifies the output format, frame count and duration. Intermediate frames stay in `.local/rider-surround-qa/motion-clip`.
