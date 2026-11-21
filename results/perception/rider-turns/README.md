# Follow camera through recorded junctions

The two panels follow the same 59 Town03 exposures through a right turn, from frame 1319 at 59.2 s to frame 1377 at 65.0 s. The right panel is the recorded CARLA chase camera. Its lens and mounting position differ from the virtual rider camera. The clip uses the native 10 Hz cadence. [Play the side-by-side turn](right-turn.mp4)

On the featured drive's eight major turns, the old rider camera's heading trailed the vehicle by about a third of a second. Its look-ahead pivot turned immediately while its boom followed that lag, moving the car across the screen. The new pivot and boom share a bearing. The camera smooths body heading over 0.8 seconds on each side of the replay time, follows a constant turn rate without delay, and stops smoothing at a recording discontinuity.

## Camera measurements

The replay contains 3,173 synchronized exposures at 10 Hz. A major turn here means a sweep of at least 20° during a sustained turn above 8°/s, allowing brief dips. The eight windows span 34° to 84°. The two heading wraps at frames 2277 and 3179 were also checked. The rounded map heading was within 0.005° of the full-precision body yaw. The chase camera had a median mounting offset of −2.002° from body yaw. [Turn intervals, steering inputs and source digest](turns.json)

- **Visible movie.** At a modeled 60 Hz display rate, RMS camera-to-body heading error fell from 4.80° to 0.24°, the 95th percentile from 8.19° to 0.60°, and the maximum from 8.99° to 1.02°. Best-fit lag fell from 333 ms to 0 ms at the study's 16.7 ms resolution.
- **Sensor-only playback.** RMS error fell from 4.33° to 0.24° and best-fit lag from 300 ms to 0 ms. At 60 Hz, the 95th-percentile absolute angular jerk fell from 62.1 to 14.0°/s³.
- **Native exposure boundaries.** On visible-movie playback, peak absolute angular jerk fell from 41.8 to 24.9°/s³ and RMS jerk from 6.62 to 5.69°/s³. Jerk is the third time difference of the camera’s horizontal bearing. These are playback calculations using recorded telemetry and each camera algorithm. Actual browser refresh rates depend on hardware and load.

## Registration and seeking

A software WebGL check exercised 427 exposures across those turns and wraps, including 280 with visible tracks. At each exposure the camera panned through 75% of the interval while the ego pose, acquisition transform, return buffers, drawn track poses, preview buffer and route station stayed fixed. Maximum screen-center drift of the ego origin was below 10⁻¹³ in normalized screen coordinates. Native ego and acquisition transforms matched the map pose to floating-point precision. All 48 backward seeks reproduced the original camera transform. The check also exercised Overview and a reduced-motion transition to live telemetry. [Renderer results and clip checksum](render-check.json)

While a movie plays, only the rider camera can pan between visible exposures. Pausing, seeking or stepping samples the camera at the selected exposure. With footage hidden, the rider scene continues to interpolate the ego and drawn tracks between measurements. Live Follow uses its incoming measurements.

An additional headless run opened the homepage on this drive and played its chase movie. At native frame 1376 the scene received an exposure fraction of zero and a camera fraction of 0.286. Seeking backward with the homepage slider to 60.1 s selected native frame 1328, paused, with both fractions at zero.

From the repository root, run `node tools/audit_follow_heading.mjs` for the numerical comparison and `python tools/check_follow_scene.py` to repeat the registered scene checks and render the clip. The browser harness uses the published turn windows, one headless SwiftShader browser and a single FFmpeg encoding thread.
