# Traffic-light transition audit

On the original High Town03 drive, CARLA changed signal 2024 from red to green at frame 1450. The rider display showed red on two heads that tick. A fresh red vote at frame 1451 had confidence 0.9408 and passed the old confirmation rule despite an intervening abstention. Retention extended that reading through frame 1454. In all, the display emitted six wrong reds, five carried from earlier readings. Its first confirmed green was at frame 1488, 3.8 seconds after the reference changed. The recording had 2,634 correct colored head-frames and 14,829 unknowns out of 17,469 mapped head-frames. [Release counts](signal-reads.json)

The audit compares the unmodified v0.3 reader and current reader on identical images. The original calibration checkpoint and its 0.85 threshold are shared. CARLA states enter only the scorer after the two readers run. The observer measures rider-display color. The learned controller has its own camera inputs. [Hashes, counts and mismatch frames](signal-transition-rescore.json)

## Archived video checks

The original v0.3 live counts came from lossless CARLA camera frames. Its recorded camera streams use JPEG, and the published videos use H.264. Decoding either can attenuate or shift the color of a small bulb. The absolute counts and first-green times below describe rescoring decoded video.

- Across all 17,469 mapped head-frames, the five original camera journals gave 1,070 correct, 22 wrong and 16,377 unknown with the published reader. The revised reader gave 882 correct, seven wrong and 16,580 unknown. Mismatches for signal 2024 fell from four to two. Both readers first confirmed green at frame 1503.
- The five public camera videos gave 549 correct, 85 wrong and 16,835 unknown with the published reader. The revised reader gave 423 correct, 63 wrong and 16,983 unknown. Mismatches for signal 2024 fell from four to zero. Both readers first confirmed green at frame 1554.

The encoded reruns preserve the tradeoff between fewer incorrect colors and more neutral heads. The archived replay always displays the camera decisions recorded during the original drive.

A separate experimental reader let a current local camera crop borrow corroboration from a nonoverlapping view of another head on the same signal, or a second camera. That restored correct readings across the drive, reaching 1,277 on MJPEG and 609 on H.264, with 20 and 77 overall errors respectively. It made four errors on signal 2024 with each encoding. The revised reader keeps the size-aware confirmation rule while lossless validation is pending.

## Lossless validation protocol

The capture tool uses an idle High Town03_Opt server. It records the original camera poses around frames 1427–1527 and times synchronized red, green, yellow and unlit cycles at two distances on a pole excluded from fitting. It saves raw RGB alongside native frame numbers, camera registration and independently read signal states. The original route is re-sensed without the original vehicle traffic. The evaluation tool replays each captured image through both readers and reports confirmed colors, errors, unknowns and time to the first agreement after each transition. No lossless revised-reader result is available yet. [Capture tool](../../tools/capture_signal_transitions.py) · [Comparison tool](../../tools/rescore_signal_rgb.py)
