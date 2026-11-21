# Traffic-light transition audit

On the original High Town03 drive, CARLA changed signal 2024 from red to green at frame 1450. The rider display showed red on two heads that tick. A fresh red vote at frame 1451 had confidence 0.9408 and passed the old confirmation rule despite an intervening abstention. Retention extended that reading through frame 1454. In all, the display emitted six wrong reds, five carried from earlier readings. Its first confirmed green was at frame 1488, 3.8 seconds after the reference changed. The recording had 2,634 correct colored head-frames and 14,829 unknowns out of 17,469 mapped head-frames. [Release counts](signal-reads.json)

The audit compares the unmodified v0.3 reader and current reader on identical images. The original calibration checkpoint and its 0.85 threshold are shared. CARLA states enter the scorer after the two readers run. The observer measures rider-display color. The learned controller has its own camera inputs.

## Complete v0.4 drive on identical live RGB

Both readers processed the same onboard RGB during the completed Town03 High drive. The paired journal covers all 3,354 native ticks from frames 2212 to 5565. We verified every revised reading and reference state against the recorded perception journal. [Paired scores and timing](signal-native-drive-v04.json) · [Complete compressed signal journal](signal-native-drive-v04.jsonl.gz)

- The published reader reported 2,570 correct, three wrong and 15,812 unknown out of 18,385 mapped head-frames. Its three mistakes were green on a red signal at pole 2025, one fresh reading followed by two retained readings.
- The revised reader reported 2,057 correct, zero wrong and 16,328 unknown. It emitted 513 fewer correct colors, a 20.0% reduction. At staged pole 2024 both readers had zero wrong colors after its red-to-green change at frame 2948. The published reader confirmed green at 3.6 seconds and the revised reader at 4.0 seconds.
- Running the published reader and writing the paired journal added about 0.73 ms per tick. The revised reader itself averaged 0.71 ms.

These figures describe the native drive. The controlled range test below still shows repeated false reds at 25 m.

## Lossless High Town03 checks

The first panel re-sensed the original Market approach poses around frames 1427–1527 and ran red, green, yellow and unlit cycles at two stationary poses by pole 2010. It used the original front camera settings in an empty scene at the recorded poses. These sequences informed the revised gate. The reference at the Market approach switched exactly at the pose for source frame 1450. The stationary poses were 10 and 22 metres before the stop waypoint, corresponding to about 45 and 57 metres from their housings. [Native sequence counts, hashes and mismatch frames](signal-native-transitions.json)

- At the Market approach the published reader recorded 99 correct, one wrong and 203 unknown on 303 mapped head-frames. The revised reader recorded 87 correct, zero wrong and 216 unknown. Its first confirmed green arrived 4.1 seconds after the reference changed, compared with 3.2 seconds for the published reader and 3.3 seconds with temporal confirmation alone.
- At about 45 m, correct readings fell from 41 to five and wrong colors from 64 to zero on 150 mapped head-frames. At about 57 m, correct readings fell from 18 to 10 and wrong colors from 11 to three on 150 mapped head-frames. The remaining wrong colors were red during a true green phase. Both distances exposed a camera classifier that could assign high probability to red even when the rendered bulb scarcely changed.
- On the 264 stationary crops withheld from the checkpoint fit, both readers preserve 49 correct colors and zero wrong colors on crops at least 15 pixels high. For smaller crops the added gate changed 54 correct and five wrong to 21 correct and two wrong. These stationary samples were inspected during gate development.

A second test scored the finalized optical rule on a different surveyed pole excluded from both the calibration corpus and initial diagnostics. Each distance includes 24 synchronized exposures and 72 mapped head-frames. Camera distances were set to the mean of the three surveyed heads. Median ranges to individual heads were 12.39, 25.18 and 50.08 m. [Native range results and optical measurements](signal-native-ranges.json)

- At 12 m the published reader produced 36 correct, 18 wrong and 18 unknown. The revised reader produced 35 correct, 16 wrong and 21 unknown. The housing occupied a median 29.29 by 11.01 unpadded pixels, equivalent to about 9.76 vertical pixels per bulb row.
- At 25 m the published reader produced 15 correct, 41 wrong and 16 unknown. The revised reader produced 15 correct, 38 wrong and 19 unknown. The housing occupied a median 12.21 by 4.22 unpadded pixels, equivalent to 4.07 pixels per bulb row. Its padded crop remained above the optical gate's height cutoff. The baseline classifier labeled red throughout much of the green, yellow and unlit phases. Confident correct-red and green-as-red crops had nearly identical median center-relative chromatic contrasts of 3.34 and 3.20 RGB levels. The remaining error comes from repeated fresh votes as well as retention.
- At 50 m the published reader produced four correct, 13 wrong and 55 unknown. With 5.81 by 1.92 unpadded pixels per housing, each bulb row spanned about 1.94 vertical pixels. The optical gate left all 72 head-frames neutral. The temporal-only reader would have emitted two correct and seven wrong.

## Archived video checks

The original v0.3 live counts came from lossless CARLA camera frames. Its recorded camera streams use JPEG, and the published videos use H.264. Decoding either can attenuate or shift the color of a small bulb. The absolute counts and first-green times below describe rescoring decoded video.

- Across all 17,469 mapped head-frames, the five original camera journals gave 1,070 correct, 22 wrong and 16,377 unknown with the published reader. The revised reader gave 768 correct, one wrong and 16,700 unknown. Mismatches for signal 2024 fell from four to zero. Both first confirmed green at frame 1503.
- The five public camera videos gave 549 correct, 85 wrong and 16,835 unknown with the published reader. The revised reader gave 341 correct, 40 wrong and 17,088 unknown. Mismatches for signal 2024 fell from four to zero. Both first confirmed green at frame 1554.

The archived replay displays the camera decisions recorded during the original drive. [Encoded counts and provenance](signal-transition-rescore.json)

A separate experimental reader combined a current local camera crop with a nonoverlapping view of another head on the same signal, or a second camera. Across the encoded drive it reached 1,277 correct on MJPEG and 609 on H.264, with 20 and 77 overall errors respectively. It made four errors on signal 2024 in each encoding. In the lossless approach its resolved-crop setting added 26 correct readings over the published reader. At 45 m it also increased false readings from 64 to 78. The selected reader uses the local optical gate and temporal confirmation.

An exploratory change also applied the contrast check when a padded crop was at most 10 pixels wide. On the already inspected 25 m recording it reduced wrong labels from 38 to four, correct labels from 15 to three, and increased unknowns from 19 to 65. The other five lossless clips were unchanged. On the checkpoint-held-out stationary images it cut correct single-crop readings from 70 to 33 and left both errors. This width setting was tested after inspecting the 25 m result and remains exploratory. [Width and coverage tradeoff](signal-width-exploration.json)

## Reproduction

The capture tool uses an idle High Town03_Opt server. Its `transitions` study records the original approach and two parked poses. The `ranges` study records the unused pole. Both save raw RGB alongside native frame numbers, camera registration and independently read signal states. The evaluation tool replays each captured image through both readers and reports confirmed colors, errors, unknowns and time to the first agreement after each transition. [Capture tool](../../tools/capture_signal_transitions.py) · [Comparison tool](../../tools/rescore_signal_rgb.py) · [Whole-drive paired journal tool](../../tools/record_signal_comparison.py)

The 257 KiB audit archive preserves all 819 exact RGB rectangles used in the six lossless clips, alongside the geometry, clock, scored states and both readers' results. The verifier recreates the original camera dimensions, restores those pixels and checks that both readers reproduce the full-frame scores and mismatch frames. The archive SHA-256 is `a95fab2387f8a9d9bff8aca7217ee78f279bf257be19d72c69766a825c7da358`. [Lossless RGB archive](signal-native-rois.npz) · [Verifier and crop extraction](../../tools/package_signal_rois.py)

Run from the Git checkout with the CARLA Python client, NumPy and OpenCV. Verification loads the released reader from the pinned v0.3.0 revision. The extraction example shows the three projected housings where both readers reported red on true green at 25 m. It prints the exact mismatches and saves a nearest-neighbor crop montage.

```powershell
python -m tools.package_signal_rois verify results/surround-city/signal-native-rois.npz
python -m tools.package_signal_rois extract results/surround-city/signal-native-rois.npz unused_pole_1995_25m 1880 signal-frame-1880.png
```

Other inspection examples are Market `original_approach` frame 94, `heldout_10m` frame 190, `heldout_22m` frame 250, `unused_pole_1995_12m` frame 1854 and `unused_pole_1995_50m` frame 1912.

To re-score the complete drive, unpack `signal-native-drive-v04.jsonl.gz` and run `python -m tools.record_signal_comparison <unpacked-journal> <report.json>`.
