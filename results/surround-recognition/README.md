# Road-user recognition on the archived Town03 drive

The YOLOX-S camera observer detected 20 of 24 hand-annotated pedestrians and cyclists in the 38-frame sample. NanoDet detected 11. YOLOX-S produced zero spurious pedestrian or bicycle boxes, and NanoDet produced two. Both recognized all nine annotated motor vehicles at least 40 pixels tall. YOLOX-S produced two extra motor-vehicle boxes and NanoDet produced six. The extra YOLOX-S boxes included a duplicate label on a van and a car box on a cropped cyclist.

The front observer uses a 640×640 YOLOX-S input. Two 320×180 auxiliary camera images share one input with a ten-pixel gap. Each image keeps its aspect ratio, camera identity and separate class-wise nonmaximum suppression. When replaying the same audited frames downsampled to auxiliary resolution, both packing positions found 18 of the 24 vulnerable road users with zero spurious boxes. These measurements use front footage as a resolution and packing check. Actual side and rear viewpoints need separate evaluation.

Warm median CPU times on the shared Ryzen 7 5800X were 68.5 ms for one NanoDet inference, 138.3 ms for one YOLOX-S front inference, and 142.4 ms for a pair of views. OpenCV used two computation threads and ONNX Runtime 1.32.0 used two CPU threads. Timings include image preparation, inference and box decoding. They exclude image capture and downstream sensor fusion.

## Original misses and bicycle recognition

- At 4.6 s, the original online NanoDet reported a person at 0.477. YOLOX-S also recognized this crossing pedestrian.
- At 36.2 s, NanoDet reported only the nearby van as a bus. YOLOX-S reported both the van and the pedestrian beyond it.
- At 297.6 s, NanoDet returned no boxes. YOLOX-S reported a person on the moving bicycle at 0.813 and a sidewalk pedestrian at 0.714. The network had no qualifying bicycle box on that frame.
- At 313.6 s, NanoDet returned no boxes. YOLOX-S reported the crossing pedestrian at 0.65.

Of the six hand-audited frames containing the cyclist, one had an explicit YOLOX-S bicycle box. In a separate 5 Hz replay from 291.6 through 309.8 s, YOLOX-S produced a bicycle box in 26 of 92 frames. The first appeared at 298.6 s. Every sampled frame from 303.8 through 306.4 s had a bicycle box. The original online NanoDet produced none during that window. The window includes time after the cyclist leaves view, and these 92 frames have no individual ground-truth boxes. The count measures detection continuity. A person-only box retains its original class.

YOLOX-S missed the far pedestrian at 0.5 s, the distant cyclist at 291.6 s, a sidewalk pedestrian at 294.6 s and the cropped cyclist at 306.6 s. This single daylight drive contains repeated observations of the same four encounters. Neither the frame-level scores nor the reported precision establish performance on new drives, night scenes or crowded traffic.

## Sample and scoring

The sample contains 18 observations at 20-second intervals starting at 0.5 s, plus 20 observations of the three pedestrian encounters and the cyclist encounter, including the four diagnostic times. Each archived JPEG is copied byte-for-byte from the original front MJPEG. Time is video index divided by 10. The annotated sample includes 18 pedestrian and six cyclist observations. Incidentally visible sidewalk pedestrians are included. The supplementary motor-vehicle score covers visually identifiable car, bus or truck instances at least 40 pixels tall in the clipped frame. The sample contains nine qualifying motor vehicles. No scorer state, event times, world coordinates or labels are inputs to the network.

Boxes are matched against the hand-drawn visible extents at intersection-over-union of at least 0.30. The six cyclist annotations bound the combined rider and bike. Correctly located person and bicycle boxes can both support the same cyclist. The strict bicycle count requires the COCO bicycle class. A second motor-vehicle box on a matched vehicle counts as a false positive. Predictions use the standard 0.35 confidence cutoff for all classes. Both detectors use their own confidence scales. The same archived images were used to compare candidates and report these development results.

- [Archived camera sample](sample.mjpeg), [human-readable pixel labels](labels.json) and [per-frame raw outputs, false positives and misses](audit.json)
- [Dense cyclist window, YOLOX-S and original online NanoDet outputs](cyclist-approach.json)
- [Observer weights, upstream source and license](../../vendor/yolox/README.md)
- [Scoring and reproduction code](../../tools/audit_surround_recognition.py)

From the repository root, run:

```powershell
.\.venv-city\Scripts\python.exe -m tools.audit_surround_recognition --output out/recognition-recheck.json
```

## Other candidates

The upstream YOLOX-Tiny ONNX model at its native 416×416 resolution took 43 ms at the warm median in ONNX Runtime. Using the original project's BGR convention, it found 15 of 24 vulnerable road users with zero spurious boxes. The RGB experiment found 14 with one spurious box. The model hash was `427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7`. [YOLOX release 0.1.1rc0](https://github.com/Megvii-BaseDetection/YOLOX/releases/tag/0.1.1rc0)

A second YOLOX-S pass on the central half-width/half-height crop, combined with the full-frame results and class-wise duplicate removal at 0.50 IoU, found 22 of 24 with two spurious boxes. The implemented observer uses the single full-frame pass to conserve compute for the other four cameras. The unpaired 320×180 input evaluation found 17 of 24 with one spurious box. [Exploratory model outputs](alternatives.json)
