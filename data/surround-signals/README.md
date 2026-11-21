# Native signal color calibration

These images come from paired red, yellow, green and unlit native CARLA Town03 signal heads. The camera was held at fixed poses while the simulator changed its physical light state. The observations consist solely of registered RGB pixels. Surveyed housing geometry identifies which image patch to inspect. Seven entire poles were used for fitting and four different poles were reserved for validation.

The released fit combines **696 measured crops** from Low and High rendering. At a minimum per-image probability of 0.85, the classifier emitted 108 color estimates on 264 held-out crops. **103 were correct**, five were incorrect and 156 had insufficient evidence for a color. The runtime additionally requires two consistent observations and retains a confirmed reading for at most 0.3 seconds. These single-crop figures describe calibration performance. A driving-camera score is recorded separately on the featured run.

The low-only fit was used during early commissioning. Its checkpoint and report are retained beside the combined fit. The high-quality capture embeds its actual classifier SHA-256 in the run manifest.

To reproduce the combined fit, use Python with NumPy, OpenCV and SciPy from the repo root.

```powershell
python -m crosswalk.signal_color_fit data/surround-signals/town03-low.npz data/surround-signals/town03-high.npz --output out/refit-signals
```

[Calibration capture](../../crosswalk/signal_calibration.py) · [Held-out confusion matrix](../../models/surround-signals/evaluation.json) · [Color reader](../../crosswalk/signal_observer.py)
