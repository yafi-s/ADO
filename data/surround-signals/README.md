# Native signal color calibration

These images come from paired red, yellow, green and unlit native CARLA Town03 signal heads. The camera was held at fixed poses while the simulator changed its physical light state. The observations consist solely of registered RGB pixels. Surveyed housing geometry identifies which image patch to inspect. Seven entire poles were used for fitting and four different poles were reserved for validation.

The fit combines **696 measured crops** from Low and High rendering. At a minimum per-image probability of 0.85, the classifier emitted 108 color estimates on 264 held-out crops. **103 were correct**, five were incorrect and 156 had insufficient evidence for a color. These single-crop figures describe calibration performance. The driving-camera score also includes confirmation and history. [Featured drive signal results](../../results/surround-city/README.md)

The reader requires two agreeing camera observations. A crop shorter than 13 pixels, including the four-pixel guard band, requires consecutive confident exposures and expires as soon as it supplies insufficient evidence. A color confirmed from a larger crop can persist for one 0.1-second tick with current supporting pixels or a still-resolved but ambiguous view. An opposing color, a confident unlit reading or a lost view clears retention when there is no new confirmed vote.

At a padded crop height of 14 pixels or less, the reader also requires a crop of at least 10 by 7 pixels and localized chromatic and brightness contrast. It compares the expected bulb's center with adjacent housing pixels. The calibration checkpoint remains the same. [Native checks at 12, 25 and 50 m and the coverage tradeoff](../../results/surround-city/signal-reader-transition.md)

The low-only fit was used during early commissioning. Its checkpoint and report are retained beside the combined fit. The high-quality capture embeds its actual classifier SHA-256 in the run manifest.

To reproduce the combined fit, use Python with NumPy, OpenCV and SciPy from the repo root.

```powershell
python -m crosswalk.signal_color_fit data/surround-signals/town03-low.npz data/surround-signals/town03-high.npz --output out/refit-signals
```

[Calibration capture](../../crosswalk/signal_calibration.py) · [Held-out confusion matrix](../../models/surround-signals/evaluation.json) · [Color reader](../../crosswalk/signal_observer.py)
