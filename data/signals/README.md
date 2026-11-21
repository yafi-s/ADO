# Signal lessons

`lamp.npz` contains the exact RGB classifier, calibration parameters and signed-pedal examples used in the five closed-loop development checks. Both camera-equipped policies can load the same expert. Its SHA-256 is `5d8479b212c386af7fa5df8128f468727b4a55abb1f8abd7c68fd40ac5d09d33`.

## Reproduce the fit

From the project root, use the project Python environment and a new output directory. The fit reads the archived observations and runs without a simulator.

```sh
python -m zipfile -e data/signals/lessons.zip out/signal-inputs
python -m crosswalk.lamp_study fit --study out/signal-inputs/study --lessons out/signal-inputs/lessons --output out/signal-reproduction
python -c "from pathlib import Path; import hashlib; h=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest(); assert all(h('data/signals/'+f)==h('out/signal-reproduction/'+f) for f in ('lamp.npz','fit.json'))"
```

The archive contains 80 native, lossless RGB crops, 10 static-capture records with world-space head geometry and per-frame image coordinates, and eight actual Traffic Manager approach records. Each drive record contains the four RGB measurements, speed, bumper-to-stop-line distance, and next applied teacher throttle and brake. CARLA 1.0 produced the observations in Town01_Opt. Camera settings, selected camera, label, frame number and measured distance are recorded for each stationary pair.

## Training and checks

- **Classifier fit.** Seeds 11–14 supply 16 red/green pairs at four distinct poles. Each pair switches the same physical head at a fixed camera pose. A ridge classifier learns the signed color score. Logistic calibration uses scores from withholding each fit pole in turn.
- **Pedal fit.** At each of the same four seeds, one Traffic Manager drive sees red then green, and another starts green. These eight drives supply 486 aligned controls. The local regression uses the inferred red probability, static stop-line distance and ego speed. Steering comes from the original policy.
- **Development.** Seeds 31 and 32 evaluate the first two unseen-to-fit poles. The initial whole-crop features missed three of eight red images. After that observation, the revised features measured the upper and lower lamps against the central housing. Seeds 41–44 then supplied additional paired observations. Seeds 41 and 43 repeat fit poles and are excluded from recognition accuracy. Seeds 42 and 44 supply fresh captures on the same physical poles as 32 and 31.
- **Inference.** Static lane associations and planned route identify applicable signal heads. Their projected locations select the highest-resolution fully visible crop among the three RGB cameras. The model consumes that image measurement and vehicle motion. Simulator signal state appears in training labels and development scoring only. If there is no applicable projection, the base policy supplies all controls.

`fit.json` records the split, individual probabilities, class counts and Brier score. See [closed-loop development results](../../results/signal-development/README.md) for stop and restart measurements.
