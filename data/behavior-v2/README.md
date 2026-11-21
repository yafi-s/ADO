# Reproducing the ADO behavior-v2 fit

The frozen controller was fitted on Google Colab CPU using 17,236 observation–action pairs from 80 CARLA 1.0 demonstrations in the Town01_Opt training district. Repeating the fit in that runtime reproduced every byte of the 254,599-byte model and its training report. The model SHA-256 is `25b7059a6a559ece65e2c8c64ddc1197d95c763f008be9bb31797d69617b1a6c`. Fit quality and closed-loop driving outcomes have different denominators. [Frozen model and fit report](../../models/behavior-v2/) · [Driving evaluation protocol](../../docs/behavior-v2-protocol.md)

## Refit on Colab

From the root of this checkout, create the upload bundle with any recent Python 3 installation. This step uses the standard library and verifies the input, policy source, and reference model checksums.

```sh
python tools/reproduce_behavior_v2.py --colab-zip out/behavior-v2-colab.zip
```

Open [fit.ipynb](fit.ipynb) in Google Colab and choose a CPU runtime. Run its three cells, upload the ZIP when prompted, and download the resulting model and report. The original runtime used Linux x86-64, Python 3.13.18, NumPy 2.4.6, SciPy 1.16.3, scikit-learn 1.6.1 and OpenCV 5.0.0.93. The notebook installs the recorded package versions. [Full package pins](requirements-colab.txt)

On the matching Python environment with this checkout, the equivalent commands are

```sh
python -m pip install -r data/behavior-v2/requirements-colab.txt
python tools/reproduce_behavior_v2.py --output out/behavior-v2-refit
```

The runner checks the source and each artifact, extracts the 32 lessons in the recorded order, fits on one CPU thread and verifies the model hash and training report before saving the output. To check the packaged inputs alone, run `python tools/reproduce_behavior_v2.py --verify-only`. [Training runner](../../tools/reproduce_behavior_v2.py) · [Provenance and per-episode checksums](provenance.json)

## Demonstrations and supervision

- The existing [training archive](../training.npz) supplies 10,086 pairs from 48 episodes on seeds 11–16. It includes front-camera detector summaries, measured range features, six route coordinates and controls. The archived sensors stored an ego-aligned range grid. The converter approximates the new range features from those measurements and downweights these rows.
- [labels.npz](labels.npz) contains the corresponding episode IDs, measured progress, training-only reference clearance and static map junction distance. Each archived trace was joined to the measured speeds with a 0.00051 m/s tolerance. All 48 planned routes were reconstructed from the original OpenDRIVE and matched their stored XY points within 3 mm. The map digest is recorded in the provenance.
- [lessons.zip](lessons.zip) contains 7,150 additional pairs from 32 train-district episodes on seeds 43, 44, 47 and 48. Each seed covers a clear street, a stopped lead vehicle, an unprotected left, an occluded pedestrian, a pedestrian crossing, a one-second delayed pedestrian, a turn and a multi-intersection trip. The 38 causal features include route-relative lidar, ego-compensated radar Doppler, RGB detector output and recent observations. Each observation is paired with the next applied Traffic Manager actuator command. The archive also stores the instructor's corresponding speed request, reference clearance and route progress for supervision. [Collection code](../../crosswalk/behavior_collect.py)
- CARLA Traffic Manager produced the demonstrated steering, throttle and brake. The project instructor supplied privileged target speeds for junctions and scripted stop lines. During fitting, actor clearance and a 0.7-second forward window supply speed targets around walking road users. The fitted model receives only the measured observation and planned navigation geometry at driving time. These data contain simulated CARLA observations. [CARLA project](https://github.com/carla-simulator/carla) · [Instructor and control alignment](../../crosswalk/drive.py)
- The longitudinal planner fits 40 randomized regression trees. A second fit maps speed error and measured speed to signed pedal using the demonstrated controls. Steering uses a 192-candidate search over three lookahead gains on recorded training route polylines. The model embeds the previously trained RGB lamp specialist from the released fusion artifact. Inference also reads its existing validated [approach-gate parameters](../signals/approach-gate.json). [Training implementation](../../crosswalk/behavior_v2.py) · [Lamp training and calibration](../signals/README.md)

## Regenerating the supplemental collection and legacy labels

To recollect the 32 supplementary training episodes on the project's installed CARLA simulator, attach to an idle CARLA 1.0 Town01_Opt server using the release rendering configuration. Each complete lesson is checkpointed separately.

```sh
python -m crosswalk.behavior_collect --attach --port 2868 --seeds 43 44 47 48 --output out/behavior-v2-lessons
```

The compact legacy labels can be rebuilt with the original archived train-demonstrator folders and Town01_Opt OpenDRIVE export. This operation uses the CARLA 1.0 map parser offline.

```sh
python -m crosswalk.behavior_v2 --legacy data/training.npz --legacy-traces /path/to/archived/collection --xodr /path/to/Town01_Opt.xodr --export-labels out/behavior-v2-training/labels.npz
```

The public fit uses the packaged, checksummed observations. Recollecting a stochastic simulator episode is a separate experiment. The inference package needs NumPy and OpenCV; SciPy and scikit-learn are fitting dependencies.
