# Train a waypoint predictor on Waymo driving logs

This offline research workflow trains an independent predictor on recorded Waymo camera, ego-history, and routing data from the Waymo Open Dataset End-to-End Driving v1.0.0. It predicts 20 future ego waypoints over five seconds. The pilot uses Waymo data exclusively. Keep records and feature caches in private runtime storage and treat checkpoints as restricted research artifacts under [Waymo's non-commercial terms](NOTICE.md).

The completed Colab pilot evaluated 371 windows from 220 held-out Waymo validation segments. History and routing alone achieved 1.780 m ADE, with 1.808 m for the three-camera model and 2.984 m for constant velocity. [Pilot results and simulator transfer checks](RESULTS.md)

## Run through the Colab CLI

From a full checkout, preview the job with `python -m waymo_training.colab_cli_run --output "A:\ADO-Private-Research\runs" --dry-run`. Remove `--dry-run` to create a dedicated T4 and perform the bounded extraction, fit, evaluation, portable export and verified download. Use your own private output directory. The default cloud job uses an authorized, short-lived read grant limited to the official training and validation object prefixes. The interactive ADC option is available with `--auth adc`. Training and bulk reads run on the remote machine.

## Run in Google Colab

1. Open the [training notebook in Colab](https://colab.research.google.com/github/yafi-s/ADO/blob/2976891ae4a0a8e3917c4f1c2be8ecb32c599a2b/waymo_training/ADO_Waymo_Colab.ipynb) and connect a **GPU** runtime. Use the Google account with access to the Waymo dataset when authorizing Cloud Storage. The notebook's `google.colab.auth.authenticate_user()` supplies credentials inside the runtime. [Waymo access and terms](https://waymo.com/open/terms/) · [Colab Cloud Storage example](https://colab.research.google.com/notebooks/io.ipynb)
2. Stage the sampled data directly from Waymo Cloud Storage into the Colab runtime. The extractor validates the TFRecord checksums and writes compact feature shards from complete examples. It records source object generations and hashes in `/content/ado-data/manifest.json`. Train after extraction finishes so a repeat starts with the same data.
3. Use a private Google Drive folder for checkpoints if the run must survive the Colab runtime ending. Keep temporary records and feature files under `/content`. Recreate the same feature set before resuming a checkpoint in a new runtime.

From the repository root **inside** an authenticated Colab runtime, the individual stages are:

```bash
python -m waymo_training.extract --auth adc --out /content/ado-data
python -m waymo_training.train --features /content/ado-data/features --output /content/ado-run --device gpu --seeds 41 73
python -m waymo_training.evaluate --features /content/ado-data/features --run /content/ado-run --device gpu
```

Training also runs evaluation. To resume, use the original feature snapshot, package versions, training options, and output directory with `--resume`. Preserve the full run folder for resuming, including its `resume/` state. Keep the resulting `config.json`, `training_curve.json`, `evaluation.json`, and checkpoint files together with the [data and model notice](NOTICE.md).

## Play a private CARLA transfer recording

From the research branch checkout, point the replay launcher to the private simulator results folder containing `report.json`. The viewer serves the recording on this computer. Press Ctrl+C in the terminal to stop it.

```powershell
python -m waymo_training.replay --results "A:\path\to\private-transfer-results" --open-browser
```

## Dataset and targets

- Use the official April 2025 release of v1.0.0. It has 2,037 training segments, 479 validation segments, and 1,505 test segments. Train and validation contain 20-second recordings. The validation set also includes rater feedback at one frame per segment and scenario tags. Test recordings are 12 seconds and withhold both future driving logs and rater feedback. Download through a Google account that has accepted the Waymo agreement. [Waymo download and version history](https://waymo.com/intl/en-ca/open/download/) · [Dataset composition](https://waymo.com/intl/en-ca/open/data/e2e/) · [Dataset access and terms](https://waymo.com/open/terms/)
- Each `E2EDFrame` can carry eight JPEG camera views at 10 Hz, per-camera intrinsics and extrinsics, a frame ID and timestamp field, and a routing command (straight, left, or right, plus an unknown enum). Past states cover `(-4 s, 0]` at 4 Hz and include positions, velocities, and accelerations. Supervised targets are `future_states.pos_x` and `pos_y` over `(0 s, 5 s]` at 4 Hz. For trajectory coordinates use the proto's rear-axle origin and axes in metres, with +x forward and +y left. LiDAR, maps, and object labels are outside the populated fields of this release's E2ED record. [Dataset fields](https://waymo.com/intl/en-ca/open/data/e2e/) · [Pinned E2EDFrame schema](https://github.com/waymo-research/waymo-open-dataset/blob/99a4cb3ff07e2fe06c2ce73da001f850f628e45a/src/waymo_open_dataset/protos/end_to_end_driving_data.proto) · [Waypoint submission schema](https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/protos/end_to_end_driving_submission.proto)
- Waymo derives the route command from the next ten seconds of recorded driving. The model receives that command as a supplied route instruction. Report results as route-conditioned forecasting. [Waymo WOD-E2E paper, §3.1.3](https://arxiv.org/html/2510.26125v1#S3.SS1.SSS3)

## Interpreting evaluation

The pilot reads the first 32 MiB from each of 112 selected training shards and 40 validation shards with the default arguments. That is 4.75 GiB before any retries. It fits on official training segments and assigns validation segments by a deterministic segment hash to pilot development or pilot holdout. Consecutive camera frames share most of their ego history and future, so segment-level assignment avoids mixing those frames across splits. The pilot holdout comes from Waymo's validation set. The prefix sample favors early records in each shard and may miss the rated moments.

Compare constant-velocity extrapolation, a history-and-intent neural predictor, and a predictor that adds contemporaneous front, front-left and front-right images. The cache stores these three views at 64 × 96 pixels and the model pools them to 32 × 48 pixels. The same-seed comparison and shuffled-image diagnostic help assess whether the predictor used the camera inputs in this sample.

Logged average displacement error (ADE) is the mean Euclidean waypoint error. Logged final displacement error (FDE) is the Euclidean error at the five-second endpoint. Both measure agreement with the observed trajectory. Waymo's challenge ranks by Rater Feedback Score (RFS), which compares predicted paths against several alternatives scored by humans at selected moments. Its tie-breaking ADE uses the highest-rated reference. Label any pilot results as logged-trajectory metrics and report the evaluated segment and frame counts. [Waymo WOD-E2E paper, §3.5](https://arxiv.org/html/2510.26125v1#S3.SS5) · [Waymo challenge rules](https://waymo.com/open/terms/) · [Official metric schema](https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/protos/end_to_end_driving_metrics.proto)

The portal includes one short, CARLA-generated [Waymo-model transfer recording](https://yafi-s.github.io/ADO/?run=waymo-research-v2-lead-seed41-start3) with synchronized controls and a [non-commercial research notice](../web/media/waymo-research-pilot/NOTICE.md). The model left its lane; the clip is evidence of this test's failure. The other transfer recordings and learned weights stay private.
