# Town03 city speed planner

The two candidate checkpoints share the frozen Town01 v2 steering and signed speed-feedback coefficients. The city fit changes longitudinal speed planning. Both candidates use a learned monotonic speed curve for measured body-path clearance and a separate learned speed request from navigation, Doppler motion and RGB lamp measurements. The vehicle takes the lower of those two learned requests. A green signal can no longer raise a speed request constrained by an approaching road user.

At driving time the input row contains the 38 causal v2 features, the RGB classifier's red probability and the mapped bumper-to-stop-line distance. An unavailable housing contributes probability 0 and distance 60 m. The approach model uses a rolling twelve-observation lamp window. The observer resets at the beginning of each episode. After one call to `predict`, `last_speed_request` exposes exactly that frame's requested speed. [Policy and fitting source](../../crosswalk/city_policy.py)

## Checkpoints

- **Pilot** — [candidate.npz](candidate.npz), 70,094 bytes, SHA-256 `9d13c4295aa26bfb990441ac87f8f16a8bdd18af23e442459a4e96001ecabc9e`. The pilot uses 7,150 previously recorded Town01 measurements plus 1,614 Town03 rows from the first part of seed 73. The instructor hit a lead vehicle after its scripted encounter had cleared. The fit excluded the last 30 frames because the instructor continued requesting speed toward the vehicle. [Fit provenance](fit.json)
- **Two-lesson city fit** — [candidate-73-74.npz](candidate-73-74.npz), 79,166 bytes, SHA-256 `e139089bf706909f6caa711dc189a8a7cc29c7c2d6a9918a6f350b604cb3025e`. It adds 1,855 Town03 observations from seed 74 starting at 680 m. This covers later encounters, including the unprotected left. A second Colab CPU fit reproduced every model byte after adding the diagnostic request field. [Fit provenance](fit-73-74.json)
- **Prior model on the omitted second half** — The frozen 73-only pilot's mean absolute speed-request error across all 1,855 second-half observations was 0.191 m/s. Its mean absolute error on the 30 instructor-stop observations reached 1.994 m/s. Twenty-three of these were yielding to the oncoming car before a left turn, for which the pilot's median request was 2.45 m/s. After fitting the second lesson the median on these same frames was 0.04 m/s. That latter measurement uses training data. [Independent pilot comparison](pilot-heldout-74.json)

The recorded teacher computed speed from simulator reference clearance and junction proximity. Fitting uses those privileged labels and encounter categories. Runtime inputs omit actor identities, authoritative signal states, event names, seed, route station and reference clearance. The first exploratory pilot's logger called the planner twice per tick, so its effective lamp history covers fewer physical frames. Compare actual closed-loop driving separately from these observation-level diagnostics.

## Reproduce on Colab CPU

Use a checkout containing the policy, both city lessons, and the existing behavior-v2 inputs. The fitting runtime uses the [recorded Colab package pins](../../data/behavior-v2/requirements-colab.txt). The [city input archive](../../data/city-v1/town03-lessons.zip) contains both original observation files, synchronized trace logs and episode summaries. [Archive and member checksums](../../data/city-v1/sources.json)

```sh
python -m pip install -r data/behavior-v2/requirements-colab.txt
python -m zipfile -e data/city-v1/town03-lessons.zip out/city-training
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m crosswalk.city_policy \
  --base models/behavior-v2/candidate.npz --town01 data/behavior-v2/lessons.zip \
  --city out/city-training/demonstration-73/lesson.npz out/city-training/demonstration-74-from680/lesson.npz \
  --out out/city-refit/candidate-73-74.npz --report out/city-refit/fit-73-74.json
python -m unittest discover -s tests -p test_city_policy.py
```

For the pilot fit, pass just the seed 73 lesson to `--city`. Each fit keeps whole episodes together for its preliminary validation, then refits on all supplied demonstrations. The output report lists per-file hashes, participating rows, excluded collision-tail frames, package versions and learned speed values.
