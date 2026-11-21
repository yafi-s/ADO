# Town03 normal-speed urban driver

This checkpoint extends the Town03 city driver with a 48-tree regression forest fitted on a fresh, normal-speed Town03 teaching drive. Its outputs are additional metres per second above the instructor’s 5.2 m/s interaction speed. The fit uses measured lane curvature, junction distance, LiDAR and radar clearance, camera detections, and mapped signal distance. At runtime the checkpoint contains its original low-speed city policy, including the fitted obstacle and lamp response, steering coefficients and pedal feedback. It takes the same 40 fused inputs and also observes CARLA’s current posted speed in km/h through the vehicle interface.

The controller tapers the learned increment when the original policy requests a slowdown. A stopping-distance envelope bounds the increment when approaching a junction or occupied lane. The maximum request is 39.6 km/h. The native vehicle speed-limit reading imposes a lower ceiling wherever it applies, including CARLA’s initial 30 km/h limit. Actual speed can lag changes in the speed request. Reported speeds come from vehicle velocity.

## Training

The seed 75 instructor used privileged route and encounter locations to anticipate common urban slowdowns. It used actual hazard clearance to choose its stopping speed. A complete demonstration supplied 3,141 aligned observations, including 397 instructor requests above 35 km/h. The archived lesson contains simulator observations, actions, instructor speed requests, the full scored trace and encounter summary. The instructor completed ten encounters and eleven junctions with zero contacts. This is a simulation demonstration on a known course. Its ten staged encounters cover a narrow range of traffic behavior.

The 48 trees were fitted on Google Colab CPU with one fitting worker. The fit uses the same course as the development runs. Seeds change the start perturbation and aspects of the staged traffic. The evaluation measures closed-loop behavior on this course and does not establish driving ability on unfamiliar roads. The five-camera perception view can run alongside the driver. The cruise model uses the inherited driving sensors, while the tactical pass planner evaluates its own LiDAR observations.

[Checkpoint and component hashes](fit.json) · [Native training episode archive](../../data/city-cruise/lessons.zip) · [Fitting and inference source](../../crosswalk/city_cruise.py)

## Reproduce

The fitting environment needs NumPy, scikit-learn and OpenCV. The simulator is needed to collect another demonstration or evaluate a model. The packaged checkpoint runs without scikit-learn.

```sh
python -m zipfile -e data/city-cruise/lessons.zip .
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m crosswalk.city_cruise \
  --lessons data/city-cruise/demonstration-75/lesson.npz --output out/cruise-refit
python -m crosswalk.city_drive --model out/cruise-refit/candidate.npz --seed 211 --tactical
python tools/measure_city_cruise.py out/city-drives/<attempt>
```

For a new instructor recording, use `python -m crosswalk.city_collect --seed 75 --urban-cruise --output out/new-demonstration-75`. Each autonomous drive retains native speed limits, model speed requests, measured speed, frame numbers, scene events and scoring outcomes.
