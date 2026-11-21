# Town03 city pace model

The model combines a learned estimate of available cruise speed with a pedal calibration fitted to measured CARLA vehicle motion. It uses the original 40 fused navigation, RGB, LiDAR and radar features. CARLA's current posted speed separately caps the request, up to 39.6 km/h. Watch-City and the city-drive command select this checkpoint by default. The deployment needs NumPy and the existing CARLA driving dependencies.

The free-speed fit uses 48 randomized regression trees on the seed-75 instructor drive. Each leaf estimates the 85th percentile of its training requests above 5.2 m/s. Where the same observations accompanied both cruising and an instructor anticipating a scripted event, this estimates the available cruising speed. The existing fitted obstacle, signal and yield responses retain priority. Detected corridor surfaces keep the original speed envelope. A nearby side return also limits approach speed according to the lower 15th percentile of instructor speeds during moving curbside exposures. The 36 m no-return sentinel contributes no obstacle constraint. Junction geometry and native posted limits also bound the requested speed. The request can rise by at most 0.185 m/s on each 0.1 s observation, so a transient clear return cannot cause an immediate surge.

The actuator fit aligns each applied pedal with the following 0.1 seconds of vehicle motion. A robust regression estimates road-load compensation, acceleration effort and braking effort at seven speed knots from 3,015 moving transitions. Runtime interpolation tracks a speed error with ordinary requested acceleration between −2.35 and +1.85 m/s². These are control targets. As the measured corridor closes, the actuator calculates the braking needed to preserve the 4.26 m stand-off learned by the original city model. The original feedback blends in near standstill and for urgent braking. [Fit coefficients, dependencies and source digests](fit.json) · [Training and inference source](../../crosswalk/city_pace.py)

## Reproduce the fit

Use the Python and package versions recorded in the fit report. To reproduce the source fingerprint as well as the weights, unpack the archived Colab source in a separate checkout before running the recipe. Google Colab CPU ran the recorded fit with one worker. From the repository root, run the following in an environment with NumPy, SciPy, scikit-learn and OpenCV.

```sh
python -m zipfile -e data/city-cruise/lessons.zip .
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m crosswalk.city_pace \
  --lesson data/city-cruise/demonstration-75/lesson.npz --output out/refit-pace
```

The instructor uses privileged training supervision. The deployed policy receives sensors, navigation, vehicle speed and the native posted limit. The recorded city evaluations use the same course with different start or encounter conditions. [Closed-loop measurements and scored encounters](../../results/city-pace/README.md)

The runtime also accepts an optional measured tactical speed ceiling and a route-matched preview of lower posted limits. The preview reads same-direction roadside speed signs from the active CARLA OpenDRIVE map. The vehicle's current posted speed remains independent and is recorded each tick. The weight file retains its original fit hash. [Exact source used on Colab](colab-source.zip) · [Signed speed preview](../../crosswalk/speed_signs.py)
