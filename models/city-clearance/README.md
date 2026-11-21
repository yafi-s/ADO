# Clearance-aware city driver

This checkpoint retains the City Pace cruise forest and pedal controller. It adds a fitted stopping-distance response for passing a row of stationary roadside objects. The input contains measured LiDAR shape, spacing and pavement visibility along the navigation path. A half-second of motion-compatible returns distinguishes a persistent roadside surface from a single scan. A pedestrian detection, a body-corridor return, a tall occluder, missing pavement evidence or an unconfirmed close return restores the original cautious approach.

The new response uses native CARLA vehicle velocity and stopping distance. Four approaches at commanded speeds of 2, 4, 6 and 8 m/s trained the response. From a cue at the measured approach speed, the Lincoln coasted for 0.5 s and then applied 0.26 brake until speed fell to 0.15 m/s. The fit predicts stopping distance in metres as **0.185761v² + 0.595528v + 0.027066**, with speed in m/s. The intercept covers the largest training underprediction. It is a measured response for this dry simulation setting. [Training traces](../../results/roadworks-pace/stopping-calibration.json) · [Coefficients and fingerprints](fit.json)

Three reserved tests started 20 road metres farther along the development lane, at intermediate target speeds of 3, 5 and 7 m/s. Their predicted distances differed from the measured distances by +0.032, −0.006 and −0.039 m. The stopping trials used Town03 road 67. The urban journey, its six-marker taper and its pedestrian encounters supplied no fitting labels. The original High recording of that taper informed the feature design. [Calibration trial source](../../crosswalk/clearance_calibration.py)

During driving, the planner measures consecutive two-metre bands of visible pavement beyond the lateral returns. It reserves the original fitted 4.26 m stand-off. Tight lateral clearance reduces the distance available for the stop. It inverts the fitted curve to choose the speed supported by that distance. The new head cannot exceed its highest measured training speed. Existing body-obstacle, yield, junction, posted-limit and acceleration constraints still govern the combined command. The visibility checks, geometry tolerances and lateral-clearance discount are engineered. The stopping response, underlying city model and pedal calibration are fitted. [Runtime and fit](../../crosswalk/city_clearance.py) · [Closed-loop evidence](../../results/roadworks-pace/README.md)

To reproduce the checkpoint from the archived measurements, run the following in the project environment.

```sh
python -m crosswalk.city_clearance --calibration results/roadworks-pace/stopping-calibration.json --output out/refit-clearance/candidate.npz
```

To repeat the physical measurements in an idle High Town03 server, run the following command. An independent repeat can differ slightly in actual release speed and measured stopping distance.

```sh
python -m crosswalk.clearance_calibration --attach --port 3178 --output out/refit-clearance/native-stops.json
```
