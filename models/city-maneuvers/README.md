# City Maneuvers longitudinal checkpoint

The driver keeps the fitted v0.6 city controller and adds measured constraints for bends, roadside objects and faster open road. The checkpoint embeds its exact v0.6 parent, two fitted body-curvature predictors and a stopping response from native 40, 50 and 60 km/h cued-stop trials. The ordinary request ceiling is 39.6 km/h. Clear higher-posted sections can request up to 60 km/h when their sensed stopping horizon, curvature, speed signs and traffic support it. [Fit data and source hashes](fit.json) · [Stopping trials](../../results/city-maneuvers/speed/stopping-envelope.json)

## What enters a request

- The curvature fit uses the three navigation heading changes available to the driver. Training labels come from measured body yaw per metre over the following 7 m and 7–20 m of a previous City Life drive. The first 1,045 route metres fit the model. The final 349 metres were withheld in route order. The 80th-quantile fits cover 85.1% and 88.5% of their respective withheld observations. A curve and the approach to a curve have separate speed constraints.
- The camera observer projects current laser rays into a person detection. It checks foreground surfaces beneath the box, positive pavement and a range of possible positions relative to the upcoming path. Central people, unresolved depth and two-wheelers retain the prior vulnerable-user response. Well-supported off-path candidates can still cap speed at their earliest plausible station. The crossing-motion detector independently retains its ability to stop. A major left turn across opposing traffic keeps the prior validated yield approach before the oncoming car is visible.
- A low object beside the car needs repeated range observations compatible with a stationary location after accounting for ego travel. Nearby ground has to be visible behind it and on both sides of the roadway. Unknown or taller surfaces consume the available sight distance. Moving a low obstruction into the swept body removes this permission. A separate Doppler check measures the curved-path location of distant approaching returns. Nearby, sparse or intruding traffic keeps the existing opposing-traffic restraint.
- The longer-range observer requires ground in three strips across the road. It registers up to six preceding pavement sweeps against current odometry to fill gaps between laser elevation rings. Current observations must still extend beyond the registered evidence in each strip. A new roof-laser or front-radar obstruction directly shortens the horizon. A distant upright surface can be distinguished from a blind crossing only when its measured footprint is narrow, its height exceeds 2.15 m, its entire width lies more than 5.7 m from the curved path, and multiple scans verify a stationary footprint and pavement behind it. The signed approach preview uses the tested higher ceiling, so a 60 or 30 sign can constrain a fast request before the car reaches it.

The higher-speed stopping estimate reserves the original 4.26 m stand-off and a further 2.26 m fitted guard. The measured response used a 0.5 s coast followed by 0.45 brake. A real intrusion inside this envelope at higher speed requests at least that brake effort. Native telemetry records the operating regime, cruise ceiling, high-speed sight distance, binding condition and requested speed. The route planner can independently impose a maneuver speed limit. [Longitudinal controller](../../crosswalk/city_maneuvers.py) · [Longer-range measurement](../../crosswalk/city_open_road.py)

## Reproduce

Run from the repository root with the city Python environment.

    .venv-city/Scripts/python.exe -m crosswalk.city_maneuvers_fit --output out/reproduced-city-maneuvers
    .venv-city/Scripts/python.exe -m unittest discover -s tests -p test_city_maneuvers_pace.py

The recorded-input review holds the original car poses fixed. Matched native drives report achieved speeds, clearance, stops and route time separately. [Recorded-input measurements](../../results/city-maneuvers/pace/recorded-input.json) · [Independent hill and crossing controls](../../results/city-maneuvers/pace/junction-controls.json)
