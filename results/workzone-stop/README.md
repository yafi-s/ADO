# Closed-lane and curbside barrier comparison

The two Town03 runs use the same 90 m route and start from rest. The distinguishing input is the physical location of a CARLA striped street barrier 55 m ahead. In the occupied-lane case the board starts across the ego lane and moves aside between 17 and 20 simulated seconds. In the curbside case it stays outside the lane. The driver receives its normal on-board sensors and navigation. The independent scorer measures the vehicle footprint, barrier position, sustained stop time and actual speed after reopening.

The revised city-pace checkpoint completed both cases without contact, footprint overlap or lane departure. In the occupied lane it held a stop for **9.9 seconds**, with a minimum **3.19 m** bumper-to-board gap while stopped. It reached **28.72 km/h** after reopening and passed with **1.32 m** lateral clearance. The curbside trial reached **28.5 km/h** on approach and recorded zero unnecessary approach stops. Range-supported barrier tracks matched the physical board on **180** and **58** frames respectively using the trial's one-metre position gate. A successful stop tests the controller's measured obstacle response. The class-recognition holdouts separately test identification of different barrier types.

[Watch the occupied lane](https://yafi-s.github.io/ADO/media/workzone-tests/closed-then-open.mp4) · [Watch the curbside control](https://yafi-s.github.io/ADO/media/workzone-tests/curbside.mp4) · [Occupied-lane measurements](closed-then-open.json) · [Curbside measurements](curbside.json) · [Barrier recognition and controls](../barrier-recognition/README.md)

The curbside run still slowed as the car approached the board. This is a conservative response to a nearby lateral object even with an open path. [Measured speed curves](../urban-pace/speed-traces.png)

The earlier checkpoint also finished these short trials but subsequently collided with an occluded pedestrian on the long route. Its measurements are preserved in [first-fit](first-fit). These controlled trials use the same site and daytime conditions. They do not establish suitability for real vehicles.

## Wet rendering and reduced wheel grip

A second pair used native High rendering, the WetNoon weather preset and a measured wheel-friction multiplier of 0.75. The car completed both cases with zero contacts. It held for 9.9 s in the occupied lane, with a 3.18 m minimum stopped gap, and reached 28.7 km/h after reopening. The curbside case had no approach stop. The tracker matched the physical barrier on 193 and 61 frames. These settings exercise CARLA's handling and visual pipeline. [Wet blocked lane](https://yafi-s.github.io/ADO/media/workzone-tests/closed-then-open-wet.mp4) · [Wet curbside](https://yafi-s.github.io/ADO/media/workzone-tests/curbside-wet.mp4) · [Both scores and traces](wet-grip-075)
