# Featured surround city drive

CARLA recorded a single **317.3-second, 1,415 m** Town03 drive on seed 211 with the urban cruise checkpoint. The car completed all ten staged encounters and eleven junctions, with zero recorded contacts, red-light violations or yield violations. Each of the six videos contains 3,173 exposures. The camera frames join directly to CARLA frames 727 through 3899. [Play the drive](https://yafi-s.github.io/ADO/?run=surround-city) · [Camera and scoring manifest](../../web/media/surround-city/manifest.json)

## Measured cruising speed

On clear straights, median speed rose from 17.79 km/h in the v0.2 film to **29.07 km/h**. On the 15.3 seconds of clear road with higher posted limits, the new film measured a 32.09 km/h median, 36.46 km/h 90th percentile and **36.58 km/h peak**. The median inside turning junctions was 12.61 km/h. The log recorded zero seconds above the native posted limit plus 1 km/h. The two Low-rendering trials measured 35.01 and 34.85 km/h medians on permissive clear straights. The Market route was also used to collect city-specific training examples. [Featured speeds](speeds.json) · [Definitions and two Low trials](../city-cruise/README.md)

## What the sensors and camera saw

The car stopped behind the actual red-light queue and crossed on green. Across 17,469 mapped signal-head exposures within 60 m, the rider pipeline emitted 2,640 red or green readings. Of these, **2,634 agreed** with the independent CARLA light state. Six still indicated red after the signal turned green. The remaining 14,829 mapped-head exposures were neutral, including hidden and back-facing heads. Repeated frames and neighboring heads are correlated. [Full signal counts](signal-reads.json) · [Calibration and confusion matrix](../../data/surround-signals/README.md)

The fitted pass planner measured the source and passing lanes before committing to the bus pass. The recorded minimum alongside-bus clearance was 0.928 m. The separate moving-vehicle replays use the same fitted tactical choice and show what happens when a car also occupies the passing lane. [Bus approach](https://yafi-s.github.io/ADO/?run=surround-city&at=179) · [Open-lane pass](https://yafi-s.github.io/ADO/?run=overtake-clear) · [Yield before passing](https://yafi-s.github.io/ADO/?run=overtake-yield)

The rider visualization uses sensor observations. Independent actor geometry is available for scoring and frame inspection. A projected 2D detection can precede a supported depth estimate. [Inspect the overtake frame by frame](https://yafi-s.github.io/ADO/inspect.html?run=overtake-clear&frame=7849&camera=chase) · [Perception audit, misses and false classifications](../perception/README.md) · [Frame-matched motion comparison](../perception/rider-motion/README.md)
