# Town03 traffic interaction audit

The audit joins each 10 Hz perception record to the scored exposure with the same native CARLA frame and simulation timestamp. It grades the driver and the display independently using saved recordings.

## Physical clearance

The scorer projects the eight corners of each CARLA bounding box onto the ground plane. It composes the box’s local rotation and center offset with the actor transform. The ego uses its recorded six-degree pose and the measured Town03 Lincoln MKZ box. Each sample with overlapping height intervals is scored by the shortest distance between projected edges. A negative gap is the minimum planar translation needed to separate two overlapping projections. Height-disjoint views appear in their own diagnostic. These are sampled box clearances, and native collision callbacks provide the contact count. Tiny projected overlaps can reflect box envelopes, steep surfaces, or pose rounding. The 10 Hz samples do not resolve substep clearance.

Ego clearance includes every recorded registered car, van, bus, truck, walker and cyclist within the original 100 m actor query. Roles distinguish staged traffic from background traffic. The participant-pair sweep covers co-observed actors within 12 m of one another and does not cover actors outside the 100 m ego query. Parked map vehicles use the independent CARLA environment-box export and have their own group. The parked export covers the queried map labels, so any unregistered road user or roadside geometry remains outside the measured set. Moving-only results use a consecutive 0.5 s displacement window with thresholds of 1 m/s for vehicles and 0.5 m/s for walkers.

v0.4 used the same geometric method with known blueprint-local box offsets and yaw-only road-user poses. Five archived walker blueprints lack local calibration, so those actors contribute actor-origin observations. The closest of those was the delayed walker at 5.47 m. The populated-drive capture records box offset, box rotation, and actor pitch and roll directly.

## Simultaneous front-camera evidence

Each verified match needs a fresh native front inference, a ranged detection with the same current track ID, and lidar or radar evidence. Its optical box must have at least 0.25 intersection over union with exactly one compatible projected actor box. The measured track center must be within 1 m of a pedestrian box center or 2 m of a vehicle box center. The displayed object must use that track ID, meet the same center gate, and be current at opacity 0.5 or greater. Counts are sampled exposures where these requirements hold simultaneously. The eligibility check favors clear unambiguous evidence and does not measure total image-visible recall. Each matched record retains the original image box and range evidence. Selected exposures also save the original JPEG bytes and link to the same native frame in the inspector.

The matcher uses registered dynamic actors with calibrated or recorded boxes. Map-parked cars are outside the numerator because this audit measures the newly populated traffic. The replay object memory runs with vehicle localization and recorded sensor measurements. The independent target list enters only after the display has emitted its objects. For an original JPEG, the raw camera index is the native CARLA frame minus the capture’s first scored frame, plus warmup for the chase camera. Extraction verifies the JPEG count and reproduces the original saved poster.

Each representative image also has a one-second identity check. It follows the exact raw track ID and displayed object ID across eleven samples centered on the exposure, limited to the captured interval. The position check compares the displayed center against the native box center. For consecutive qualifying samples, the reported displacement residual subtracts the actor’s actual world displacement from the displayed object’s world displacement. The residual measures local display motion error in those clips.

## Original v0.4 High recording

The route covered 1,416.32 m in 335.4 s across 3,354 scored exposures. The native grade recorded 11 handled events, zero ego contacts and zero red-light and yield violations. All 18 registered dynamic actors appear in the audit. The closest calibrated projected clearance was 0.92 m beside the stopped bus at 209.8 s, while ego was traveling at 2.82 m/s. The pass remained below 1 m for 71 sampled exposures. The native staged minimum hazard-gap metric reported 2.235 m and measures a different quantity. [Chase frame 4310](v04-high/v04-high-chase-4310.jpg)

The closest absolute background-car gap was 1.42 m at 114.5 s. At the closest sample with the background car moving at least 1 m/s, the gap was 1.43 m at 114.7 s. No calibrated dynamic footprint overlapped the ego at matching heights. The nearest parked level vehicle had 14.36 m of projected clearance. [Framewise audit summary](v04-high.json)

The strongest measured 0.5 s braking window reduced ego speed from 3.536 to 1.305 m/s while closing on the staged lead at 135.8 s, equivalent to −4.46 m/s² on average. [Front frame 3570](v04-high/v04-high-front-3570.jpg)

At 114.4 s, the native front camera shows a white car entering the left edge and a pedestrian on the right sidewalk. This was the one fresh exposure with a verified ranged vehicle and pedestrian together. The display had the matching car, while pedestrian track 99 did not have a drawn object with the same source ID and class at that exposure. [Front frame 3356](v04-high/v04-high-front-3356.jpg) · [Screen match report](v04-high-screen.json)

## Reproduce

```powershell
$stage = 'out/journey-captures/long-city-drive-nxksaiij'
python tools/score_city_interactions.py $stage results/city-life/interactions/v04-high.json --candidates out/v04-hero-candidates.jsonl --image chase:4310 --image front:3356 --image front:3570
node tools/score_city_interaction_display.mjs $stage out/v04-hero-candidates.jsonl results/city-life/interactions/v04-high-screen.json --barriers web/media/urban-pace/barrier-review.json --cones web/media/urban-pace/cone-observations.json
python -m unittest tests.test_score_city_interactions
```
