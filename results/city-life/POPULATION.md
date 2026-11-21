# Street activity on the Town03 Market route

On the populated drive, a moving vehicle and a moving pedestrian were within 50 m of the ego together for **165.4 seconds**, or 49.9% of eligible driving time. On the previous Market drive they were together for 11.3 seconds, or 3.4%. The new drive spent 40.0 seconds with at least two moving vehicles and two moving pedestrians within 50 m. The earlier drive had zero such frames. Both runs used Town03, seed 212, the Market destination and a 0.5-second additional pedestrian entry delay. The new run also used the fitted clearance controller. [New score](population-score.json) · [Previous score](baseline-population-score.json)

## What moved

The populated scenario spawned 22 background cars and 30 background pedestrians before the first exposure. Seventeen cars used opposing lanes, four approached on side streets and one traveled in a parallel lane. Cars drove bounded segments under native physics when the ego approached the corresponding part of the route. Each traveled between 12.8 and 141.7 m. Eighteen drove within 16 m of the ego. The earlier scenario had three background cars and five background pedestrians. [Native drive manifest](../../web/media/city-life/manifest.json) · [Previous manifest](../../web/media/urban-pace/manifest.json)

Pedestrians walked between endpoints on native sidewalks, with some pairs walking abreast. Each traveled at least 72.6 m over the run, and 29 walked at least one metre while within 35 m of the ego. Three staged crossings included a person emerging from behind a parked van. The ego stopped for each crossing. Native frame 185 (12.1 s) had two moving cars and six moving pedestrians within 50 m. At frame 399 (33.5 s), two cars and three pedestrians were moving within 35 m. [Frame-aligned actor references](population-reference.jsonl.gz)

## How to interpret the counts

The score measures the planar displacement of a registered actor's native position over 0.5 seconds of uninterrupted 10 Hz observations. The velocity thresholds are 1.0 m/s for vehicles and 0.5 m/s for pedestrians. The first five frames are excluded, leaving 3,315 eligible frames in the new recording and 3,349 in the previous one. The vehicle category includes cars, vans and buses. The cyclist has its own encounter. Actors can be occluded, traveling in the opposite direction or behind the ego. Camera and rider-display visibility require a separate match to the sensor data. [Measurement protocol](PROTOCOL.md) · [Scoring implementation](../../crosswalk/population_score.py)

- **Within 35 m**, both classes were moving on 1,104 frames, including 152 with two or more of each. The earlier drive had 95 and zero.
- **Within 50 m**, the counts were 1,654 and 400. The earlier drive had 113 and zero.
- **Within 65 m**, the counts were 1,885 and 856. The earlier drive had 123 and zero.
- **Within 100 m**, the counts were 2,042 and 1,165. The earlier drive had 148 and zero.

## Capture and physical outcomes

The drive covered 1,416.064 m of a 1,416.438 m route in 332.0 seconds, reaching the scorer's 0.6 m completion tolerance. It completed 11 junctions and handled all 11 encounters. The ego registered zero contacts, red-light violations or yield violations. Native collision sensors on every background car registered zero contacts with vehicle or pedestrian actors. All 22 cars stayed within the scripted corridor tolerance. All 30 background pedestrians remained present. The six cones and two roadside barriers had zero measured displacement. [Native drive manifest](../../web/media/city-life/manifest.json)

CARLA 1.0 generated the scored exposures on consecutive frames 64 through 3383. The chase observer captured eight additional warmup frames. Verification matched the presentation time of every frame in the final 1080p chase and five onboard videos to the same 3,320-frame timeline. All 51,660 reference actor samples lay within 100 m of the corresponding ego pose. [Synchronization audit](synchronization.json)

The source archive identifies the executed modules from commit `d4a10b9cc9bfa6e6f40ee2dd6b5690cf508df5c9`. An earlier attempt was stopped after its scoring reference selected some actors using the wrong origin. Its 403 complete source rows and diagnostic were retained separately. [Capture source](capture-source.zip) · [Earlier attempt](attempts/01-reference-filter.json)
