# Rider display on the featured High recording

This check replays the sealed six-camera capture from driving source `a6cad56` through the public display code offline. It covers all 3,670 native frames, 118–3787. It compares each drawn rider symbol with the native body of the staged cones, the moving lead and the two adjacent cars. [Frame-by-frame display evidence](display.json)

| Native actor | Actor-frames within 35 m | Matched symbol | Current camera and lidar | Camera box confirmed on the correct body |
|---|---:|---:|---:|---:|
| Three in-lane cones (per cone) | 299 | 124 | 103 | 74 |
| Moving lead | 348 | 348 | 158 | 158 |
| Adjacent-rear car | 282 | 262 | 135 | 135 |
| Adjacent-front car | 26 | 21 | 10 | 10 |

- **Alongside the cones, frame 1101 at 98.3 s.** The right camera and lidar supported a cone symbol 0.15 m from the native cone at 6.0 m range. [Open this exposure](https://yafi-s.github.io/ADO/?run=city-maneuvers&frame=1101&camera=right)
- **Yielding to the adjacent-rear car, frame 3498 at 338.0 s.** The rear left camera and lidar supported a car symbol 0.84 m from the native body at 14.6 m range. [Open this exposure](https://yafi-s.github.io/ADO/?run=city-maneuvers&frame=3498&camera=rear_left)

A symbol matches when it has the same class, at least 0.5 memory opacity and lies within 0.65 m of a prop or 3 m of a vehicle. Nearby counts include occluded actors and symbols outside the Follow viewport, so not every nearby frame can be drawn. A current link needs a same-exposure camera rectangle and at least one lidar hit. Confirmation also requires the rectangle to overlap the projected native body more than any competing body. [Method and per-event exposures](display.json)
