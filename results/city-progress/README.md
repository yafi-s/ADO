# City driving pace

## Where the earlier driver slowed

ADO spent 202.0 of the original 332.0 simulated seconds below 15 km/h on the populated Market route. That includes appropriate stops. The stronger evidence of unnecessary slowing was the same median request of **13.0 km/h at every one of 11 junctions**, despite different geometry and traffic. The training instructor used a fixed low target near junctions. Across the six whose headings changed by less than four degrees, the car drove at a pooled median of **13.12 km/h** and spent all **55.0 seconds** below 15 km/h. [Native speeds, timing and clearances](audit/README.md) · [Original recording](https://yafi-s.github.io/ADO/?run=city-life)

The mapped bus event starts before the vehicle draws alongside. The original car spent 7.7 seconds beside the physical bus, from 858 to 872 m along the route. Its median speed was 6.13 km/h and its minimum body clearance was 0.923 m. The bus hid the sidewalk near its nose. The original car also remained slow after that occlusion cleared. On other stretches it already reached nearly 40 km/h, so raising the overall speed cap would not address either cause. [Physical bus location](audit/BUS-LOCATION.md) · [Inspect the original pass](https://yafi-s.github.io/ADO/inspect.html?run=city-life&frame=2064&camera=front)

## Testing the change

The comparison uses the same populated Town03 route, seed, pedestrian delay, rendering profile and native sensor clock. It includes all 57 consecutive 25 m sections, every junction, and the first five seconds after each independently scored release. A second set of trials compares an open approach and a conflicting road user on the same road. Since the car can arrive earlier after the update, the moving traffic may occupy different positions. Each recording retains its own camera and controls. [Protocol fixed before testing](audit/PROTOCOL.md)

Two early candidates exposed why a general speed increase was insufficient. One briefly applied throttle while adjacent traffic still blocked a pass. Another recognized perpendicular traffic and withdrew its higher request, but fell back to the old junction target too late to stop. It hit a real simulated Mini at an intersection outside the Market route. The earlier City Life policy avoided contact there with only 0.459 m of body clearance, also below the prescribed one-metre test margin. The unsuccessful attempts and their inputs remain available. [First crossing test](junction-transfer/attempt-01/) · [Paired native trials](trials/)
