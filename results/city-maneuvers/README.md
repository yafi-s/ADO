# Passing traffic and driving around obstructions

## What the earlier City Progress recording shows

On the 90 km/h posted section, the earlier ADO driver had its own 40 km/h city cap. Across that section it reached **38.42 km/h** and spent **7.0 of 39.4 seconds at or above 35 km/h**. The section includes a drum taper, a bend, two roadside boards and the approach to a stopped car. It also includes an open stretch where the prior driver already sustained a high-30s pace. The 90, 60 and 30 km/h boundaries are measured separately from the operating ceiling and actual speed. [Station-by-station baseline and native views](audit/baseline.md) · [Earlier full recording](https://yafi-s.github.io/ADO/?run=city-progress)

The old passing decision was connected to the mapped lane change around the bus. It did not consider passing the stopped lead on Road 67, even where the dashed markings allowed a same-direction lane change. We assessed available distance, projected footprints and actual sensor returns separately. A neighbor being farther away in the simulator registry does not establish that the driver saw enough to pass. [Map, camera and range evidence](audit/protocol.md)

## What counts as a pass

The controller gets navigation geometry, onboard sensors, ego motion and actuator feedback. A slow vehicle, a stationary lane obstruction and moving traffic in the adjacent lane are independent simulator participants. The tests require physical evidence of leaving the original lane, drawing alongside the target in the permitted lane, clearing it and returning to the route. For the moving-vehicle case, the grade also checks that the lead really moved while alongside and that ADO yielded to the passing-lane traffic. Turning on a signal, asking for a pass, or driving merely around the bus on its preplanned route cannot by itself earn either new encounter.
