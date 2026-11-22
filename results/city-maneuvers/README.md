# Passing traffic and driving around obstructions

## What the earlier City Progress recording shows

On the 90 km/h posted section, the earlier ADO driver had its own 40 km/h city cap. Across that section it reached **38.42 km/h** and spent **7.0 of 39.4 seconds at or above 35 km/h**. The section includes a drum taper, a bend, two roadside boards and the approach to a stopped car. It also includes an open stretch where the prior driver already sustained a high-30s pace. The 90, 60 and 30 km/h boundaries are measured separately from the operating ceiling and actual speed. [Station-by-station baseline and native views](audit/baseline.md) · [Earlier full recording](https://yafi-s.github.io/ADO/?run=city-progress)

The old passing decision was connected to the mapped lane change around the bus. It did not consider passing the stopped lead on Road 67, even where the dashed markings allowed a same-direction lane change. We assessed available distance, projected footprints and actual sensor returns separately. A neighbor being farther away in the simulator registry does not establish that the driver saw enough to pass. [Map, camera and range evidence](audit/protocol.md)

## Faster cruising and bends

Native stopping trials included a half-second coast before braking. Stopping from 41.43, 50.72 and 60.08 km/h took 21.87, 32.17 and 42.48 m. The held-out dry 55.51 km/h stop took 37.46 m. A separate reduced-grip 55.00 km/h stop took 38.04 m. These measurements fit a braking envelope. The closed-loop road tests separately measure what speed the driver actually reaches when it can see sufficient road. [Calibration and physical road tests](speed/README.md)

On the matched 454 m Town03 corridor, enabling higher cruising on the same checkpoint changed the straight-section median from **38.49 to 40.95 km/h**. The car spent **7.4 s above 40 km/h**, including **5.5 s continuously**, and peaked at 43.92 km/h. A requested speed above 50 was brief. No exposure on the empty arm reached 50. The next 30 km/h section was entered at 29.60 km/h. Five abrupt target reductions persisted on the straight; the earlier urban-cap arm had zero. In the matched occupied arm the car subsequently stopped 4.12 m behind the queue. The published study reports actual speed, requests, sign transition and braking separately. [Native evidence](speed/README.md)

The 40° bend on the original Market route is another distinct case. With the same geometry, seed and population in short closed-loop runs, median measured speed rose from **15.42 to 22.26 km/h**, with zero contacts or traffic violations. The two runs still each had two abrupt target drops. [Bend comparison](pace/bend-matched.json)

## What counts as a pass

The controller gets navigation geometry, onboard sensors, ego motion and actuator feedback. A slow vehicle, a stationary lane obstruction and moving traffic in the adjacent lane are independent simulator participants. The tests require physical evidence of leaving the original lane, drawing alongside the target in the permitted lane, clearing it and returning to the route. For the moving-vehicle case, the grade also checks that the lead really moved while alongside and that ADO yielded to the passing-lane traffic. Turning on a signal, asking for a pass, or driving merely around the bus on its preplanned route cannot by itself earn either new encounter.
