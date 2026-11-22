# Town03 v0.7 independent route audit

The sealed references are the High Town03 Market captures `city-life` and
`city-progress`, seed 212, 0.5 s crossing delay and 10 matched native
exposures per simulated second. Compare all candidate drives in their
actual clock order. Do not pair different runs by exposure number or assume
that traffic timed by route progress occupies the same place in both.

## Matched speed and geometry

- Use native posted limits at each exposure. Audit the entire 90 km/h
  section and the fixed geographic subsections below, allowing the first
  and last candidate samples to fall at different stations. Report entry
  and exit frame/time, actual and requested speed distributions, time
  below 10/15 and at or above 35/37 km/h, longest continuous ≥35 km/h,
  and 0.5-second acceleration and braking. Separately report any actual
  speed above the posted limit plus 1 km/h and above the 40 km/h cap.
- For the separately requested higher-speed study, retain the 40 km/h
  baseline and add achieved/requested 45/50/55/60 km/h thresholds, time
  and percent above the current posted limit, worst excess, and the
  longest continuous measured interval above 40 km/h. Report exactly
  where the faster pace was reached and what ended it. Achieved speed
  below 90 km/h has no independent pass/fail implication. Compare at the
  same stations and keep all braking, lateral and prop/actor clearances.
- Split the original 90 section by physical feature and sample station.
  The interval starts on the taper approach (358–375 m), passes the cones
  (375–398 m), recovers on the open entrance (398–420 m), reaches the
  tighter part of the sweeper (420–456 m), continues through its easing
  portion (456–500 m), leaves the bend and approaches the boards
  (500–525 m), passes the boards (525–547 m), and approaches the staged
  stopped car (547–570 m). Also audit the lead's departure through 640 m,
  the braking lead at 950–1046 m and the cyclist approach from 1175 m.
  Geographic bounds stay fixed if an attempt stops early. Report coverage.
- Re-extract the navigation path, widths, road and lane IDs, markings,
  same-direction lanes, joins and native waypoints from the installed
  Town03 OpenDRIVE. Report unwrapped heading change and sampled curvature.
  Use measured body yaw and speed for the 0.5 s lateral acceleration
  estimate. Align actors to road elevation to avoid overpasses.

## Occupancy and potential passing

- Join telemetry to population reference by frame, time and station.
  Use rotated, offset native boxes for vehicles, pedestrians, cyclists,
  roadwork cones and boards, plus the static Town03 parked-box map. Measure
  planar signed footprint gap after a height filter. Record the nearest
  sample per actor type and role, plus overlaps and unscorable samples.
  Compare native collision, signal and yield scores independently. This
  measures sampled footprints and cannot exclude contact between frames.
- Keep source driving lane, source vehicle-width path and left/right
  alternative footprints distinct. Distinguish objects beside the lane,
  intrusions of part of a lane, and overlap with the ego-width swept
  path. Classify moving and parked road users from per-actor motion and
  role. Report side pedestrians separately from forward conflict, and
  inspect original front/side frames at any occlusion or abrupt shift
  cited in the analysis.
- Screen a **plausible** pass only for a sustained slower same-lane lead
  within 35 m of the ego bumper, on a same-direction adjacent driving lane
  with permission over a full leave/pass/rejoin interval and reasonable
  room before the next route branch. Use a minimum 24 m transition each
  way, 2.0 s front/rear clearance with 15/12 m floors, closing TTC of at
  least 5 s, and at least 1.0 s persistence. Treat junctions, a blocked
  alternative, solid paint, oncoming direction, adjacent moving road
  users, crossing activity and insufficient sight as separate gates. Use
  the 40 km/h cap in the timing screen. Report cases that cannot be
  established by replay. Actor truth screenings cannot certify that the
  policy had sufficient sensor evidence or that the counterfactual would
  complete safely.
- Candidate success requires actual motion in the alternative, sufficient
  longitudinal body overlap to prove passing, completion ahead with
  clearance, a legal return and no subsequent departure from the mapped
  route. Use both native control/pose and scenario state. Check genuine
  occupied lane and unsafe rear/front gaps for waiting and later release.
  Record failures even if the overall route reaches its destination.
- Report natural traffic exposure counts at 35, 50 and 65 m along with
  encounter/contact summaries. Preserve High capture and checkpoint
  evidence. These truth references are used only by the offline evaluator.
