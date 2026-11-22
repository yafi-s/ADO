# Native Market route audit

The original High City Progress run took 298.7 seconds to travel the
1.42 km Town03 Market route. Its one posted 90 km/h section took
39.4 seconds and peaked at 38.42 km/h. The audit identifies each
curve, roadside prop and in-path vehicle by route position and
synchronized camera frame. [Original drive and its physical context](baseline.md)

## Passing and waiting

On Road 67 the source lane is −2 and the legal same-direction passing
lane is −1. Both surveyed driving lanes share a broken divider.
The audit checks the entire ego footprint throughout each move, the
ordering of the actual vehicle bodies while alongside, and the
settled return to lane −2. Fixed props have a separate clearance
measurement. Moving vehicles must leave at least 1 m of sampled body
clearance while the ego straddles the divider.

The focused uncooperative-rear trial used the archived moving rear car,
release driving source `9bd6dafa`, and evaluator `5510f55c`. At its closest exposure
the rear car was traveling 11.42 m/s. The ego was stationary and its
whole body was in lane −2. The independent body gap was 1.509 m,
compared with the native grader's 1.507 m. All 45 observed moving-vehicle
straddle comparisons met the 1 m requirement. Earlier in that test the
rear car was still about 31 m back as the last corners of the ego crossed
into the source lane. The exact native corner projection first places
the entire body in lane −2 at frame 18128.
[Pinned focused trial and frame-level score](uncooperative-pinned-v07.json)

The integrated native sector completed both required passes and the
original bus bypass. It met the moving clearance threshold in each of
105 exposed straddle samples. While yielding in its original lane the
car's median achieved speed was 6.27 km/h and median requested speed
was 5.79 km/h. While already in the passing lane and preparing a
return, its median achieved speed was 29.88 km/h and median request
was 10.8 km/h. The latter group includes deceleration before the return.
The sector records zero samples above 40 km/h of either actual speed
or request during divider straddles or confirmed waits.
[Integrated sector evidence](integrated-section01.json)

## Pace and comfort

On the changed, populated sector the added in-path cones and moving
traffic lengthened passage through the 90 km/h zone. It took 53.3 seconds
at 11.07 km/h median. The original took 39.4 seconds at 16.48 km/h median.
The clear-source control at the existing J1932 bend isolates the pace
change. Median measured speed increased from 15.42 to 22.26 km/h and
the matched interval took 5.8 seconds instead of 6.8. The original
board case took 11.2 seconds instead of 9.3 because the controller issued
an early zero-speed request while adjacent traffic was closing. Both
drivers maintained the same measured board clearance in that case.
[Matched native controls](../pace/comparison.json)

The integrated sector reached a 4.27 m/s² half-second kinematic lateral
estimate as the ego settled into the passing lane at the first bend.
The displacement-based estimate from the same native exposures was
3.93 m/s². The candidate also had two bursts above 4 m/s² of half-second
braking while preparing to return to the source lane. The original
recorded one such burst while approaching its stopped lead. The quieter
matched J1932 runs reached 1.32 and 1.57 m/s² lateral, respectively.
[Timing, steering and actual velocity](integrated-section01-comfort.json)

The isolated higher-speed experiments use different traffic and report
measured speed alongside the active native limit. On the matched longer
clear route, enabling the higher-speed head raised straight-section
median speed from 38.49 to 40.95 km/h. It reached 43.92 km/h and spent
7.4 seconds above 40. In the separate release-source queue trial it
briefly reached 50.37 km/h and stopped 4.09 m behind its lead in its own
lane. [Qualification and stopping evidence](../speed/README.md)

## Evidence limits

The recorded body and actor positions enter only the offline scorer.
The driver uses onboard measurements, navigation, ego motion and
measured actuator feedback. The population differs between the
original and revised Market setups. Route differences can reflect the
new maneuvers and different traffic arrival times. The native
evidence covers simulated Town03 and the named test controls.
The full clearance check samples every 0.1 seconds. The earlier
rehearsal's 0.36 m moving near miss, failed return attempts and incomplete
focused cases remain available for comparison.
[Audit protocol](protocol.md) · [Development and qualification attempts](native-attempts.md)
