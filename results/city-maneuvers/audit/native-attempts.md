# Native maneuver attempts

## Duo06 stopped short of the second return

The car cleared and rejoined after the cones. It later waited 27 samples
for a moving adjacent gap before passing the slow car, but the
second return never started. In the strongest missed window
(32.1–34.6 s), the controller accepted the gap for 26 consecutive
exposures. Independent reconstructed minimum bumper room was about
25.39 m ahead and 13.52 m behind. The controller's own estimates were
25.12 m and 12.97 m. The pass landmark stayed undefined, so its return
counter never advanced. By the end of the probe, the stopped lead ahead
had also reduced available forward room. [Gap, scans, and pose evidence](duo06-return-window.json)

## Duo07 began the second return and withdrew

Propagating the measured lead's position kept the pass landmark through
the outbound lane change. The native car started returning at frame
7244, then withdrew three times. At frame 7251, rear pavement reach was
23.66 m against a 24 m requirement. At frame 7263 the pavement gates
passed and the rearward range reference shifted by more than two metres
between consecutive samples, evidence of uncertain association. At
frame 7281 the pavement gates passed and the stopped lead ahead left
4.90 s of longitudinal closing time against a 5 s requirement.
[Raw coverage and closing evidence](duo07-return-aborts.json)

## Duo08 completed two physical returns in the shorter corridor

The car cleared and rejoined after both the cones and the moving car.
Native body clearance was 1.675 m while beside the cones and 1.702 m
while beside the moving lead. It waited through 17 observed occupied-gap
exposures and stopped for 0.3 s with the fast rear vehicle arriving. The
ego hull remained within the two surveyed driving lanes in all 187
retained 5 Hz pose samples. The probe ended as a third maneuver began
around the original lead, at about 572 m along the route.
[Native passes and sampled footprints](duo08-native-returns.json)

## Duo09 repeated the two shorter-corridor returns

The car again cleared and rejoined after both obstacles. It waited in 26
occupied-gap samples and came to a stop for 0.8 s. The native scorer
measured a 1.586 m minimum gap to the moving adjacent rear car, 1.675 m
to the cones and 1.678 m to the moving lead. All 201 retained ego pose
samples stayed inside the two surveyed lanes. The probe ended near 572 m.
[Native passes and sampled footprints](duo09-native-returns.json)

## Full rehearsal 02 failed the moving separation check

The car physically returned after both Road 67 passes. Four brief
withdrawals during the first return left the car in the adjacent lane
as a fast rear vehicle approached. The rear car briefly braked behind
the ego. As the ego crossed toward the source lane, the moving car
accelerated and passed the partly straddling body. Eight consecutive
10 Hz samples came within 1 m, with a minimum reconstructed body gap
of **0.361 m** at frame 9755 and about **6.68 m/s** of relative
longitudinal speed. The sampled footprint check found zero overlaps.
The native collision sensor reported zero contacts.

![Four overhead views reconstructed from native vehicle poses near the close pass](rehearsal02-near-miss.png)

The original Road 4 bus pass also stalled. At 821.935 m along the
route, the lane projection capped observable rear pavement at 22.05 m
while the merge required more than 24 m. The ego was stationary and the
bus was still about 42.4 m ahead by lidar. The attempt ended there.
[Actor clearance timeline](rehearsal02-physical-gap-timeline.json) ·
[Independent native rehearsal grade](full-rehearsal02.json)

## The integrated section completed both passes and the bus bypass

The revised run began near 278 m and continued to its planned stop near
907 m. The first whole-body return was at 475.035 m. The fast adjacent
vehicle passed after the ego was fully in its source lane and the closest
sample measured 1.561 m of side clearance. The next outbound and return
motions cleared the moving car and restored the whole body to its lane at
556.944 m. Both passes together recorded 103 straddling samples near
moving vehicles. Every measured body gap in those samples met 1 m.
The sequence includes 30 verified waiting samples and passes the original
bus with 0.926 m of sampled clearance.

The added maneuvers change the posted-90 section. It takes 53.3 s at
11.07 km/h median, compared with 39.4 s at 16.48 km/h median in the
original setup. The board subsection is faster during the new alternate
lane pass. The tighter bend and easing bend include the added obstruction
and the new wait for closing traffic. In the paired original-board scene,
the candidate's median falls from 6.62 to 5.32 km/h and passage takes
11.2 s versus 9.3 s. The native board clearance stays at 1.046 m.
A closing car in the adjacent lane requires a yield. The controller
holds its current-lane speed request at zero for 5.3 s while the front
lead is still about 35 m away. The paired
clear-source portion of the original 420–500 m bend stays close to
38 km/h in both runs. [Matched pace controls](../pace/comparison.json)

The candidate's brief maximum lateral acceleration estimate is **4.27
m/s²** while settling into the alternate lane at frame 13934, compared
with **2.37 m/s²** anywhere in the original posted-90 section. The direct
world-trajectory estimate for that exposure is **3.93 m/s²**.
The candidate has two half-second braking bursts above 4 m/s² while
preparing to return. The original posts one such burst lasting 0.3 s
while approaching the stopped lead. The first candidate request drops
from 39.29 to 11.20 km/h in one exposure. These comfort costs are part
of the physical maneuver demonstration.

[Independent section grade](integrated-section01.json) ·
[Speed, steering, and comfort exposures](integrated-section01-comfort.json)
