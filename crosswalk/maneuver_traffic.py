"""Forecast longitudinal bumper separation while both lane envelopes are in use."""
import numpy as np


def rear_projection(*, rear, rear_speed, speed, gap, lead_speed, source_sight,
                    phase, transition_length, source_ahead=60., rear_seen=True, target_front=60., target_speed=0.):
    """Return time-to-vacate, worst predicted rear bumper gap and required gap.

    Ego acceleration is bounded, and a new source obstruction or an unseen end
    of the return corridor reduces the assumed returning speed. A measured rear
    vehicle may accelerate again throughout its exposure, including after it
    slows to follow the ego. No favorable reaction or braking is assumed.
    """
    speed = max(0.,speed)
    if phase=="entry":
        cruise = min(9.5,max(7.5,speed+3.))
        # Gain a car length and sufficient rear body clearance before rejoining.
        clear = max(transition_length,gap+19.)
        expected_return = min(cruise,4.5) if source_sight<clear+20. else cruise
        return_length = float(np.clip(3.5+2.7*expected_return,14.,30.))
        leave_station = None
    else:
        cruise = max(3.,speed)
        clear = 0.
        return_length = max(0.,transition_length)
        leave_station = 0.
    blind_or_occupied = source_sight < clear + .85*return_length or source_ahead < 55.
    vx, sx = speed, 0.
    vr, sr = max(0.,rear_speed), 0.
    front_v = max(0.,target_speed)
    front_station = target_front
    front_margin = 60.
    minimum = 60.
    worst_margin = 60.
    reach_time = None
    for step in range(1,161):
        t = step*.125
        if leave_station is None:
            # Progress is limited by the observed slow lead, both transitions
            # and the larger of expected and current lane-change length.
            if sx >= clear+lead_speed*t:
                leave_station = sx
            desired = cruise
        else:
            desired = min(cruise, max(2.75,(source_ahead-19.)/6.)) if source_ahead<55. else cruise
            if blind_or_occupied:
                desired = min(desired, max(3.5,.72*cruise))
        next_v = min(max(desired,vx-.25),vx+.18)
        sx += .0625*(vx+next_v)
        vx = next_v
        # Bound reacceleration by a measured passenger-car scenario envelope.
        # Preserve a higher measured rear speed instead of clipping it downward.
        next_r = min(max(13.,rear_speed), vr+.45)
        sr += .0625*(vr+next_r)
        vr = next_r
        if target_front<60. and phase=="entry":
            # Positive target-lane traffic limits usable room. Do not plan
            # to pass through that bumper or rely on its speed continuing.
            next_front = max(0.,front_v-.2)
            front_station += .0625*(front_v+next_front)
            front_v = next_front
            front_margin = min(front_margin,front_station-sx-2.6-max(3.,1.3*max(0.,vx-front_v)))
        bumper = rear-2.6+sx-sr
        required = max(3.,1.+1.1*max(0.,vr-vx))
        minimum = min(minimum,bumper)
        worst_margin = min(worst_margin,bumper-required)
        exit_distance = return_length if phase=="ongoing" else .82*return_length+1.
        if leave_station is not None and sx-leave_station>=exit_distance:
            reach_time = t
            break
    plausible = reach_time is not None
    return dict(time_to_vacate_s=reach_time, projected_rear_bumper_gap_m=float(minimum) if rear_seen else None,
                rear_margin_m=float(worst_margin) if rear_seen else None,
                target_front_margin_m=float(front_margin) if target_front<60. and phase=="entry" else None,
                allowed=bool(plausible and front_margin>0. and (not rear_seen or worst_margin>0.)))
