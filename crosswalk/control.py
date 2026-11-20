"""Shared pure-pursuit steering and low-level speed tracking in CARLA."""
import math

import carla


def vehicle_speed(ego):
    v = ego.get_velocity()
    return math.sqrt(v.x**2 + v.y**2 + v.z**2)


def command(ego, route, speed, target):
    transform = ego.get_transform()
    pos = transform.location
    yaw = math.radians(transform.rotation.yaw)
    # The sedan's reference point is near its center; pure pursuit uses the rear axle.
    rear_x, rear_y = pos.x - 1.4 * math.cos(yaw), pos.y - 1.4 * math.sin(yaw)
    nearest = min(range(len(route)), key=lambda i: (route[i].x - rear_x)**2 + (route[i].y - rear_y)**2)
    lookahead = 4.0 + speed * 0.45
    goal = next((p for p in route[nearest:] if math.hypot(p.x - rear_x, p.y - rear_y) >= lookahead), route[-1])
    dx, dy = goal.x - rear_x, goal.y - rear_y
    lateral = -dx * math.sin(yaw) + dy * math.cos(yaw)
    curvature = 2 * lateral / max(1., dx*dx + dy*dy)
    steer = max(-1., min(1., math.atan(2.8 * curvature) / math.radians(55)))
    error = target - speed
    throttle = min(0.70, max(0.0, 0.10 + 0.32 * error)) if target > 0.12 else 0.
    brake = min(1., max(0., -0.35 * error))
    if target <= 0.12 and speed < 0.35:
        brake = 1.
    return carla.VehicleControl(throttle=throttle if brake == 0 else 0., brake=brake, steer=steer)
