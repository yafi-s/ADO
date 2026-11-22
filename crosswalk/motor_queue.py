"""Learned following preference for a sensor-confirmed vehicle in the source lane."""
from collections import deque

import numpy as np

from .maneuver_map import discover
from .route_maneuvers import LOOK, RouteManeuvers, compare_progress
from .source_vehicle import source_surface_vehicle


class MotorQueue:
    """Track one measured source-lane bumper in native road coordinates."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sightings = deque(maxlen=6)
        self.motion = deque(maxlen=24)
        self.anchor = None
        self.last_surface = None
        self.since = None
        self.wait = 0
        self.latest = None
        self.preference_inputs = None
        self.preference = None
        self.probability_follow = None
        self.intent_gain_m = None
        self.intent_leaf = None

    def _associate(self, frame, observed, witness):
        box_frame = witness["box_frame"]
        if self.sightings and self.sightings[-1][0] == box_frame:
            return
        self.sightings.append((box_frame, observed, witness["label"], witness.get("radial_ground_speed_mps")))
        matching = [
            hit for hit in self.sightings
            if 0 <= frame - hit[0] <= 9 and abs(hit[1] - observed) <= .85
            and hit[3] is not None and abs(hit[3]) <= .70
        ]
        if len(matching) < 2 or matching[-1][0] - matching[0][0] < 2:
            return
        if self.anchor is None or abs(observed - self.anchor) > 2.:
            if self.anchor is not None:
                self.wait = 0
                self.motion.clear()
            self.since = matching[0][0]
        self.anchor = float(np.median([hit[1] for hit in matching]))
        self.last_surface = frame

    def update(self, frame, station, gap, raw, source_v, source_age, ego_v, witness):
        self.latest = witness
        measured = source_age == 0 and raw < 59.
        if measured:
            self.motion.append((frame, station + raw))
        if (witness and witness.get("stationary_doppler") and source_age in (0, 1, 2)
                and gap < 59. and source_v is not None and abs(source_v) < .65):
            self._associate(frame, station + gap, witness)
        if self.anchor is None:
            return
        if measured:
            elapsed = (frame - self.last_surface) * .1
            observed = station + raw
            if elapsed > 2. or abs(observed - self.anchor) > 1.25 + 5. * elapsed:
                self.reset()
                return
            self.anchor = observed
            self.last_surface = frame
        elif frame - self.last_surface > 10:
            self.reset()
            return

        velocity = source_v if source_v is not None else 0.
        if measured and abs(velocity) <= .55:
            self.wait = self.wait + 1 if gap - 2.5 < 15. and ego_v < .50 else 0
        elif measured and (abs(velocity) > .85 or ego_v > 1.):
            self.wait = 0

    @staticmethod
    def _surface_speed(points, fallback):
        rates = [
            (later - earlier) / (.1 * (end - start))
            for i, (end, later) in enumerate(points)
            for start, earlier in points[:i] if end - start >= 3
        ]
        return float(np.clip(np.median(rates), 0., 30.)) if rates else fallback

    def preference_features(self, frame, gap, speed, source_v, gain):
        if self.anchor is None or self.since is None:
            return None
        recent = [point for point in self.motion if frame - point[0] <= 9]
        earlier = [point for point in self.motion if 10 <= frame - point[0] <= 19]
        now = self._surface_speed(recent, max(0., source_v or 0.))
        previous = self._surface_speed(earlier, now)
        interval = 1.
        if earlier and recent:
            interval = .1 * (np.mean([t for t, _ in recent]) - np.mean([t for t, _ in earlier]))
        return dict(
            source_gap_m=gap, ego_speed_mps=speed, source_speed_mps=now,
            previous_source_speed_mps=previous, source_acceleration_mps2=(now - previous) / max(.3, interval),
            observed_stopped_wait_s=.1 * self.wait, source_observed_s=.1 * (frame - self.since),
            projected_lane_gain_m=gain,
        )

    def learned_preference(self, model, features):
        self.preference_inputs = features
        details = (None, None, None, None) if features is None else model.details(features)
        self.preference, self.probability_follow, self.intent_gain_m, self.intent_leaf = details

    def choose(self, fitted, required_merge=False):
        if not required_merge and self.anchor is not None and self.preference == "follow":
            return "follow"
        return fitted


class _VehicleChoice:
    """Preserve the original fitted action and probabilities in planner telemetry."""

    def __init__(self, planner, original):
        self.planner = planner
        self.original = original
        self.fitted = None

    def predict(self, values):
        self.fitted, probabilities = self.original.predict(values)
        planner = self.planner
        action = planner.queue.choose(self.fitted) if planner.evaluating_source else self.fitted
        if planner.defer_source and action == "follow":
            planner.holding = False
            planner.pass_votes = 0
        return action, probabilities


class VehicleQueuePlanner(RouteManeuvers):
    """Apply the selected stationary-intent head before a discretionary lane change."""

    def __init__(self, corridors, *, intent, choice):
        if intent is None or choice is None:
            raise ValueError("The queue planner requires a loaded intent head and route-choice model")
        self.intent = intent
        self.stationary_intent_checkpoint = intent.checkpoint_path
        self.stationary_intent_sha256 = intent.sha256
        super().__init__(corridors, _VehicleChoice(self, choice))
        self.camera_inputs = None
        self._bound_camera = None

    @classmethod
    def from_route(cls, road_map, route, *, intent, choice):
        return cls(discover(road_map, route), intent=intent, choice=choice)

    def _reset(self):
        super()._reset()
        self.queue = MotorQueue()
        self.evaluating_source = False
        self.defer_source = False

    def _measure(self, pose, scan, station, speed, lateral=0.):
        result = super()._measure(pose, scan, station, speed, lateral)
        values, raw = result[:2]
        data = self.camera_inputs
        early_outbound = (
            self.start is not None and self.return_start is None
            and self.phase == "change_left" and lateral > -.35
        )
        self.defer_source = False
        self.evaluating_source = (data is not None and self.corridor.merge_at is None
                                  and (self.start is None or early_outbound))
        if not self.evaluating_source:
            return result

        gap, source_speed, front, rear = values[:4]
        witness = source_surface_vehicle(
            data["boxes"], scan, data["radar"], pose, data["box_pose"],
            self.corridor, station, gap, speed, data["frame"], data["box_frame"],
        )
        source = self.surfaces[0]
        self.queue.update(data["frame"], station, gap, raw[0], source.speed, source.age, speed, witness)
        front_speed = self.surfaces[2].speed or 0.
        _, gain, _ = compare_progress("pass", gap, source_speed, front, rear, front_speed, result[3][2])
        features = self.queue.preference_features(data["frame"], gap, speed, source.speed, gain)
        self.queue.learned_preference(self.intent, features)
        self.defer_source = self.queue.anchor is not None and self.queue.preference == "follow"
        if self.defer_source and self.start is not None:
            # An early signal can be cancelled while the body is still in
            # the source lane. The lateral guard above protects an active crossing.
            self.start = self.return_start = None
        return result

    def bind_camera(self, camera_inputs):
        """Supply this tick's control-camera exposure ahead of the shared preview call."""
        self._bound_camera = camera_inputs

    def preview(self, ego, lidar, defaults, speed, lookaheads=LOOK, *, camera_inputs=None):
        if camera_inputs is None:
            camera_inputs, self._bound_camera = self._bound_camera, None
        self.camera_inputs = camera_inputs
        points = super().preview(ego, lidar, defaults, speed, lookaheads)
        if "source_gap_m" in self.info:
            if self.model.fitted is not None:
                self.info["fitted_choice"] = self.model.fitted
            queue = self.queue
            self.info["source_motor_vehicle"] = dict(
                source_track_state="associated_vehicle" if queue.anchor is not None else "unconfirmed",
                evaluated=self.evaluating_source, queue_follow=self.defer_source,
                last_supported_camera=queue.sightings[-1][0] if queue.sightings else None,
                tracked_bumper_m=queue.anchor, observed_dwell_s=round(.1 * queue.wait, 1),
                matched_vehicle=queue.latest, preference=queue.preference,
                probability_follow=queue.probability_follow, learned_pass_gain_m=queue.intent_gain_m,
                leaf_index=queue.intent_leaf, preference_inputs=queue.preference_inputs,
            )
        return points
