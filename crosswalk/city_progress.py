"""Visibility-limited driving speed from native open-road, turning and braking measurements."""
from collections import deque
import io

import numpy as np

from . import behavior_v2 as v2
from .city_clearance import BODY, ClearanceModel, EdgeObserver, FEATURE_NAMES as CLEARANCE_NAMES
from .city_cruise import CRUISE_MAX
from .city_crossing import CrossingObserver
from .city_policy import INDEX
from .policy import _returns, LIDAR_Z


VERSION = "ado-city-progress-v1"
PROFILE = ("observed_corridor_m", "observed_conflict_m", "observed_blind_corner_m",
           "adjacent_clearance_m", "confirmed_scans", "crossing_stop_gap_m",
           "crossing_entry_s", "crossing_inward_mps", "crossing_age_s")
FEATURE_NAMES = (*CLEARANCE_NAMES[:-2], *PROFILE, *CLEARANCE_NAMES[-2:])
PARAMETERS = ("free_speed_mps", "lateral_acceleration_mps2", "recovery_acceleration_mps2")
CELL = 4.
HORIZON = 24.


def curvature(x):
    # Distance between the midpoints of the navigation segments that supplied each heading.
    return np.maximum.reduce([np.abs(x[..., INDEX[name]]) / length for name, length in
                              (("bend_7_13", 5.), ("bend_13_25", 9.), ("bend_25_40", 14.))])


def _pavement_horizon(station, mask):
    cells = np.bincount((station[mask] / CELL).astype(int), minlength=int(HORIZON / CELL))
    gaps = np.flatnonzero(cells < 2)
    return CELL * (int(gaps[0]) if len(gaps) else len(cells))


class FlankMotion:
    """Track the far end of a continuous flank in ego-motion-compensated metres."""

    def __init__(self):
        self.ends = deque(maxlen=8)
        self.last = None
        self.anchored = False

    def stationary(self, frame, travel, tip, margin, tactical):
        current = (travel + tip, margin)
        if (self.last != frame - 1 or self.ends and
                (abs(current[0] - self.ends[-1][0]) > 1. or abs(margin-self.ends[-1][1]) > .25)):
            self.ends.clear()
            self.anchored = False
        self.last = frame
        self.ends.append(current)
        # Alongside a long face, the nearest usable source beam rides at the
        # scanner's exclusion boundary. Its apparent speed is then ego speed.
        # Only continue a previous stationary identification at this boundary,
        # using several fresh observations of the *far* physical end.
        saturated = tactical.get("raw_source_gap_m", np.inf) < 3.05
        direct = abs(tactical.get("source_speed_mps", np.inf)) < .8
        if not saturated:
            self.anchored = direct
        if len(self.ends) >= 6:
            values = np.asarray(self.ends)
            drift = abs(np.median(values[-3:], axis=0) - np.median(values[:3], axis=0))
            if drift[0] > .4 or drift[1] > .18:
                self.anchored = False
        return self.anchored and (direct or saturated and len(self.ends) >= 6)


def _profile(scan, path, tactical, *, motion=None, frame=0, travel=0.):
    cloud = _returns(scan)
    x, y, z = cloud[:, 0], cloud[:, 1], cloud[:, 2] + LIDAR_Z
    ground = v2._ground_plane(x, y, z)
    nearby = (x >= 0.) & (x < 32.) & (abs(y) < 22.)
    station, lateral = v2._project(x[nearby], y[nearby], path)
    h = z[nearby] - (ground[0] * x[nearby] + ground[1] * y[nearby] + ground[2])
    keep = (station >= 0.) & (station < HORIZON)
    s, side, h = station[keep], lateral[keep], h[keep]
    pavement = abs(h) < .32
    solid = (h > .28) & (h < 3.)
    corridor = _pavement_horizon(s, pavement & (abs(side) < BODY))
    body = v2._gap(s, solid & (abs(side) <= BODY), support=2)
    corridor = min(corridor, body)
    horizons = []
    for sign in (-1., 1.):
        d = sign * side
        seen = _pavement_horizon(s, pavement & (d >= 4.) & (d <= 9.))
        obstruction = v2._gap(s, solid & (d > BODY) & (d < 7.), support=4)
        horizons.append(min(seen, obstruction))
    # For a confirmed pass on a separate, measured lane, the solid flank still
    # hides the roadway. Limit extra speed by the distance to the blind end.
    # It must be a continuous face; a pedestrian, posts, or a short return is insufficient.
    corner, margin = 0., 0.
    if (tactical.get("committed") and tactical.get("source_age_scans") == 0
            and tactical.get("adjacent_pavement_seen") == [True, True]
            and min(tactical.get("target_front_m", 0.), tactical.get("target_rear_m", 0.)) > 20.
            and abs(tactical.get("source_lateral_m", 0.)) > 2.6):
        sign = -np.sign(tactical["source_lateral_m"])
        flank = sign * side
        face = solid & (flank > BODY + .45) & (flank < 3.5)
        near = s[face]
        if len(near) >= 12:
            inner = float(np.quantile(flank[face], .10))
            plane = face & (abs(flank - inner) < .25)
            points = np.sort(s[plane])
            if len(points) >= 12:
                sections = np.split(points, np.flatnonzero(np.diff(points) > 1.15) + 1)
                first = sections[0]
                if first[0] < 8. and np.ptp(first) >= 2. and np.ptp(h[plane]) > .8:
                    tip = float(first[-1])
                    clearance = max(0., inner - BODY)
                    stationary = (motion.stationary(frame, travel, tip, clearance, tactical) if motion is not None
                                  else abs(tactical.get("source_speed_mps", np.inf)) < .8)
                    occupied = solid & (flank >= BODY) & (flank < inner - .25)
                    # A person or other surface between us and the measured wall cancels the pass.
                    nearer = v2._gap(s, occupied, support=2)
                    if stationary and first[0] < 3.:
                        corner = min(tip, nearer, horizons[0 if sign > 0 else 1])
                        margin = clearance
    return corridor, min(horizons), corner, margin


class ProgressObserver(EdgeObserver):
    uses_conflict_profile = True

    def __init__(self):
        super().__init__()
        self.recent = deque(maxlen=3)
        self.prior_frame = None
        self.flank = FlankMotion()
        self.crossing = CrossingObserver()

    def observe(self, obs, detections, local_route, speed, *, route_preview=None, stopline_gap=None):
        x = super().observe(obs, detections, local_route, speed, route_preview=route_preview, stopline_gap=stopline_gap)
        frame = obs.get("progress_frame")
        if frame is None or self.prior_frame is None or frame != self.prior_frame + 1:
            self.recent.clear()
        self.prior_frame = frame
        path = v2._checked_route(local_route, route_preview)
        cross = self.crossing.observe(obs["lidar"], obs.get("progress_ego_pose"), frame, speed, path)
        image = obs.get("front")
        camera_ok = False
        if image is not None and image.ndim == 3 and image.size:
            sample = image[image.shape[0] // 3:3 * image.shape[0] // 4:12, ::12]
            camera_ok = bool(sample.size and np.quantile(sample, .75) > 24 and np.std(sample) > 7)
        near_person = any(d["label"] in ("person", "bicycle", "motorcycle") and
                          d["box"][3] - d["box"][1] > .055 for d in detections)
        if (frame is None or not camera_ok or not 0. <= obs.get("progress_camera_age_s", np.inf) <= .15
                or near_person or not len(_returns(obs["radar"]))):
            self.recent.clear()
        else:
            measured = _profile(obs["lidar"], path, obs.get("progress_tactical", {}),
                                motion=self.flank, frame=frame, travel=self.travel)
            self.recent.append(measured)
        profile = np.min(self.recent, axis=0) if len(self.recent) == self.recent.maxlen else np.zeros(4)
        return np.r_[x, profile, len(self.recent), cross].astype(np.float32)


class ProgressModel(ClearanceModel):
    confirmed_tactical_release = True

    def __init__(self, base, coefficients):
        prior = ClearanceModel.load(io.BytesIO(base))
        super().__init__(prior.base_bytes, prior.cruise_forest, prior.actuator, prior.side_speed, prior.stopping)
        self.parent_checkpoint = base
        self.coefficients = np.asarray(coefficients, dtype=float)
        if (self.coefficients.shape != (3,) or not np.isfinite(self.coefficients).all()
                or np.any(self.coefficients <= 0.) or np.any(self.coefficients > [CRUISE_MAX, 4., 4.])):
            raise ValueError("Invalid native speed calibration")
        self.progress_evidence = {}

    def new_observer(self):
        super().new_observer()
        self.progress_evidence = {}
        return ProgressObserver()

    def predict(self, x):
        x = np.asarray(x)
        if x.shape != (len(FEATURE_NAMES),) or not np.isfinite(x).all():
            raise ValueError("Invalid city progress observation")
        self.profile = x[-len(PROFILE)-2:-2]
        return super().predict(np.r_[x[:-len(PROFILE)-2], x[-2:]])

    def _stoppable_speed(self, budget):
        a, b, slack, tested = self.stopping
        distance = max(0., budget - slack)
        return min(tested, 2. * distance / max(1.e-9, b + np.sqrt(b*b + 4.*a*distance)))

    def _actuation_distance(self, x):
        return min(super()._actuation_distance(x), self.profile[5])

    def _finish_request(self, x, baseline, requested):
        frozen = super()._finish_request(x, baseline, requested)
        corridor, sides, corner, margin, scans, conflict, eta, inward, age = map(float, self.profile)
        bend = float(curvature(x))
        turn_speed = np.sqrt(self.coefficients[1] / max(.001, bend))
        visibility = min(corridor, max(sides, corner))
        # For the separate, confirmed flank use the observed blind end only while
        # lateral clearance exceeds half a vehicle-width. Tighter gaps shrink the budget.
        if corner > sides:
            visibility = min(corridor, max(sides, corner * min(1., margin / BODY)))
        self.progress_evidence = dict(binding="legacy", base_request_mps=float(frozen),
            proposal_mps=float(frozen), sight_m=float(visibility), corridor_m=corridor,
            cross_sight_m=sides, blind_end_m=corner, adjacent_margin_m=margin,
            curvature_per_m=bend, supported_scans=int(scans), crossing_gap_m=conflict,
            crossing_entry_s=eta, crossing_inward_mps=inward, crossing_age_s=age)
        evidence = self.progress_evidence
        if conflict < 60.:
            # An approaching lateral surface is an independent braking constraint;
            # it may command a complete stop even when both cruise heads prefer motion.
            cap = self._stoppable_speed(conflict - self.standoff)
            evidence.update(binding="crossing_yield", proposal_mps=cap)
            return min(frozen, cap)
        if scans < 3:
            evidence["binding"] = "unconfirmed"
            return frozen
        if x[INDEX["mapped_lamp_gap_m"]] < 59. or max(self.light_window, default=0.) > .25:
            evidence["binding"] = "signal_approach"
            return frozen
        if min(x[INDEX[n]] for n in ("oncoming_gap", "oncoming_gap_recent", "oncoming_near_gap_recent")) < 34.:
            evidence["binding"] = "opposing_traffic"
            return frozen
        if max(x[INDEX["rgb_person_height"]], x[INDEX["rgb_person_height_recent"]]) > .055:
            evidence["binding"] = "vulnerable_road_user"
            return frozen
        distance = min(x[INDEX[n]] for n in ("path_body_lidar", "path_body_radar",
                                             "path_body_lidar_recent", "path_body_radar_recent"))
        if distance < HORIZON:
            evidence["binding"] = "occupied_corridor"
            return frozen
        limits = {"demonstrated_cruise": self.coefficients[0], "turn": turn_speed,
                  "sight_distance": self._stoppable_speed(visibility - self.standoff),
                  "body_distance": self._stoppable_speed(distance - self.standoff),
                  "posted": (self.posted_limit_kmh if np.isfinite(self.posted_limit_kmh)
                              and self.posted_limit_kmh > 0. else 30.) / 3.6}
        if self.signed_speed_limit is not None:
            limits["signed_approach"] = self.signed_speed_limit
        if self.tactical_speed_limit is not None:
            limits["tactical"] = self.tactical_speed_limit
        binding = min(limits, key=limits.get)
        proposal = float(limits[binding])
        evidence.update(proposal_mps=proposal, binding=binding)
        if proposal <= frozen:
            return frozen
        if self.last_speed_request is not None:
            recovery = self.last_speed_request + .1 * self.coefficients[2]
            if recovery < proposal:
                evidence["binding"] = "recovery"
                proposal = recovery
        return float(max(frozen, proposal))

    def save(self, path):
        np.savez_compressed(path, version=VERSION, revision=3, feature_names=FEATURE_NAMES, parameters=PARAMETERS,
                            coefficients=self.coefficients, base=np.frombuffer(self.parent_checkpoint, dtype=np.uint8))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as d:
            if (str(d["version"].item()) != VERSION or "revision" not in d or int(d["revision"].item()) != 3
                    or tuple(d["feature_names"]) != FEATURE_NAMES
                    or tuple(d["parameters"]) != PARAMETERS):
                raise ValueError("Incompatible city progress checkpoint")
            return cls(d["base"].tobytes(), d["coefficients"])
