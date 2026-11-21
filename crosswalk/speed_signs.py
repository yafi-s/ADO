"""Preview direction-matched native speed signs along the driver's navigation route."""
import math
import re
import xml.etree.ElementTree as ET

from .city_cruise import CRUISE_MAX


def matching_objects(xml):
    """Town03 exports roadside Speed_XX objects separately from traffic-light signals."""
    signs = {}
    for road in ET.fromstring(xml).iter("road"):
        for item in road.findall("objects/object"):
            label = re.fullmatch(r"Speed_(\d+(?:\.\d+)?)", item.get("name", ""), re.IGNORECASE)
            facing = item.get("orientation")
            if not label or facing not in ("-", "+"):
                continue
            limit = float(label.group(1))
            lateral = float(item.get("t", 0))
            # Native roadside objects face the indicated carriageway and stand
            # on that side of the reference line. Opposing signs are excluded.
            if not 0. < limit <= 200. or not lateral or (lateral < 0.) != (facing == "-"):
                continue
            key = (int(road.get("id")), -1 if facing == "-" else 1)
            signs.setdefault(key, []).append((float(item.get("s")), limit))
    return signs


class SignedRouteLimits:
    def __init__(self, zones, nose_m):
        self.zones = tuple(sorted(zones))
        self.nose_m = nose_m

    @classmethod
    def from_map(cls, road_map, locations, arc, nose_m):
        signs = matching_objects(road_map.to_opendrive())
        if not signs:
            return cls([], nose_m)
        points = [road_map.get_waypoint(location) for location in locations]
        zones = set()
        for i, (a,b) in enumerate(zip(points, points[1:])):
            if a is None or b is None or a.road_id != b.road_id or a.lane_id != b.lane_id or a.s == b.s:
                continue
            lane_side = -1 if a.lane_id < 0 else 1
            if (b.s > a.s) != (lane_side == -1):
                continue
            for s, limit in signs.get((a.road_id, lane_side), ()):
                ratio = (s-a.s) / (b.s-a.s)
                if 0. <= ratio < 1.:
                    station = arc[i] + ratio*(arc[i+1]-arc[i])
                    zones.add((round(float(station), 3), limit))
        return cls(zones, nose_m)

    def approach(self, progress, speed, posted):
        """Return an active approach cap and diagnostics; the actual sign remains authoritative."""
        best = None
        if not math.isfinite(posted) or posted <= 0.:
            return None
        for station, kmh in self.zones:
            distance = station - progress
            if distance < -self.nose_m or kmh >= posted:
                continue
            # Anticipate a comfortable 1.5 m/s² transition. Allow the tracking
            # actuator 1.2 s to settle before the front bumper enters the zone.
            terminal = max(0., kmh / 3.6 - .2)
            room = max(0., distance - self.nose_m - 1.2 * max(speed, terminal))
            cap = math.sqrt(terminal*terminal + 3.*room)
            if cap < min(posted/3.6, CRUISE_MAX) and (best is None or cap < best[0]):
                best = (cap, kmh, distance)
        return best
