"""Signals and lane markings of a town, read from its OpenDRIVE file without a simulator.

The town geometry exported by ``city_map`` holds lane centre lines and building bounds.
This adds what the road description itself records about those lanes: which way each one
is driven, how its two edges are painted, where each traffic signal's post stands, which
way the traffic it governs is heading and how far the governed lanes are from the post. Lanes are keyed by their end points, so the file matches a town export
however that export orders its lanes.
"""
from collections import defaultdict
from pathlib import Path
import argparse
import hashlib
import json
import math

import carla


SIGNAL = "1000001"      # OpenDRIVE type of a three-lamp traffic signal in the CARLA towns
STEP_M = 2.             # the spacing city_map samples lane centres at


def _xy(location):
    return [round(location.x, 2), round(location.y, 2)]


def _paint(marking):
    kind = str(marking.type).lower()
    if kind in ("none", "other", "curb", "grass", "bottsdots"):
        return None
    return {"line": "broken" if "broken" in kind else "solid",
            "color": "yellow" if str(marking.color).lower() == "yellow" else "white"}


def _lanes(road_map):
    groups = defaultdict(list)
    for point in road_map.generate_waypoints(STEP_M):
        if point.lane_type == carla.LaneType.Driving:
            groups[point.road_id, point.section_id, point.lane_id].append(point)
    lanes = []
    for (_, _, lane_id), points in sorted(groups.items()):
        if len(points) < 2:
            continue
        points.sort(key=lambda p: p.s)
        middle = points[len(points) // 2]
        # Points are stored by rising s. Lanes right of the reference line are driven that way.
        forward = lane_id < 0
        left, right = _paint(middle.left_lane_marking), _paint(middle.right_lane_marking)
        # `plus` is the edge on the right of the stored direction, `minus` the edge on its left.
        plus, minus = (right, left) if forward else (left, right)
        lanes.append({"a": _xy(points[0].transform.location), "b": _xy(points[-1].transform.location),
                      "forward": forward, "width": round(middle.lane_width, 2), "plus": plus, "minus": minus})
    return lanes


def _signals(road_map):
    """One entry per signal post. The road description lists a post once for every junction lane it governs."""
    posts = defaultdict(list)
    for landmark in road_map.get_all_landmarks_of_type(SIGNAL):
        posts[int(landmark.id)].append(landmark)
    signals = []
    for key in sorted(posts):
        post = posts[key][0].transform
        # The landmark points the way the traffic it governs is heading. Its lamps face the other way.
        yaw = math.radians(post.rotation.yaw)
        entry = {"id": key, "post": _xy(post.location), "governs_yaw": round((post.rotation.yaw + 180.) % 360. - 180., 1)}
        offsets = []
        for landmark in posts[key]:
            for first, last in landmark.get_lane_validities():
                for lane_id in range(min(first, last), max(first, last) + 1):
                    governed = road_map.get_waypoint_xodr(landmark.road_id, lane_id, landmark.s) if lane_id else None
                    if governed is not None:
                        delta = governed.transform.location - post.location
                        # Metres from the post to that lane's centre, positive to the left of its traffic.
                        offsets.append(delta.x * math.sin(yaw) - delta.y * math.cos(yaw))
        if offsets:
            # A head hangs over the governed lanes: out as far as the middle of them.
            entry["reach"] = round((min(offsets) + max(offsets)) / 2, 2)
        signals.append(entry)
    return signals


def features(name, xodr):
    road_map = carla.Map(name, xodr)
    return {"map": name, "source_sha256": hashlib.sha256(xodr.encode("utf-8")).hexdigest(),
            "units": "metres and degrees, CARLA world coordinates",
            "lanes": _lanes(road_map), "signals": _signals(road_map)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xodr", type=Path, help="the town's OpenDRIVE file")
    parser.add_argument("--map", required=True, help="the town's name, as recordings report it")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = features(args.map, args.xodr.read_text(encoding="utf-8"))
    output = args.output or Path("web/media/map-features") / f"{args.map}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"{len(result['lanes'])} lanes and {len(result['signals'])} signals", flush=True)
