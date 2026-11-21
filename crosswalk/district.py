"""Geographically separated missions in CARLA's bundled Town01 map.

IDs are OpenDRIVE roads and junctions, shared by all lanes. The unused central
blocks and perimeter roads form a buffer between the two districts.
"""

DISTRICTS = {
    "train": ({0, 1, 2, 3, 9, 10, 16, 21, 25}, {26, 54, 110, 171, 222, 332}),
    "heldout": ({5, 6, 7, 12, 19, 24}, {87, 143, 194, 255}),
}

# Approach, through-block, and destination streets. Both ends of the middle
# block are intersections. The destination is 20 m onto the final street.
DESTINATIONS = {
    "train": {"market": (25, 1, 16), "station": (16, 1, 25)},
    "heldout": {"market": (6, 24, 12), "station": (12, 19, 6)},
}


def admits(points, district):
    if district is None:
        return True
    roads, junctions = DISTRICTS[district]
    return all((w.junction_id in junctions if w.is_junction else w.road_id in roads) for w in points)


def connectors(road_map):
    result = {}
    for entry, end in road_map.get_topology():
        if not entry.is_junction:
            continue
        previous, following = entry.previous(2.0), end.next(2.0)
        if len(previous) == len(following) == 1:
            result[previous[0].road_id, following[0].road_id] = entry
    return result


def trip_points(road_map, district, destination):
    streets = DESTINATIONS[district][destination]
    turns = connectors(road_map)
    first, second = (turns[a, b] for a, b in zip(streets, streets[1:]))
    points = [first.previous(26.0)[0]]
    planned = (first, second)
    turn = 0
    final_street_steps = 0
    for _ in range(300):
        choices = points[-1].next(1.0)
        if len(choices) > 1 and turn < len(planned):
            target = planned[turn]
            choices = [p for p in choices if (p.road_id, p.lane_id) == (target.road_id, target.lane_id)]
        if len(choices) != 1:
            raise RuntimeError(f"Broken Town01 path to {district}/{destination}")
        nxt = choices[0]
        points.append(nxt)
        if turn < len(planned) and nxt.road_id == planned[turn].road_id:
            turn += 1
        if nxt.road_id == streets[-1]:
            final_street_steps += 1
            if final_street_steps == 20:
                return points
    raise RuntimeError(f"Destination {district}/{destination} beyond mission bounds")
