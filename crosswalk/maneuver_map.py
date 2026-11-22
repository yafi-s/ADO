"""Find connected, reversible passing lanes along the requested navigation route."""
import numpy as np
import carla

from .urban_map import _left_lane


def key(wp):
    return wp.road_id, wp.lane_id


def reversible(wp):
    try:
        target = _left_lane(wp)
    except RuntimeError:
        return None
    if (target.right_lane_marking.type != carla.LaneMarkingType.Broken or
            target.lane_change not in (carla.LaneChange.Right, carla.LaneChange.Both)):
        return None
    return target


def xyz(points):
    return np.asarray([(p.x, p.y, p.z) for p in points], dtype=float)


class NativeCorridor:
    def __init__(self, source, target, begin, end, merge_at=None):
        self.source_key, self.target_key = key(source[begin]), key(target[begin])
        self.source = xyz([p.transform.location for p in source])
        self.target = xyz([p.transform.location for p in target])
        delta = np.diff(self.source[:, :2], axis=0)
        lengths = np.linalg.norm(delta, axis=1)
        self.arc = np.r_[0., np.cumsum(lengths)]
        self.direction = np.vstack((delta/lengths[:,None], delta[-1]/lengths[-1]))
        self.begin, self.end = self.arc[begin], self.arc[end]
        self.merge_at = self.arc[merge_at] if merge_at is not None else None
        self.source_width = np.asarray([wp.lane_width for wp in source])
        self.width = np.linalg.norm(self.target[:,:2]-self.source[:,:2], axis=1)

    def interpolate(self, stations, values):
        return np.stack([np.interp(stations, self.arc, column) for column in values.T], axis=-1)

    def locate(self, xy, heading):
        offset = xy-self.source[:-1,:2]
        along = np.clip(np.sum(offset*self.direction[:-1], axis=1), 0., np.diff(self.arc))
        error = offset-along[:,None]*self.direction[:-1]
        i = int(np.argmin((error**2).sum(axis=1)))
        s = self.arc[i]+along[i]
        side = float(error[i] @ (-self.direction[i,1], self.direction[i,0]))
        if (self.begin-1. <= s <= self.end+.5 and np.linalg.norm(error[i]) < 5.15
                and -self.width[i]-1.65 < side < 1.75 and heading @ self.direction[i] > .85):
            return float(s), side
        return None

    def project_scan(self, world, station):
        # Iteratively project onto the native curved centerline. Each correction
        # uses the local tangent, including road banking/elevation in the height.
        distance = (world[:,:2]-self.interpolate(station, self.source)[:2]) @ self.interpolate(station, self.direction)
        s = np.clip(station+distance, 0., self.arc[-1])
        for _ in range(3):
            native = self.interpolate(s, self.source)
            tangent = self.interpolate(s, self.direction)
            rel = world[:,:2] - native[:,:2]
            adjustment = (rel*tangent).sum(axis=1)
            s = np.clip(s+adjustment, 0., self.arc[-1])
        native = self.interpolate(s, self.source)
        tangent = self.interpolate(s, self.direction)
        delta = world[:,:2]-native[:,:2]
        side = delta[:,1]*tangent[:,0]-delta[:,0]*tangent[:,1]
        # Reject endpoint projections of objects from other native segments.
        inside = abs((delta*tangent).sum(axis=1)) < 1.1
        return s-station, side, world[:,2]-native[:,2], np.interp(s, self.arc, self.width), inside

    def goals(self, stations, alpha):
        src = self.interpolate(stations, self.source[:,:2])
        dst = self.interpolate(stations, self.target[:,:2])
        return src+(dst-src)*alpha[:,None]


def _append_external(road_map, wp, delta, n):
    pairs = []
    last = wp
    for i in range(1,n+1):
        sample = road_map.get_waypoint_xodr(*key(wp), wp.s+delta*i)
        if not sample or abs(sample.s-last.s) < .3 or not (adjacent := reversible(sample)):
            break
        if sample.transform.location.distance(last.transform.location) > 2.:
            break
        pairs.append((sample,adjacent))
        last = sample
    return pairs


def discover(road_map, route):
    """Sample at <=1.1 m; split at junctions, markings, turns and discontinuities.

    The optional navigation merge is detected by location-to-lane correspondence.
    It is distinct from an out-and-back discretionary overtake.
    """
    if len(route) < 2:
        return []
    centers = xyz(route)
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(centers, axis=0), axis=1))]
    stations = np.r_[np.arange(0.,arc[-1],1.),arc[-1]]
    path = np.stack([np.interp(stations,arc,c) for c in centers.T],axis=1)
    points = [road_map.get_waypoint(carla.Location(*p), project_to_road=False,
                                    lane_type=carla.LaneType.Driving) for p in path]
    found = []
    i = 0
    while i < len(points):
        start = i
        first = points[i]
        if first is None or not (adjacent := reversible(first)):
            i += 1
            continue
        src_key, dst_key = key(first), key(adjacent)
        samples, merge = [], None
        sign = -1 if first.lane_id > 0 else 1
        while i < len(points):
            wp = points[i]
            if not wp or wp.is_junction or wp.road_id != first.road_id:
                break
            if key(wp) == src_key and merge is None:
                src = wp
            elif key(wp) == dst_key and samples:
                src = road_map.get_waypoint_xodr(*src_key, wp.s)
                if merge is None:
                    merge = len(samples)
            else:
                break
            if not src or not (dst := reversible(src)) or key(dst) != dst_key:
                break
            if samples:
                prev = samples[-1][0]
                advance = sign*(src.s-prev.s)
                if advance < -.05 or src.transform.location.distance(prev.transform.location) > 2.5:
                    break
                if advance < .15:
                    i += 1
                    continue
            samples.append((src,dst))
            i += 1
        if len(samples) >= 38:
            prefix = _append_external(road_map,samples[0][0],-sign,55)[::-1]
            suffix = _append_external(road_map,samples[-1][0],sign,45)
            begin, end = len(prefix), len(prefix)+len(samples)-1
            if merge is not None and merge >= len(samples):
                merge = None
            full = prefix+samples+suffix
            found.append(NativeCorridor(*zip(*full),begin,end,None if merge is None else begin+merge))
        i = max(i,start+1)
    return found
