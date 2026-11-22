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
    def __init__(self, source, target, begin, end, merge_at=None, target_from=None):
        self.source_key, self.target_key = key(source[begin]), key(target[begin])
        self.source = xyz([p.transform.location for p in source])
        self.target = xyz([p.transform.location for p in target])
        delta = np.diff(self.source[:, :2], axis=0)
        lengths = np.linalg.norm(delta, axis=1)
        self.arc = np.r_[0., np.cumsum(lengths)]
        self.direction = np.vstack((delta/lengths[:,None], delta[-1]/lengths[-1]))
        self.begin, self.end = self.arc[begin], self.arc[end]
        self.merge_at = self.arc[merge_at] if merge_at is not None else None
        self.navigation_target_from = self.arc[target_from] if target_from is not None else None
        self.source_width = np.asarray([wp.lane_width for wp in source])
        self.target_width = np.asarray([wp.lane_width for wp in target])
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

    def navigation_target_settled(self, station, lateral, heading, body_vertices):
        """Release steering only where the remaining route follows the new lane."""
        if self.navigation_target_from is None or station < self.navigation_target_from:
            return False
        width = np.interp(station,self.arc,self.width)
        if abs(lateral+width) >= .45 or heading @ self.interpolate(station,self.direction) <= np.cos(np.deg2rad(7.)):
            return False
        corners = np.array([(p.x,p.y,p.z) for p in body_vertices])
        rel, side, _, separation, inside = self.project_scan(corners,station)
        lane_half = np.interp(station+rel,self.arc,self.target_width)/2.
        return bool(np.all(inside & (abs(side+separation) < lane_half-.10)))


def merge_approach_ceiling(corridor, station, lateral, features, covered, sight, source_path_seen):
    """Approach the rear observation window in the source lane at walking speed."""
    gap, _, front, rear, _, _, speed = features
    # At a road entrance the surveyed pair simply does not extend 24 m behind
    # the ego yet. Never extend it across the upstream junction or interpret
    # unmapped space as an open lane. A few metres of source-lane travel can
    # expose the missing area; the usual full scans still gate any crossing.
    if (corridor.merge_at is None or abs(lateral) >= .35 or not source_path_seen or
            not covered[0] or not covered[2] or sight[0] < 24. or
            sight[2] <= max(24., 2.*speed+5.) or sight[3] < min(14., .75*station) or
            min(front,rear) < 60.):
        return 0.
    # Allow some extra sight beyond the 24 m threshold for sparse sweeps.
    # Leave space to complete the 23 m change before either the source
    # obstruction or the end of the permitted lane pair.
    advance = max(0., min(34.-station, gap-29., corridor.end-station-28.))
    return min(2., advance/1.4)


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
        samples, centered, merge = [], [], None
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
            target_pos = dst.transform.location
            centered.append(np.linalg.norm(path[i,:2] - (target_pos.x,target_pos.y)) < .10)
            i += 1
        if len(samples) >= 38:
            prefix = _append_external(road_map,samples[0][0],-sign,55)[::-1]
            suffix = _append_external(road_map,samples[-1][0],sign,45)
            begin, end = len(prefix), len(prefix)+len(samples)-1
            if merge is not None and merge >= len(samples):
                merge = None
            full = prefix+samples+suffix
            target_from = None
            if merge is not None:
                # The route must remain at the new centerline for its full
                # remaining surveyed segment. Merely crossing the stripe is
                # insufficient: the navigation preview may still be turning.
                first_centered = max((j+1 for j,matched in enumerate(centered) if not matched),default=0)
                if first_centered+3 <= len(samples):
                    target_from = begin+first_centered
            found.append(NativeCorridor(*zip(*full),begin,end,None if merge is None else begin+merge,target_from))
        i = max(i,start+1)
    return found
