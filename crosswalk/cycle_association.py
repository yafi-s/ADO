"""Associate a camera bicycle and its rider while preserving the source of range.

Only elevated lidar returns outside a padded rider box can establish separately
ranged bicycle structure. Returns in the overlap are reported as shared range.
"""
import math

import numpy as np


def _aligned(p, b):
    pw, ph = p[2]-p[0], p[3]-p[1]
    bw, bh = b[2]-b[0], b[3]-b[1]
    if min(pw, ph, bw, bh) <= 0:
        return False
    overlap = min(p[2], b[2])-max(p[0], b[0])
    return (abs(p[0]+p[2]-b[0]-b[2]) < .85*max(pw,bw)
            and overlap >= .65*min(pw,bw) and b[1] > p[1]+.10*ph
            and b[1] < p[3] and b[3] >= p[3]-.08*ph and b[3] <= p[3]+.75*ph
            and b[1]+b[3] > p[1]+p[3]+.22*ph)


def pairs(detections):
    people = [d for d in detections if d["label"] == "person"]
    bikes = [d for d in detections if d["label"] == "bicycle"]
    candidates = [(p,b) for b in bikes for p in people if _aligned(p["box"], b["box"])]
    return [(p,b) for p,b in candidates if sum(a is p for a,_ in candidates)==1
            and sum(c is b for _,c in candidates)==1]


def _pixels(u,v,box,px=0.,py=0.):
    a,b,c,d = box
    return (u>=a-px)&(u<=c+px)&(v>=b-py)&(v<=d+py)


def pair_surface(points, depths, u, v, owner_masks, person, bike, cam):
    """Require a contiguous, compact body with upper and lower returns at one range."""
    p,b = person["box"],bike["box"]
    pm,bm = _pixels(u,v,p),_pixels(u,v,b)
    clear = pm|bm
    for other,mask in owner_masks:
        if other is not person and other is not bike:
            clear &= ~mask
    order = np.flatnonzero(clear)
    order = order[np.argsort(depths[order])]
    viable = []
    for group in np.split(order,np.flatnonzero(np.diff(depths[order])>.65)+1):
        if len(group)<4 or np.ptp(depths[group])>2.2:
            continue
        depth = np.median(depths[group])
        if not .85 < (max(p[3],b[3])-p[1])*depth/cam.fy < 2.45:
            continue
        if (max(p[2],b[2])-min(p[0],b[0]))*depth/cam.fx>2.35:
            continue
        z = points[group,2]
        upper = pm[group] & (v[group] < p[1]+.48*(p[3]-p[1])) & (z>.75)
        lower = bm[group] & (z>.22) & (z<.95)
        if upper.sum()<2 or lower.sum()<2 or np.median(z[upper])-np.median(z[lower])<.32:
            continue
        # A native rider box can envelop the wheels. Such hits are shared, even
        # when their elevation resembles a wheel. Allow a pixel of edge error.
        exclusive = lower & ~_pixels(u[group],v[group],p,1/cam.width,1/cam.height)
        wheel = group[exclusive]
        if len(wheel)<3 or np.max(np.ptp(points[wheel],axis=0))<.10:
            wheel = np.empty(0,dtype=int)
        viable.append((group, wheel, np.median(points[group[upper],:2],axis=0)))
    return viable[0] if len(viable)==1 else None


class CycleAssociator:
    """A history belongs to one unambiguous pair of optical identities in one camera."""

    def __init__(self):
        self.paths = {}

    def observe(self, view, person, bike, support, pts, depths, camera, pose, now):
        if "image_id" not in person or "image_id" not in bike:
            return None
        group, wheel, upper = support
        self.paths = {k:path for k,path in self.paths.items() if now-path[-1][0]<=.46}
        key = (view, person["image_id"], bike["image_id"])
        path = self.paths.setdefault(key, [])
        yaw = math.radians(pose[2])
        rotation = np.array([[math.cos(yaw),-math.sin(yaw)], [math.sin(yaw),math.cos(yaw)]])
        origin = np.asarray(pose[:2])
        upper_world = rotation @ upper + origin
        # Intersect the optical bicycle centre ray with the measured rider depth.
        # Its displacement across the sightline must agree with the upper body.
        angle = math.radians(camera.yaw)
        axis = np.array([math.cos(angle), math.sin(angle)])
        side = np.array([-axis[1],axis[0]])
        midpoint = (bike["box"][0]+bike["box"][2])*.5
        lateral = (midpoint-.5)/camera.fx
        axle = rotation @ (np.array([camera.x,camera.y]) + np.median(depths[group])*
                           (axis+lateral*side)) + origin
        measured_wheel = pts[wheel,:2] @ rotation.T + origin
        path[:] = [q for q in path if now-q[0]<=1.]
        path.append((now,upper_world,axle,bike["score"],measured_wheel))
        if len(path)<3 or now-path[0][0]<.19 or max(q[3] for q in path)<.53:
            return None
        delta = upper_world-path[0][1]
        side_world = rotation @ side
        disagreement = [abs(((upper_world-q[1])-(axle-q[2])) @ side_world) for q in path[:-1]]
        if np.linalg.norm(delta)<.35 or np.median(disagreement)>.30:
            return None
        # A wheel can be discontinuous. Match actual points after translating
        # earlier wheel-only points by rider displacement, instead of assuming
        # each sweep sampled the same wheel or using purely apparent pixel flow.
        separate = False
        for _, earlier_upper, _, _, earlier_wheel in path[:-1]:
            motion = upper_world-earlier_upper
            if len(wheel)<3 or len(earlier_wheel)<3 or np.linalg.norm(motion)<.23:
                continue
            residual = measured_wheel[:,None,:]-(earlier_wheel+motion)[None,:,:]
            moving_error = np.linalg.norm(residual,axis=2).min(axis=1)
            stationary_error = np.linalg.norm(measured_wheel[:,None,:]-earlier_wheel[None,:,:],axis=2).min(axis=1)
            if (np.linalg.norm(motion)>=.45 and np.median(stationary_error)<.10 and
                    np.median(moving_error)>np.median(stationary_error)+.25):
                return None
            if (moving_error<.28).sum()>=2:
                separate = True
        return dict(camera=view, bicycle_box=bike["box"], rider_box=person["box"],
                    bicycle_image_id=bike["image_id"], rider_image_id=person["image_id"],
                    range_source="bicycle-exclusive lidar" if separate else "shared rider/bicycle lidar",
                    surface_hits=len(group), bicycle_exclusive_hits=len(wheel) if separate else 0,
                    bicycle_surface_xyz=pts[wheel].round(3).tolist()[:8] if separate else [],
                    paired_observations=len(path), age_s=0.)
