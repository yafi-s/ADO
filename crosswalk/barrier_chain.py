"""A short sagging chain between independently measured low posts.

The recognizer proposes pairs from roof-lidar columns. A visible RGB curve and
a range return on the suspended span must agree before it names a chain barrier.
"""
import math

import cv2
import numpy as np


def post_candidates(points):
    # An upright needs returns at two different heights within its narrow footprint.
    step = .12
    side = 352
    ij = np.floor(points[:, :2] / step).astype(int) + side // 2
    low = ((ij >= 1) & (ij < side - 1)).all(axis=1) & (points[:, 2] < 1.10)
    idx = np.flatnonzero(low)
    if len(idx) < 5:
        return []
    rows, cols = ij[idx].T
    top = np.zeros((side, side), np.float32)
    bottom = np.full_like(top, 2.)
    np.maximum.at(top, (rows, cols), points[idx, 2])
    np.minimum.at(bottom, (rows, cols), points[idx, 2])
    kernel = np.ones((3, 3), np.uint8)
    vertical = (top > 0) & (cv2.dilate(top, kernel) - cv2.erode(bottom, kernel) > .10)
    _, regions, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(vertical.astype(np.uint8), kernel))
    labels = regions[rows, cols]
    result = []
    for label, (_, _, w, h, _) in enumerate(stats[1:], 1):
        if w > 8 or h > 8:
            continue
        hits = idx[labels == label]
        body = points[hits]
        if len(body) < 2 or np.ptp(body[:, 2]) < .10:
            continue
        center = np.median(body[:, :2], axis=0)
        if np.max(np.linalg.norm(body[:, :2] - center, axis=1)) > .38:
            continue
        center += .075 * center / max(np.linalg.norm(center), .1)
        result.append((center, hits))
    return result


def project(xyz, camera):
    points = xyz - np.array([camera.x, camera.y, camera.z])
    c, s = math.cos(math.radians(camera.yaw)), math.sin(math.radians(camera.yaw))
    depth = c*points[..., 0] + s*points[..., 1]
    side = -s*points[..., 0] + c*points[..., 1]
    return (np.stack((camera.width*(.5 + camera.fx*side/depth),
                      camera.height*(.5 - camera.fy*points[..., 2]/depth)), axis=-1), depth)


def curve_score(gray, camera, start, end, span_pts):
    axis = end - start
    fraction = ((span_pts[:, :2]-start) @ axis) / np.dot(axis, axis)
    # One laser hit constrains the curve's altitude. Multiple hits may occur
    # on the pavement behind it, so RGB must support most of the inner span.
    fractions = np.linspace(.14, .86, 27)
    xy = start + fractions[:, None]*axis
    best = None
    for sag in (.19, .25, .31):
        apex = span_pts[:, 2] + 4*sag*fraction*(1-fraction)
        for top in np.unique(np.round(apex[:, None] + [-.03, 0., .03], 2)):
            if not .30 < top < 1.05:
                continue
            height = top - 4*sag*fractions*(1-fractions)
            coords, depth = project(np.column_stack((xy, height)), camera)
            pix = camera.fx*camera.width / depth.mean()
            if ((coords < [2, 5]) | (coords >= [camera.width-2, camera.height-6])).any():
                continue
            delta = max(1.7, .045*pix)
            # Compare the predicted chain to both neighbouring pixels on
            # the same background. A lone diagonal edge cannot satisfy both.
            shifts = np.array([-delta, -.65, 0., .65, delta])[:, None]
            xs = np.broadcast_to(coords[:, 0], (5, len(coords))).astype(np.float32)
            ys = (coords[:, 1] + shifts).astype(np.float32)
            level = cv2.remap(gray, xs, ys, cv2.INTER_LINEAR).astype(float)
            dark = np.minimum(level[0],level[-1]) - level[1:4].min(axis=0)
            bright = level[1:4].max(axis=0) - np.maximum(level[0],level[-1])
            ridge = np.maximum(dark, bright)
            # Require evidence distributed across the arc, away from the two posts.
            contrast = float(np.mean(np.clip(ridge, -12., 24.)))
            coverage = float(np.mean(ridge >= 2.))
            score = min(contrast / 4., 1.) * coverage
            if best is None or score > best[0]:
                best = (score, float(top), sag)
    return best


def detect_chain(image, camera, points):
    if len(points) < 5:
        return []
    # At longer ranges a chain wire is subpixel and two rings rarely sample
    # either narrow post. Limit recognition to the near field of this rig.
    points_range = np.hypot(points[:, 0]-camera.x, points[:, 1]-camera.y)
    near = np.flatnonzero(points_range <= 11.)
    points = points[near]
    candidates = post_candidates(points)
    if len(candidates) < 2:
        return []
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    result = []
    for i, (start, first) in enumerate(candidates):
        for end, last in candidates[i+1:]:
            axis = end - start
            length = np.linalg.norm(axis)
            if not 1.20 <= length <= 1.68:
                continue
            along = (points[:, :2]-start) @ axis / length**2
            inside = np.flatnonzero((along >= .17) & (along <= .83) & (points[:, 2] < 1.))
            if not len(inside):
                continue
            offset = points[inside, :2] - (start + along[inside, None]*axis)
            hits = inside[np.linalg.norm(offset, axis=1) < .115]
            if not len(hits):
                continue
            best = curve_score(gray, camera, start, end, points[hits])
            if best is None or best[0] < .19:
                continue
            score, top, sag = best
            arc_z = top - 4*sag*along[hits]*(1-along[hits])
            middle = hits[np.abs(points[hits, 2]-arc_z) < .06]
            if not len(middle):
                continue
            # Use only post and span returns belonging to this geometry. The
            # tracker will exclude hits shared with another detected object.
            corners = np.array([[xy[0], xy[1], z] for xy in (start,end) for z in (top+.12,top-.59)])
            uv, depth = project(corners, camera)
            if depth.min() < 1. or np.ptp(uv[:, 0]) < 12:
                continue
            lo, hi = uv.min(axis=0), uv.max(axis=0)
            margin = .11 * camera.width * camera.fx / depth.mean()
            box = np.clip(np.r_[lo-[margin,1.],hi+[margin,1.]] /
                          [camera.width,camera.height,camera.width,camera.height], 0, 1)
            result.append(dict(label="barrier", barrier_subtype="chain", detector="ado-suspended-chain-v1",
                               inference_layout="single", score=float(score), box=box.tolist(),
                               chain_support={"posts": [len(first), len(last)], "span_hits": len(middle)},
                               _groups=[near[first],near[middle],near[last]],
                               _endpoints=[start, end]))
    return result
