"""Measure the striped Town03 channelizer using synchronized RGB and obstacle lidar.

All range inputs to detect_cones are in the ego frame, +X forward, +Y right,
+Z up. The RGB classifier looks for the three orange sections separated by two
neutral reflective bands. Its fitted box excludes the laterally flared black foot.
"""

import math

import cv2
import numpy as np

from .camera_geometry import CAMERAS


def _stripes(rgb, threshold):
    r, g, b = cv2.split(rgb.astype(np.int16))
    red = (r - g >= threshold) & (r - b >= threshold + 2) & ((r - g) * 10 >= r) & (g - b <= (r - g) * 0.9 + 4)
    h, w = red.shape
    horizon = int(h * 0.43)
    red[:horizon] = False
    count, _, stats, _ = cv2.connectedComponentsWithStats(red.astype(np.uint8), connectivity=8)
    stripes = []
    for x, y, width, height, area in stats[1:count]:
        if (
            width >= 3
            and height >= 2
            and 0.62 <= width / height <= 8
            and width <= w * 0.45
            and area >= 0.32 * width * height
        ):
            stripes.append((int(x), int(y), int(width), int(height)))
    return stripes


def _segments(rgb, threshold):
    stripes = _stripes(rgb, threshold)
    proposals = []
    for x, y, w, h in stripes:  # middle orange band
        mid = x + w * 0.5
        neighbors = [
            (bx, by, bw, bh)
            for bx, by, bw, bh in stripes
            if 0.48 * w <= bw <= 1.95 * w and abs(mid - bx - bw * 0.5) <= 0.38 * w + 1
        ]
        above = [s for s in neighbors if s[1] < y and -2 <= y - s[1] - s[3] <= 2.4 * h + 2]
        below = [s for s in neighbors if s[1] > y and -2 <= s[1] - y - h <= 2.4 * h + 2]
        for top in above:
            for bottom in below:
                tx, ty, tw, th = top
                bx, by, bw, bh = bottom
                bottom_y = by + bh
                height = bottom_y - ty
                width = max(w, tw, bw)
                if not (
                    0.26 <= width / height <= 0.85
                    and th >= 0.55 * h
                    and bh >= 0.65 * h
                    and 0.68 <= bw / tw <= 2.1
                    and 0.26 <= (y + 0.5 * h - ty) / height <= 0.69
                ):
                    continue
                # Both intervals must be lighter, neutral and span the center.
                x1 = max(tx, x, bx)
                x2 = min(tx + tw, x + w, bx + bw)
                if x2 - x1 < 0.46 * w:
                    continue
                a = x1 + (x2 - x1) // 4
                b = max(a + 1, x2 - (x2 - x1) // 4)
                levels = []
                for cy in (
                    ty + th * 0.62,
                    (ty + th + y) * 0.5,
                    y + h * 0.5,
                    (y + h + by) * 0.5,
                    by + bh * 0.45,
                ):
                    levels.append(np.median(rgb[int(cy), a:b].astype(float), axis=0))
                gains = []
                for i in (1, 3):
                    ref = (levels[i - 1] + levels[i + 1]) * 0.5
                    gain = levels[i][1] - ref[1]
                    color_loss = (ref[0] - ref[1]) - (levels[i][0] - levels[i][1])
                    if gain < 4 or color_loss < 8:
                        break
                    gains.append(color_loss)
                if len(gains) != 2:
                    continue
                left, right = min(tx, x, bx), max(tx + tw, x + w, bx + bw)
                # The top handle and base occupy small fringes outside the orange body.
                box = (
                    left,
                    max(0, ty - round(0.035 * height)),
                    right,
                    min(rgb.shape[0], bottom_y + round(0.08 * height)),
                )
                score = min(0.99, 0.65 + min(gains) / 150.0)
                proposals.append((box, score))
    return proposals


def _appearance(rgb):
    return _segments(rgb, 11) + _segments(rgb, 20)


def _axes(points, camera):
    angle = math.radians(camera.yaw)
    dx, dy = points[:, 0] - camera.x, points[:, 1] - camera.y
    depth = dx * math.cos(angle) + dy * math.sin(angle)
    lateral = dy * math.cos(angle) - dx * math.sin(angle)
    return depth, lateral


def _range(box, camera, xyz, xy):
    x1, y1, x2, y2 = box
    groups = []
    # On archived drives XY are independently downsampled, road-filtered hits.
    # A heightless hit must agree with both the visible foot's ground intersection
    # and the solid body's apparent width and height. Never invent a return height.
    for points, has_z in ((xyz, True), (xy, False)):
        if not len(points):
            continue
        depth, lateral = _axes(points, camera)
        with np.errstate(divide="ignore", invalid="ignore"):
            u = 0.5 + camera.fx * lateral / depth
            v = 0.5 - camera.fy * (points[:, 2] - camera.z) / depth if has_z else None
        valid = (depth > 1.0) & (depth < 29.0) & (u >= x1) & (u <= x2)
        height = (y2 - y1) * depth / camera.fy
        width = (x2 - x1) * depth / camera.fx
        foot_z = camera.z - (y2 - 0.5) * depth / camera.fy
        valid &= (height > 0.84) & (height < 1.67) & (width > 0.26) & (width < 1.16) & (abs(foot_z) < 0.48)
        if has_z:
            valid &= (v >= y1) & (v <= y2) & (points[:, 2] > 0.15) & (points[:, 2] < 1.65)
        groups.extend((float(d), bool(has_z), p) for d, p in zip(depth[valid], points[valid]))
    if not groups:
        return None
    groups.sort(key=lambda item: item[0])
    clusters = [[]]
    for item in groups:
        if clusters[-1] and item[0] - clusters[-1][0][0] > 0.75:
            clusters.append([])
        clusters[-1].append(item)

    def floor_error(group):
        return abs(camera.z - (y2 - 0.5) * np.median([g[0] for g in group]) / camera.fy)

    best = min(clusters, key=lambda g: (-sum(h[1] for h in g), -len(g), floor_error(g)))
    # Reject a lone heightless hit, which might be an unrelated surface in the same bearing.
    xyz_hits = [p for _, has_z, p in best if has_z]
    # The historical XY and XYZ samples draw from the same sweep. Count a ray
    # only once when both saved representations identify the same rounded hit.
    occupied = {tuple(np.round(p[:2], 1)) for p in xyz_hits}
    xy_hits = []
    for _, has_z, p in best:
        key = tuple(np.round(p[:2], 1))
        if not has_z and key not in occupied:
            occupied.add(key)
            xy_hits.append(p)
    if not xyz_hits and len(xy_hits) < 2:
        return None
    depth = float(np.median([g[0] for g in best]))
    width = (x2 - x1) * depth / camera.fx
    # Samples strike the near shell of the rounded body; infer the center along its optical ray.
    center_depth = depth + width * 0.36
    lateral = ((x1 + x2) * 0.5 - 0.5) * center_depth / camera.fx
    yaw = math.radians(camera.yaw)
    height = (y2 - y1) * center_depth / camera.fy
    return {
        "x": round(camera.x + center_depth * math.cos(yaw) - lateral * math.sin(yaw), 3),
        "y": round(camera.y + center_depth * math.sin(yaw) + lateral * math.cos(yaw), 3),
        "z": round(camera.z - ((y1 + y2) * 0.5 - 0.5) * center_depth / camera.fy, 3),
        "width": round(width, 3),
        "length": round(width, 3),
        "height": round(height, 3),
        "lidar_hits": len(xyz_hits) + len(xy_hits),
        "lidar_xyz_hits": len(xyz_hits),
        "surface_xyz": [p.astype(float).round(3).tolist() for p in xyz_hits],
        "surface_xy": [p.astype(float).round(1).tolist() for p in xy_hits],
        "range_source": "projected_xyz" if xyz_hits else "ground_gated_xy",
    }


def detect_cones(rgb, camera_name, ego_xyz, obstacle_xy=()):
    """Return current-exposure cone measurements only. No map, history or actor inputs."""
    camera = CAMERAS[camera_name]
    h, w = rgb.shape[:2]
    if (w, h) != (camera.width, camera.height):
        raise ValueError(f"Unexpected camera dimensions for {camera_name}: {w}x{h}")
    proposals = _appearance(rgb)
    if not proposals:
        return []
    xyz = np.asarray(ego_xyz).reshape(-1, 3)
    xy = np.asarray(obstacle_xy).reshape(-1, 2)
    found = []
    for rect, score in sorted(proposals, key=lambda p: -p[1]):
        box = [round(value / scale, 6) for value, scale in zip(rect, (w, h, w, h))]
        measure = _range(box, camera, xyz, xy)
        if measure and all(math.hypot(measure["x"] - p["x"], measure["y"] - p["y"]) > 0.7 for p in found):
            found.append(
                dict(
                    label="traffic_cone",
                    camera=camera_name,
                    box=box,
                    confidence=round(score, 3),
                    sources=["rgb", "lidar"],
                    **measure,
                )
            )
    return found


def detect_views(images, ego_xyz, obstacle_xy=()):
    """Keep the strongest current camera measurement when their fields overlap."""
    observations = [
        obs for name, image in images.items() for obs in detect_cones(image, name, ego_xyz, obstacle_xy)
    ]
    found = []
    for obs in sorted(observations, key=lambda r: (-r["lidar_xyz_hits"], -r["confidence"])):
        if all(math.hypot(obs["x"] - p["x"], obs["y"] - p["y"]) > 0.7 for p in found):
            found.append(obs)
    return found
