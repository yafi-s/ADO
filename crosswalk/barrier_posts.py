"""Verify the visible post at each end of an independently ranged chain proposal."""
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .barrier_chain import project

MODEL = Path(__file__).resolve().parents[1] / "models/barrier-post-appearance.npz"
THRESHOLD = .34
MIN_BOARD_BREADTH_M = .30


def post_tiles(rgb, camera, candidate):
    """Project a fixed physical window around each measured metal bollard."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    ends = np.asarray(candidate["chain_ends_xy"])
    xy, depths = project(np.column_stack((ends, np.zeros(2))), camera)
    # The raw box includes a pixel of margin above the higher projected post.
    top = camera.z - (candidate["box"][1] + 1/camera.height - .5)*max(depths)/camera.fy
    tiles = []
    for depth, pixel_x in zip(depths, xy[:, 0]):
        scale = camera.width*camera.fx/depth
        u = pixel_x + np.linspace(-.26, .26, 24)*scale
        heights = np.linspace(top+.12, top-.85, 40)
        v = camera.height*(.5-camera.fy*(heights-camera.z)/depth)
        x, y = np.meshgrid(u, v)
        tiles.append(cv2.remap(gray, x.astype(np.float32), y.astype(np.float32), cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE))
    return tiles


def gradients(tile):
    smooth = cv2.GaussianBlur(tile.astype(np.float32), (3, 3), .7)
    return np.dstack((cv2.Sobel(smooth, -1, 1, 0, ksize=3), cv2.Sobel(smooth, -1, 0, 1, ksize=3)))


@lru_cache(maxsize=1)
def templates():
    with np.load(MODEL) as packet:
        tiles = packet["posts"]
    # The illuminated side depends on the source direction. The two sides,
    # flat cap and flared foot keep their signs under a translation.
    return [gradients(view)[4:-4, 3:-3] for tile in tiles for view in (tile, np.fliplr(tile))]


def review(rgb, camera, candidate):
    result = []
    for tile in post_tiles(rgb, camera, candidate):
        sample = gradients(tile)
        score = max(float(cv2.matchTemplate(sample, ref, cv2.TM_CCORR_NORMED).max()) for ref in templates())
        result.append(score)
    return dict(confirmed=min(result) >= THRESHOLD, post_scores=[round(s, 6) for s in result])

class ChainHistory:
    """Require a repeated pair of post measurements unless one laser scan is decisive."""

    def __init__(self):
        self.previous = []

    def qualify(self, candidate, appearance, ego_xy_yaw, now, exclusive=None):
        x, y, yaw = ego_xy_yaw
        angle = np.deg2rad(yaw)
        cosine, sine = np.cos(angle), np.sin(angle)
        rotation = np.array([[cosine, sine], [-sine, cosine]])
        ends = np.asarray(candidate["chain_ends_xy"]) @ rotation + [x, y]
        self.previous = [(stamp, prior) for stamp, prior in self.previous
                         if 0 <= now - stamp <= .45 + 1e-8]
        measured = candidate.get("range_supported", False) and candidate.get("track_id") is not None
        raw = candidate["chain_support"]
        strong = bool(measured and exclusive
                      and min(*raw["posts"], *exclusive["posts"]) >= 3
                      and min(raw["span_hits"], exclusive["span_hits"]) >= 2)
        repeat = False
        if measured and appearance["confirmed"]:
            for stamp, prior in self.previous:
                # Synchronized cameras share the same laser sweep.
                if now - stamp < .05:
                    continue
                # Two identical posts have an axis but no distinguished first end.
                same_order = np.linalg.norm(ends - prior, axis=1).max()
                other_order = np.linalg.norm(ends - prior[::-1], axis=1).max()
                if min(same_order, other_order) <= .32:
                    repeat = True
                    break
            self.previous.append((now, ends))
        return dict(confirmed=bool(measured and appearance["confirmed"] and (strong or repeat)),
                    appearance_confirmed=appearance["confirmed"], post_scores=appearance["post_scores"],
                    strong_range=strong, repeat_supported=repeat, exclusive_range=bool(measured))


def board_grade(surface):
    """A narrow pole in a stripe-like image box cannot establish a broad board."""
    xy = np.asarray(surface, dtype=float).reshape(-1, 3)[:, :2]
    diameter = float(np.linalg.norm(xy[:, None] - xy, axis=2).max()) if len(xy) >= 2 else 0.
    return dict(confirmed=diameter >= MIN_BOARD_BREADTH_M, measured_diameter_m=round(diameter, 6),
                post_scores=[])


def range_only_decision(surface):
    return dict(board_grade(surface), barrier_subtype="striped_board", range_only=True, observations=[])


def add_observation(frame, view, candidate, grade, **extra):
    source = candidate.get("track_id")
    key = str(source) if source is not None else "untracked"
    track = frame.setdefault(key, dict(confirmed=False, post_scores=[], observations=[]))
    track["barrier_subtype"] = candidate.get("barrier_subtype")
    if grade["post_scores"] and (not track["post_scores"] or min(grade["post_scores"]) > min(track["post_scores"])):
        track["post_scores"] = grade["post_scores"]
    track["confirmed"] |= grade["confirmed"]
    track["observations"].append(dict(view=view, image_id=candidate.get("image_id"), track_id=source,
                                     barrier_subtype=candidate.get("barrier_subtype"),
                                     range_supported=candidate.get("range_supported", False),
                                     box=candidate["box"], **grade, **extra))
