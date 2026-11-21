"""Level CARLA pinhole cameras, registered to vehicle-local +X forward, +Y right, +Z up."""

from dataclasses import dataclass
import math

from .robotaxi import MOUNTS


@dataclass(frozen=True)
class Camera:
    x: float
    y: float
    z: float
    yaw: float
    width: int = 320
    height: int = 180
    hfov: float = 90.

    @property
    def fx(self):
        """Focal length divided by image width."""
        return .5 / math.tan(math.radians(self.hfov) / 2)

    @property
    def fy(self):
        return self.fx * self.width / self.height


CAMERAS = {
    "front": Camera(*MOUNTS["front"], 640, 360),
    "left": Camera(*MOUNTS["left"]),
    "right": Camera(*MOUNTS["right"]),
    "rear_left": Camera(-1.10, -.70, 1.72, -145.),
    "rear_right": Camera(-1.10, .70, 1.72, 145.),
}
