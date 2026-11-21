"""Original removable sensor carrier for the CARLA 0.9.16 Lincoln MKZ.

Coordinates are vehicle-local meters, x forward, y right, z up. The scanner
and camera apertures match the perception rig. Visual parts are rigid children
with all ray-query and physics collisions disabled.
"""
from __future__ import annotations

from dataclasses import dataclass
import math



MOUNTS = {
    "front": (1.30, 0.0, 1.65, 0.0),
    "left": (1.30, -0.75, 1.65, -70.0),
    "right": (1.30, 0.75, 1.65, 70.0),
    "lidar": (0.0, 0.0, 2.30, 0.0),
    "radar": (2.0, 0.0, 0.90, 0.0),
}

WHITE = "/Engine/BasicShapes/Cylinder"
ROUND = "/Engine/BasicShapes/Sphere"
SMOKED = "/Game/Carla/Static/Pole/SM_PoleCylinder"
FLANGE = "/Game/Carla/Static/Dynamic/Bar-Restaurant/SM_Plate"
STRUT = "/Game/Carla/Static/Pole/SM_Lights/SM_Bollard01"


@dataclass(frozen=True)
class Part:
    name: str
    mesh: str
    xyz: tuple
    scale: float
    pitch: float = 0.0
    yaw: float = 0.0


def strut(name, start, end):
    """The native strut extends 0.682532 m vertically from its mounting foot."""
    dx, dy, dz = (b - a for a, b in zip(start, end))
    length = math.sqrt(dx*dx + dy*dy + dz*dz)
    return Part(name, STRUT, start, length / 0.682532,
                -math.degrees(math.atan2(math.hypot(dx, dy), dz)),
                math.degrees(math.atan2(dy, dx)))


def parts():
    yield Part("roof seal", FLANGE, (0, 0, 1.47), 3.2)
    yield Part("pedestal", WHITE, (0, 0, 1.59), .25)
    yield Part("scanner lower case", WHITE, (0, 0, 1.925), .435)
    yield Part("lower gasket", FLANGE, (0, 0, 2.137), 2.49)
    yield Part("scanner optical drum", SMOKED, (0, 0, 2.27), .27)
    yield Part("scanner rain cap", FLANGE, (0, 0, 2.395), 2.28)
    for name in ("left", "right", "front"):
        x, y, z, yaw = MOUNTS[name]
        ux, uy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        back = (x - .19*ux, y - .19*uy, z)
        foot = (.46, .47 if y > 0 else -.47 if y < 0 else 0, 1.44)
        yield strut(name + " support", foot, back)
        yield Part(name + " camera fairing", ROUND, back, .24, yaw=yaw)
        yield Part(name + " lens barrel", SMOKED,
                   (x - .078*ux, y - .078*uy, z), .115, 90, yaw)
    # The radar measures through the Lincoln's original front fascia.
    yield Part("internal radar housing", SMOKED, (1.9, 0, .90), .14, 90)


PARTS = tuple(parts())


class BodyKit:
    def __init__(self, world, ego):
        import carla

        self.actors = []
        try:
            for part in PARTS:
                bp = world.get_blueprint_library().find("static.prop.mesh")
                bp.set_attribute("mesh_path", part.mesh + "." + part.mesh.rsplit("/", 1)[1])
                bp.set_attribute("scale", str(part.scale))
                # Positive spawn mass makes the UE mesh movable so rigid parenting works.
                # Physics is turned off before the next synchronous world step.
                bp.set_attribute("mass", "0.01")
                bp.set_attribute("role_name", "robotaxi_" + part.name.replace(" ", "_"))
                pose = carla.Transform(carla.Location(*part.xyz),
                                       carla.Rotation(pitch=part.pitch, yaw=part.yaw))
                actor = world.spawn_actor(bp, pose, attach_to=ego,
                                          attachment_type=carla.AttachmentType.Rigid)
                self.actors.append(actor)
                actor.set_simulate_physics(False)
                actor.set_collisions(False)
        except BaseException:
            self.close()
            raise

    def close(self):
        for actor in reversed(self.actors):
            if actor.is_alive:
                actor.destroy()
        self.actors.clear()


