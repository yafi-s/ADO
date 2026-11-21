"""Rigid visual sensor carrier for the CARLA 1.0 Lincoln.

All poses use vehicle-local metres, +X forward, +Y right, +Z up. The stock
release can attach its cooked meshes to a moving vehicle. The corresponding
detailed Blender assembly uses the same optical registration.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


MOUNTS = {
    "front": (1.30, 0.00, 1.65, 0.0),
    "left": (1.30, -0.75, 1.65, -70.0),
    "right": (1.30, 0.75, 1.65, 70.0),
    "lidar": (0.00, 0.00, 2.30, 0.0),
    "radar": (2.00, 0.00, 0.90, 0.0),
}
PEARL = (230, 237, 238)
GRAPHITE = (52, 62, 68)
OPTIC = (21, 35, 43)


@dataclass(frozen=True)
class Mesh:
    path: str
    # Local bounding-box center and full dimensions at unit scale. CARLA 1.0
    # ships offset pivots on several meshes, including the scanner hub.
    center: tuple[float, float, float]
    size: tuple[float, float, float]


@dataclass(frozen=True)
class Part:
    name: str
    mesh: Mesh
    center: tuple[float, float, float]
    color: tuple[int, int, int]
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def pose(self, carla):
        pitch, yaw, roll = self.rotation
        rotation = carla.Rotation(pitch=pitch, yaw=yaw, roll=roll)
        basis = carla.Transform(carla.Location(), rotation)
        offset = basis.transform_vector(carla.Vector3D(*self.mesh.center))
        return carla.Transform(carla.Location(*self.center) - offset, rotation)


TRAY = Mesh(
    "/Game/Carla/Static/Dynamic/00_LegacyAssets/PedestrianProps/SM_Briefcase",
    (-3.588676e-05, 0.02178249, -0.2161659), (0.5433538, 0.1811764, 0.4328505),
)
CAP = Mesh(
    "/Game/Carla/Static/Road/RoadProps/RoadManhole",
    (-1.430511e-07, -1.621246e-07, 0.0), (0.2043753, 0.214796, 0.008210189),
)
STRUT = Mesh(
    "/Game/Carla/Static/Static/SM_Bollard_Grey",
    (0.0, 0.0, 0.3894249), (0.07906505, 0.07890905, 0.7788495),
)
HOUSING = Mesh(
    "/Game/Carla/Static/Dynamic/00_LegacyAssets/PedestrianProps/SM_bikehelmet",
    (-6.382942e-05, -0.01704701, -0.05041412), (0.1855496, 0.2770471, 0.1712976),
)
WINDOW = Mesh(
    "/Game/Carla/Static/Dynamic/00_LegacyAssets/PedestrianProps/SM_Mobile",
    (0.0, 0.003630543, 0.08320887), (0.08048341, 0.007261085, 0.1664177),
)
LENS = Mesh(
    "/Game/Carla/Static/Dynamic/00_LegacyAssets/Trash/Wastes/SM_ColaCan",
    (-0.003598429, 0.0, 0.05218715), (0.06817007, 0.07167824, 0.1095489),
)
RADAR = Mesh(
    "/Game/Carla/Static/Car/4Wheeled/TeslaModel3UE58/Meshes/SM_Model3_CaliperFL",
    (-0.1377954, -0.02785823, 0.001148329), (0.1379056, 0.1269864, 0.344945),
)


def _boom(name, start, end):
    dx, dy, dz = (b - a for a, b in zip(start, end))
    length = math.sqrt(dx * dx + dy * dy + dz * dz)
    yaw = math.degrees(math.atan2(dy, dx))
    pitch = -math.degrees(math.atan2(math.hypot(dx, dy), dz))
    # Align the upper end of each fixed-length rail with its camera pod.
    center = tuple(b - (b-a) / length * STRUT.size[2] / 2 for a, b in zip(start, end))
    return Part(name, STRUT, center, GRAPHITE, (pitch, yaw, 0))


def _parts():
    yield Part("service tray", TRAY, (0, 0, 1.51), PEARL, (0, 0, 90))
    for y in (-0.09, 0.09):
        yield Part("left scanner pillar" if y < 0 else "right scanner pillar",
                   STRUT, (0, y, 1.84), PEARL, (0, 0, -8 if y < 0 else 8))
    x, y, z, _ = MOUNTS["lidar"]
    # Nine pairs of physical optical panels enclose the scanner. Overlapping
    # circular aluminum plates close its top and underside.
    for i in range(9):
        theta = math.tau * i / 9
        yaw = math.degrees(theta)
        for row, dz in enumerate((-0.039, 0.039)):
            yield Part(f"scanner window {i+1}-{row+1}", WINDOW,
                       (x + .215*math.cos(theta), y + .215*math.sin(theta), z+dz),
                       OPTIC, (90, yaw+90, 0))
    for side, dz in (("cap", .083), ("base", -.083)):
        yield Part("scanner " + side + " center", CAP, (x,y,z+dz), PEARL)
        for i in range(6):
            theta = math.tau * i / 6
            yield Part(f"scanner {side} {i+1}", CAP,
                       (x+.131*math.cos(theta), y+.131*math.sin(theta), z+dz), PEARL)
    for name in ("front", "left", "right"):
        x, y, z, yaw = MOUNTS[name]
        radians = math.radians(yaw)
        ux, uy = math.cos(radians), math.sin(radians)

        def behind(lens_distance):
            return (x - lens_distance * ux, y - lens_distance * uy, z)

        foot = (0.42, math.copysign(0.40, y) if y else 0, 1.48)
        yield _boom(name + " header blade", foot, behind(0.205))
        yield Part(name + " optical housing", HOUSING, behind(.169), PEARL, (0, yaw - 90, 0))
        yield Part(name + " optical window", WINDOW, behind(.025), OPTIC, (90, yaw + 90, 0))
        yield Part(name + " lens barrel", LENS, behind(.008 + LENS.size[2] / 2), GRAPHITE, (90, yaw, 0))
    x, y, z, yaw = MOUNTS["radar"]
    yield Part("concealed radar", RADAR, (x - .12, y, z), GRAPHITE, (0, yaw, 90))


PARTS = tuple(_parts())


class BodyKit:
    def __init__(self, world, ego):
        import carla

        self.actors = []
        blueprint = world.get_blueprint_library().find("static.prop.briefcase")
        textures = {}
        render = not world.get_settings().no_rendering_mode
        try:
            for part in PARTS:
                path = part.mesh.path
                blueprint.set_attribute("mesh_path", path + "." + path.rsplit("/", 1)[1])
                # The registered prop factory supplies a movable, kinematic
                # root. Generic static.prop.mesh starts static or simulated,
                # either of which loses physical parenting in the 1.0 build.
                blueprint.set_attribute("role_name", "robotaxi_" + part.name.replace(" ", "_"))
                actor = world.spawn_actor(blueprint, part.pose(carla), attach_to=ego,
                                          attachment_type=carla.AttachmentType.Rigid)
                self.actors.append(actor)
                actor.set_simulate_physics(False)
                actor.set_collisions(False)
                if not render:
                    continue
                if part.color not in textures:
                    texture = carla.TextureColor(2, 2)
                    for x in range(2):
                        for y in range(2):
                            texture.set(x, y, carla.Color(*part.color))
                    textures[part.color] = texture
                actor.apply_texture(carla.MaterialParameter.Diffuse, textures[part.color])
        except BaseException:
            self.close()
            raise

    def close(self):
        for actor in reversed(self.actors):
            if actor.is_alive:
                actor.destroy()
        self.actors.clear()
