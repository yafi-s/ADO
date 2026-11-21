"""Rebuild the original roof carrier and fascia radar as a meter-scale GLB.

Run Blender in background with --python this_file. Append -- --preview to
also render the detached assembly into .local/robotaxi-evidence/design.png.
"""
from math import cos, pi, radians, sin
from pathlib import Path
import sys

import bpy
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crosswalk.robotaxi import MOUNTS

def material(name, color, metal, roughness):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*color, 1)
    mat.use_nodes = True
    shader = mat.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (*color, 1)
    shader.inputs["Metallic"].default_value = metal
    shader.inputs["Roughness"].default_value = roughness
    return mat


def surface(name, rings, mat):
    """Closed loft with hard planar ends and smooth perimeter."""
    n = len(rings[0])
    verts = [p for ring in rings for p in ring]
    faces = [tuple(reversed(range(n)))]
    for j in range(len(rings)-1):
        for i in range(n):
            a = j * n + i
            b = j * n + (i+1) % n
            faces.append((a, b, b+n, a+n))
    faces.append(tuple(range(len(verts)-n, len(verts))))
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    ob = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(ob)
    ob.data.materials.append(mat)
    for face in mesh.polygons[1:-1]:
        face.use_smooth = True
    return ob


def sign_power(value, power):
    return (1 if value >= 0 else -1) * abs(value)**power


def shell(name, profile, mat, exponent=1.0):
    """Sections are (height, half-length, half-width, forward shift)."""
    rings = []
    for z, rx, ry, x in profile:
        rings.append([(x+rx*sign_power(cos(i*pi/32), exponent),
                       ry*sign_power(sin(i*pi/32), exponent), z) for i in range(64)])
    return surface(name, rings, mat)


def disk(name, profile, mat):
    return shell(name, [(z,r,r,0) for z,r in profile], mat)


def camera_shell(name, center, yaw, profile, mat, exponent=.64, offset_y=0):
    # CARLA yaw is clockwise; Blender +Y points left.
    heading = radians(-yaw)
    c, s = cos(heading), sin(heading)
    x, y, z = center
    rings = []
    for depth, half_width, half_height in profile:
        ring = []
        for i in range(48):
            theta = 2*pi*i/48
            across = half_width*sign_power(cos(theta),exponent) + offset_y
            up = half_height*sign_power(sin(theta),exponent)
            ring.append((x + depth*c - across*s, -y + depth*s + across*c, z+up))
        rings.append(ring)
    return surface(name, rings, mat)


def box(name, center, extents, mat, radius=.01, angle=0):
    bpy.ops.mesh.primitive_cube_add(size=1, location=center)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = extents
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.rotation_euler.z = angle
    modifier = obj.modifiers.new("Edge radius", "BEVEL")
    modifier.width = radius
    modifier.segments = 4
    obj.modifiers.new("Face normals", "WEIGHTED_NORMAL")
    obj.data.materials.append(mat)
    return obj


def blade(name, points, mat):
    # Each cross section of the beam is perpendicular to horizontal travel.
    rings = []
    for x, y, z, w, h in points:
        rings.append([(x, y + w*sign_power(cos(i*pi/12),.65),
                        z + h*sign_power(sin(i*pi/12),.65)) for i in range(24)])
    return surface(name, rings, mat)


def build_carrier():
    """Add the meter-scale carrier to the current Blender collection and return its objects."""
    previous = set(bpy.context.scene.objects)
    pearl = material("Pearl white powder coat", (.77, .82, .84), .27, .255)
    graphite = material("Anodized graphite", (.045, .059, .067), .76, .27)
    seal = material("Weather seal", (.012, .016, .019), .05, .46)
    glass = material("Smoked optical window", (.015, .036, .048), .58, .12)
    lens = material("Blue-green dielectric lens", (.025, .12, .155), .74, .095)
    metal = material("Machined titanium", (.33, .38, .40), .85, .23)
    grille = material("Recessed ventilation slots", (.020, .027, .030), .20, .55)

    # Mounting gasket, low service tray and an aerofoil-shaped scanner pedestal.
    shell("Carrier / bonded perimeter", [(1.435,.50,.328,-.025),(1.457,.52,.346,-.025),
          (1.489,.515,.34,-.025),(1.498,.493,.325,-.025)], seal,.68)
    shell("Carrier / white lower fairing", [(1.484,.50,.331,-.025),(1.504,.52,.344,-.025),
          (1.548,.515,.334,-.025),(1.577,.47,.300,-.022),(1.59,.395,.274,-.015)],pearl,.68)
    shell("Carrier / service seam",[(1.577,.466,.296,-.022),(1.584,.457,.291,-.021),
          (1.589,.444,.285,-.020)],graphite,.68)
    shell("Pylon / aerodynamic neck",[(1.578,.424,.274,-.017),(1.608,.403,.258,-.016),
          (1.664,.32,.196,-.026),(1.775,.235,.137,-.045),(1.948,.169,.112,-.048),
          (2.075,.146,.122,-.020),(2.136,.155,.145,0),(2.158,.153,.149,0)],pearl,.94)
    for side in (-1,1):
        for j in range(6):
            box(f"Carrier / cooling slot {side} {j}",(-.265+.037*j,side*.324,1.536),
                (.017,.011,.032),grille,.004)
    disk("Scanner / isolator",[(2.118,.132),(2.132,.158),(2.151,.163),(2.168,.158)],seal)
    disk("Scanner / lower lip",[(2.151,.153),(2.16,.201),(2.176,.238),(2.192,.25),
         (2.208,.249),(2.216,.232)],graphite)
    disk("Scanner / white structural ring",[(2.187,.243),(2.195,.254),(2.220,.252),
         (2.232,.241),(2.235,.234)],pearl)
    disk("Scanner / smoked 360 degree band",[(2.227,.229),(2.235,.237),(2.247,.24),
         (2.337,.241),(2.35,.237),(2.356,.226)],glass)
    disk("Scanner / upper seal",[(2.348,.229),(2.352,.246),(2.359,.249),(2.364,.244)],seal)
    disk("Scanner / hydrophobic rain cap",[(2.357,.238),(2.362,.250),(2.370,.255),
         (2.381,.252),(2.394,.235),(2.402,.196),(2.406,.098),(2.407,.018)],pearl)

    # Windshield header and camera pods. Optical surfaces sit behind sensor origins.
    for name in ("front","left","right"):
        x,y,z,yaw = MOUNTS[name]
        ym = -y
        if name == "front":
            blade("Header / center blade",[(.36,0,1.514,.098,.045),(.62,0,1.53,.080,.034),
                  (.91,0,1.56,.057,.027),(1.12,0,1.60,.047,.041)],graphite)
        else:
            sign = 1 if ym>0 else -1
            blade("Header / "+name+" blade",[(.235,sign*.28,1.497,.084,.042),
                  (.46,sign*.395,1.525,.070,.038),(.73,sign*.482,1.566,.053,.030),
                  (1.025,sign*.52,1.58,.043,.027),(1.22,sign*.554,1.601,.051,.035)],graphite)
        center=(x,y,z)
        camera_shell(name+" / ivory pod",center,yaw,[(-.283,.031,.032),
           (-.264,.072,.048),(-.21,.104,.066),(-.105,.124,.076),
           (-.059,.124,.074),(-.044,.118,.068)],pearl)
        camera_shell(name+" / front bezel",center,yaw,[(-.052,.119,.069),
           (-.047,.125,.072),(-.035,.125,.072),(-.027,.115,.065)],graphite)
        camera_shell(name+" / recessed optical window",center,yaw,[(-.030,.113,.064),
           (-.025,.112,.063),(-.023,.106,.058)],glass)
        camera_shell(name+" / lens retainer",center,yaw,[(-.024,.035,.035),
           (-.020,.036,.036),(-.017,.032,.032)],metal,1)
        camera_shell(name+" / lens gasket",center,yaw,[(-.018,.030,.030),(-.014,.029,.029)],seal,1)
        camera_shell(name+" / iris",center,yaw,[(-.015,.023,.023),(-.012,.022,.022),
           (-.009,.015,.015),(-.008,.004,.004)],lens,1)
        camera_shell(name+" / service indicator",center,yaw,[(-.026,.008,.008),(-.021,.007,.007)],glass,1,.085)

    radar=MOUNTS["radar"][:3]
    camera_shell("Radar / cast back housing",radar,0,[(-.142,.078,.051),(-.131,.111,.065),
                 (-.050,.115,.069),(-.047,.11,.066)],graphite,.37)
    camera_shell("Radar / fascia radome",radar,0,[(-.053,.109,.063),(-.043,.109,.064),
                 (-.028,.106,.061),(-.022,.099,.056)],seal,.4)

    for name,(x,y,z,yaw) in MOUNTS.items():
        bpy.ops.object.empty_add(type="ARROWS",location=(x,-y,z))
        frame=bpy.context.object
        frame.name = "sensor_origin_"+name
        frame.rotation_euler.z = radians(-yaw)
        frame.empty_display_size=.075
        frame["carla_yaw_degrees"]=yaw
        frame["forward_axis"]="+X"
    return [obj for obj in bpy.context.scene.objects if obj not in previous]


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1
    build_carrier()
    model_path = Path(__file__).with_name("roof-carrier.glb")
    bpy.ops.export_scene.gltf(filepath=str(model_path), export_format="GLB",
                              export_apply=True, export_extras=True)
    # CARLA's editor imports FBX. Merge objects on import for a single static
    # carrier while retaining the material slots. Keep the editable GLB above.
    bpy.ops.export_scene.fbx(filepath=str(model_path.with_name("SM_RobotaxiCarrier.fbx")),
                             object_types={"MESH"}, apply_unit_scale=True,
                             apply_scale_options="FBX_SCALE_UNITS", axis_forward="-Y", axis_up="Z",
                             bake_anim=False, mesh_smooth_type="FACE", use_triangles=True)
    if "--preview" not in sys.argv:
        return

    preview_path = ROOT/".local/robotaxi-evidence/design.png"
    preview_path.parent.mkdir(parents=True,exist_ok=True)
    # A floating product view keeps the separate internal radar visible under the header.
    world=bpy.data.worlds.new("Softbox")
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value=(.55,.62,.69,1)
    world.node_tree.nodes["Background"].inputs[1].default_value=.47

    def aim(obj, at):
        obj.rotation_euler=(Vector(at)-obj.location).to_track_quat("-Z","Y").to_euler()

    def area(name,pos,energy,size,color):
        bpy.ops.object.light_add(type="AREA",location=pos)
        ob=bpy.context.object
        ob.name=name
        ob.data.energy=energy
        ob.data.shape="DISK"
        ob.data.size=size
        ob.data.color=color
        aim(ob,(.6,0,1.65))
    area("Large key",(1.7,1.9,5.9),475,3.3,(1,.96,.90))
    area("Edge",(-2.1,-2.3,4),610,2.7,(.8,.91,1))
    area("Lens reflection",(4,.2,2.9),100,1.2,(.81,.94,1))
    bpy.ops.mesh.primitive_plane_add(size=200,location=(0,0,.77))
    bpy.context.object.data.materials.append(material("Studio floor",(.205,.242,.266),.05,.64))
    bpy.ops.object.camera_add(location=(4.0,4.0,3.15))
    scene.camera=bpy.context.object
    aim(scene.camera,(.73,0,1.66))
    scene.camera.data.type="ORTHO"
    scene.camera.data.ortho_scale=2.69
    scene.render.engine="CYCLES"
    scene.cycles.samples=12
    scene.cycles.use_denoising=True
    scene.render.threads_mode="FIXED"
    scene.render.threads=2
    scene.render.resolution_x=840
    scene.render.resolution_y=700
    scene.render.resolution_percentage=100
    scene.render.filepath=str(preview_path)
    bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()
