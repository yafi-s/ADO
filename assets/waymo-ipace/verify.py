"""Reopen native and exchange files in Blender and numerically exercise all four wheel pivots."""
from pathlib import Path
import json
import struct

import bpy
from mathutils import Vector

OUT=Path(__file__).resolve().parent
WHEELS={f"Wheel_{a}_{s}":(x,sign*track,.365)
        for x,track,a in ((1.495,.8205,"Front"),(-1.495,.8305,"Back"))
        for sign,s in ((1,"Left"),(-1,"Right"))}
ANCHORS={"roof_lidar":(-.176,0,1.958),"roof_camera_front":(.494,0,1.656),
         "perimeter_front_left":(1.67,1.006,.974),"perimeter_front_right":(1.67,-1.006,.974),
         "perimeter_rear_left":(-2.017,1.006,.826),"perimeter_rear_right":(-2.017,-1.006,.826)}


def bounds(objects):
    lo=[float("inf")]*3
    hi=[-float("inf")]*3
    for ob in objects:
        if ob.type not in {"MESH","CURVE"} or all(c.hide_render for c in ob.users_collection):
            continue
        ev=ob.evaluated_get(bpy.context.evaluated_depsgraph_get())
        data=ev.to_mesh()
        for v in data.vertices:
            p=ev.matrix_world @ v.co
            for i in range(3):
                lo[i]=min(lo[i],p[i])
                hi[i]=max(hi[i],p[i])
        ev.to_mesh_clear()
    return lo,hi


def check_rig():
    arm,=[ob for ob in bpy.context.scene.objects if ob.type=="ARMATURE"]
    assert set(arm.data.bones.keys())=={"Vehicle_Base",*WHEELS}
    for key,xyz in WHEELS.items():
        bone=arm.data.bones[key]
        assert bone.parent.name=="Vehicle_Base"
        assert (arm.matrix_world@bone.head_local-Vector(xyz)).length<.001,(key,arm.matrix_world@bone.head_local)
    return arm


def check_anchors():
    for name,xyz in ANCHORS.items():
        ob=bpy.data.objects["sensor_origin_"+name]
        assert (ob.matrix_world.translation-Vector(xyz)).length<.001,(name,ob.matrix_world.translation)


def point(ob):
    ev=ob.evaluated_get(bpy.context.evaluated_depsgraph_get())
    return ev.matrix_world @ ev.data.vertices[0].co


def exercise(arm):
    meshes=[o for o in bpy.context.scene.objects if o.type=="MESH" and not all(c.hide_render for c in o.users_collection)]
    assert len(meshes)==5
    for o in meshes:
        expected=o.name.removeprefix("SK_IPACE_").replace("Chassis","Vehicle_Base")
        groups={g.index:g.name for g in o.vertex_groups}
        for v in o.data.vertices:
            active=[g for g in v.groups if g.weight>.00001]
            assert len(active)==1 and groups[active[0].group]==expected and abs(active[0].weight-1)<.00001
    for key in WHEELS:
        before={o.name:point(o) for o in meshes}
        b=arm.pose.bones[key]
        b.rotation_mode="XYZ"
        # Rotate about the wheel axle in armature space, independent of FBX bone roll.
        axis=b.bone.matrix_local.to_quaternion().inverted()@Vector((0,1,0))
        from mathutils import Quaternion
        b.rotation_mode="QUATERNION"
        b.rotation_quaternion=Quaternion(axis,.42)
        bpy.context.view_layer.update()
        for o in meshes:
            d=(point(o)-before[o.name]).length
            if o.name.endswith(key):
                assert d>.01,(key,d)
            else:
                assert d<.00001,(key,o.name,d)
        b.rotation_quaternion.identity()
        bpy.context.view_layer.update()


def main():
    bpy.ops.wm.open_mainfile(filepath=str(OUT/"waymo-ipace.blend"))
    check_rig()
    check_anchors()
    studio=bpy.data.collections["90 | Photography studio"]
    parts=[ob for ob in bpy.context.scene.objects if ob.name not in studio.objects]
    targets=bounds(parts)
    assert abs(targets[0][2])<.0001 and abs(targets[1][2]-2.039)<.001
    assert abs(targets[1][1]-1.0695)<.001 and abs(targets[0][1]+1.0695)<.001
    body=bounds(bpy.data.collections["01 | Painted exterior"].objects)
    assert abs(body[1][0]-body[0][0]-4.682)<.006,body
    for kind,op in (("glb",bpy.ops.import_scene.gltf),("fbx",bpy.ops.import_scene.fbx)):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        op(filepath=str(OUT/("waymo-ipace."+kind)))
        arm=check_rig()
        if kind=="glb":
            check_anchors()
        observed=bounds(bpy.context.scene.objects)
        for i in (0,1):
            assert (Vector(observed[i])-Vector(targets[i])).length<.002,(kind,observed,targets)
        exercise(arm)
    buf=(OUT/"waymo-ipace.glb").read_bytes()
    g=json.loads(buf[20:20+struct.unpack_from("<I",buf,12)[0]])
    assert len(g["skins"])>=1 and len(g["meshes"])==5
    assert not g.get("images") and not g.get("cameras") and not any("uri" in b for b in g["buffers"])
    tris=sum(g["accessors"][p["indices"]]["count"]//3 for m in g["meshes"] for p in m["primitives"])
    print("VERIFIED",json.dumps({"vehicle_bounds_m":targets,"painted_body_bounds_m":body,"original_parts":len(parts),"runtime_meshes":5,"triangles":tris,"materials":len(g["materials"]),"independently_exercised_wheels_per_exchange_format":4}))


if __name__=="__main__":
    main()
