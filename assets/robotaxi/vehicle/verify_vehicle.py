"""Reopen all interchange files, check their frames and exercise wheel skinning."""
import json
from math import cos, radians, sin
from pathlib import Path
import struct
import sys

import bpy
from mathutils import Vector


OUT = Path(__file__).resolve().parent
sys.path.insert(0,str(OUT.parents[2]))
from crosswalk.robotaxi import MOUNTS


def bounds(objects):
    graph=bpy.context.evaluated_depsgraph_get()
    lo=[float("inf")]*3
    hi=[float("-inf")]*3
    for obj in objects:
        if obj.type not in {"CURVE","MESH"}:
            continue
        evaluated=obj.evaluated_get(graph)
        data=evaluated.to_mesh()
        for vertex in data.vertices:
            xyz=evaluated.matrix_world @ vertex.co
            for axis in range(3):
                lo[axis]=min(lo[axis],xyz[axis])
                hi[axis]=max(hi[axis],xyz[axis])
        evaluated.to_mesh_clear()
    return lo,hi


def check_bounds(observed,expected):
    for actual,want in zip(observed,expected):
        assert (Vector(actual)-Vector(want)).length<.001,(observed,expected)


def check_origins():
    for name,(x,y,z,yaw) in MOUNTS.items():
        obj=bpy.data.objects["sensor_origin_"+name]
        assert (obj.matrix_world.translation-Vector((x,-y,z))).length<1e-5,name
        forward=obj.matrix_world.to_quaternion() @ Vector((1,0,0))
        heading=radians(-yaw)
        assert (forward-Vector((cos(heading),sin(heading),0))).length<1e-5,name


def check_bones():
    arm,=[o for o in bpy.context.scene.objects if o.type=="ARMATURE"]
    assert set(arm.data.bones.keys())=={"Vehicle_Base","Wheel_Front_Left","Wheel_Front_Right",
                                        "Wheel_Back_Left","Wheel_Back_Right"}
    for axle,x in (("Front",1.472),("Back",-1.389)):
        for side,y in (("Left",.796),("Right",-.796)):
            bone=arm.data.bones["Wheel_"+axle+"_"+side]
            assert bone.parent.name=="Vehicle_Base"
            point=arm.matrix_world @ bone.head_local
            assert (point-Vector((x,y,.269))).length<.001,(bone.name,point)
    return arm


def main():
    raw=(OUT/"meridian.glb").read_bytes()
    size=struct.unpack_from("<I",raw,12)[0]
    gltf=json.loads(raw[20:20+size])
    assert not gltf.get("cameras") and not gltf.get("images")
    assert not any("uri" in b for b in gltf["buffers"])
    triangles=sum(gltf["accessors"][p["indices"]]["count"]//3
                  for mesh in gltf["meshes"] for p in mesh["primitives"])
    bpy.ops.wm.open_mainfile(filepath=str(OUT/"meridian.blend"))
    check_bones()
    check_origins()
    studio=bpy.data.collections["90 • Studio — excluded from GLB"]
    native_bounds=bounds(o for o in bpy.context.scene.objects if o.name not in studio.objects)
    native_parts=sum(o.type in {"MESH","CURVE"} for o in bpy.context.scene.objects if o.name not in studio.objects)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(OUT/"meridian.glb"))
    static_bounds=bounds(bpy.context.scene.objects)
    check_bounds(static_bounds,native_bounds)
    check_origins()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=str(OUT/"meridian-skeletal.fbx"))
    arm=check_bones()
    skeletal_bounds=bounds(bpy.context.scene.objects)
    check_bounds(skeletal_bounds,native_bounds)
    meshes=[o for o in bpy.context.scene.objects if o.type=="MESH"]
    assert len(meshes)==5
    for ob in meshes:
        expected=ob.name.removeprefix("SK_Meridian_").replace("Chassis","Vehicle_Base")
        assert list(ob.vertex_groups.keys())==[expected]
        for v in ob.data.vertices:
            assert len(v.groups)==1 and abs(v.groups[0].weight-1)<1e-6
    wheel=bpy.data.objects["SK_Meridian_Wheel_Front_Left"]
    chassis=bpy.data.objects["SK_Meridian_Chassis"]
    def position(ob):
        ev=ob.evaluated_get(bpy.context.evaluated_depsgraph_get())
        return ev.matrix_world @ ev.data.vertices[0].co
    before_w=position(wheel)
    before_c=position(chassis)
    b=arm.pose.bones["Wheel_Front_Left"]
    b.rotation_mode="XYZ"
    b.rotation_euler[0]=.38
    bpy.context.view_layer.update()
    assert (position(wheel)-before_w).length>.005
    assert (position(chassis)-before_c).length<1e-6
    print("VERIFIED "+json.dumps({"editable_parts":native_parts,"gltf_meshes":len(gltf["meshes"]),
          "gltf_triangles":triangles,"gltf_materials":len(gltf["materials"]),
          "bounds_m":native_bounds,"skeletal_meshes":len(meshes),"bones":5,"sensor_frames":5}))


if __name__=="__main__":
    main()
