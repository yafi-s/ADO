from pathlib import Path
import math
import sys
import tempfile
OUT=Path(__file__).resolve().parent
DEST=Path(tempfile.gettempdir())/"ipace-inspection"
DEST.mkdir(parents=True,exist_ok=True)

def compose_sheet():
    """Combine the six locally rendered PNGs. Requires Pillow in system Python."""
    import os
    from PIL import Image, ImageDraw, ImageFont

    width,height=1840,1614
    sheet=Image.new("RGB",(width,height),(246,247,245))
    draw=ImageDraw.Draw(sheet)
    font_dir=Path(os.environ.get("WINDIR","C:/Windows"))/"Fonts"
    def font(size,bold=False):
        names=(str(font_dir/("segoeuib.ttf" if bold else "segoeui.ttf")),"DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
        for name in names:
            try:
                return ImageFont.truetype(name,size)
            except OSError:
                pass
        return ImageFont.load_default(size=size)
    ink=(28,37,41)
    mute=(76,89,94)
    draw.text((64,34),"Waymo Jaguar I-PACE",font=font(51,True),fill=ink)
    draw.text((65,105),"Fifth-generation Driver   /   Exterior model inspection",font=font(25),fill=mute)
    draw.text((width-62,55),"2.990 m wheelbase     /     4.682 m body",font=font(24,True),fill=ink,anchor="ra")
    draw.text((width-62,96),"Factory dimensions  ·  Sensor shapes fitted visually",font=font(20),fill=mute,anchor="ra")
    entries=[
     ("profile","Left profile","Wheelbase, quarter glazing and front fender package"),
     ("roof","Overhead","Bonnet outlet, aero carrier and hatch spoiler"),
     ("front","Front elevation","Grille sensor, corner optics and camera crown"),
     ("rear","Rear elevation","Hatch shut lines, rear-facing pods and valance"),
     ("rear_quarter","Rear three-quarter","Tail lamp wrap and rear bumper sensor fairings"),
     ("carrier","Roof carrier detail","Vision windows, status collar and roof lidar"),
    ]
    for i,(key,title,detail) in enumerate(entries):
        x=64+(i%2)*867
        y=165+(i//2)*457
        w,h=844,431
        draw.rounded_rectangle((x,y,x+w,y+h),radius=14,fill=(230,235,235),outline=(215,223,223),width=1)
        im=Image.open(DEST/(key+".png")).convert("RGBA")
        box=im.getchannel("A").getbbox()
        if key!="carrier":
            im=im.crop(box)
        target_w,target_h=w-60,307
        if key=="carrier":
            target_w,target_h=w-24,310
        im.thumbnail((target_w,target_h),Image.Resampling.LANCZOS)
        px=x+(w-im.width)//2
        py=y+16+(target_h-im.height)//2
        sheet.paste(im,(px,py),im)
        draw.line((x+24,y+335,x+w-24,y+335),fill=(198,208,208),width=1)
        draw.text((x+27,y+346),title,font=font(26,True),fill=ink)
        draw.text((x+27,y+386),detail,font=font(19),fill=mute)
    draw.text((64,1556),"Original renders of the editable model. Views are framed individually. Manufacturer photographs and drawings are linked in the asset notes.",font=font(20),fill=mute)
    sheet.save(OUT/"inspection.png",optimize=True)
    ends=Image.new("RGB",(1680,856),(230,235,235))
    labels=ImageDraw.Draw(ends)
    for i,key in enumerate(("front","rear")):
        im=Image.open(DEST/(key+".png")).convert("RGBA")
        im=im.crop(im.getchannel("A").getbbox())
        im.thumbnail((790,747),Image.Resampling.LANCZOS)
        ends.paste(im,(i*840+(840-im.width)//2,35+747-im.height),im)
        labels.text((i*840+420,803),key.capitalize(),anchor="mt",font=font(23,True),fill=ink)
    ends.save(OUT/"front-rear.png",optimize=True)

if "--compose" in sys.argv:
    compose_sheet()
    sys.exit(0)

import bpy
from mathutils import Vector, Quaternion

bpy.ops.wm.open_mainfile(filepath=str(OUT/"waymo-ipace.blend"))
scene=bpy.context.scene
camera=scene.camera
camera.data.type="ORTHO"
scene.render.threads_mode="FIXED";scene.render.threads=2
scene.cycles.device="CPU"
preview="--preview" in sys.argv
scene.cycles.samples=3 if preview else 12
scene.render.resolution_x=700 if preview else 950
scene.render.resolution_y=390 if preview else 555
scene.render.resolution_percentage=100
scene.render.film_transparent=True
for ob in bpy.data.collections["90 | Photography studio"].objects:
    if ob.type=="MESH":
        ob.hide_render=True

views=[
    ("profile",(0,10,0),(0,0,1.02),5.12),
    ("front",(10,0,0),(0,0,1.02),2.47),
    ("rear",(-10,0,0),(0,0,1.02),2.47),
    ("roof",(0,0,10),(0,0,0),5.12),
    ("rear_quarter",(-6.9,8,2.4),(0,0,1.02),5.22),
    ("carrier",(2.3,2.2,1.12),(-.15,0,1.72),1.78),
]
rear_rim=bpy.data.lights["Strip • rear rim"]
rim_energy=rear_rim.energy
for name,vec,ctr,scale in views:
    # A head-on windshield reflects the rear softbox straight into the camera.
    rear_rim.energy=rim_energy*.19 if name=="front" else rim_energy
    if name in ("front","rear"):
        scene.render.resolution_x,scene.render.resolution_y=(700,632) if preview else (950,860)
    else:
        scene.render.resolution_x,scene.render.resolution_y=(700,390) if preview else (950,555)
    center=Vector(ctr)
    camera.location=center+Vector(vec)
    q=(-Vector(vec)).to_track_quat("-Z","Y")
    if name=="roof":
        q=Quaternion((0,0,1),math.pi)
    camera.rotation_euler=q.to_euler()
    camera.data.ortho_scale=scale
    scene.render.filepath=str(DEST/(name+".png"))
    bpy.ops.render.render(write_still=True)
