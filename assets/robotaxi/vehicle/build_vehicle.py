"""Create the editable Meridian research vehicle, in metres around the CARLA actor origin.

Run Blender in background with --python this_file. Use -- --render for the hero.
"""
import bisect
from math import atan2, cos, pi, sin, sqrt
from pathlib import Path
import sys

import bpy
from mathutils import Vector


OUT = Path(__file__).resolve().parent
WHEELS = (1.472, -1.389)
AXLE_Z = .269
TYRE_R = .331
BODY_END = 2.445
ROCKER = .104
scene = None
assembly = None


def material(name, rgb, metallic=0, roughness=.32, coat=0, emission=0):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*rgb, 1)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    for key, value in {"Base Color": (*rgb, 1), "Metallic": metallic,
                       "Roughness": roughness, "Coat Weight": coat,
                       "Emission Color": (*rgb, 1), "Emission Strength": emission}.items():
        bsdf.inputs[key].default_value = value
    return mat


def group(name):
    global assembly
    assembly = bpy.data.collections.new(name)
    scene.collection.children.link(assembly)


def mesh(name, vertices, faces, mat, smooth=True):
    data = bpy.data.meshes.new(name)
    data.from_pydata(vertices, [], faces)
    data.update()
    obj = bpy.data.objects.new(name, data)
    assembly.objects.link(obj)
    data.materials.append(mat)
    for poly in data.polygons:
        poly.use_smooth = smooth
    return obj


def patch(name, rows, mat, flip=False):
    n = len(rows[0])
    verts = [p for row in rows for p in row]
    faces = []
    for j in range(len(rows)-1):
        for i in range(n-1):
            a = j*n+i
            f = (a, a+1, a+n+1, a+n)
            faces.append(f[::-1] if flip else f)
    return mesh(name, verts, faces, mat)


def strip(name, points, radius, mat, closed=False):
    data = bpy.data.curves.new(name, 'CURVE')
    data.dimensions = '3D'
    data.resolution_u = 1
    data.bevel_depth = radius
    data.bevel_resolution = 3
    poly = data.splines.new('POLY')
    poly.points.add(len(points)-1)
    for node, pt in zip(poly.points, points):
        node.co = (*pt, 1)
    poly.use_cyclic_u = closed
    obj = bpy.data.objects.new(name, data)
    assembly.objects.link(obj)
    obj.data.materials.append(mat)
    return obj


def fit(x, knots):
    """Monotonic cubic Hermite interpolation with shared station tangents."""
    if x <= knots[0][0]:
        return knots[0][1]
    if x >= knots[-1][0]:
        return knots[-1][1]
    i = bisect.bisect_right([k[0] for k in knots], x)-1
    def slope(j):
        a,b = knots[j:j+2]
        return (b[1]-a[1])/(b[0]-a[0])
    def tangent(j):
        if j == 0 or j == len(knots)-1:
            return slope(max(0, j-1))
        lo,hi = slope(j-1),slope(j)
        return 2*lo*hi/(lo+hi) if lo*hi>0 else 0
    a,b = knots[i:i+2]
    h=b[0]-a[0]
    t=(x-a[0])/h
    return ((2*t**3-3*t*t+1)*a[1]+(t**3-2*t*t+t)*h*tangent(i)
            +(-2*t**3+3*t*t)*b[1]+(t**3-t*t)*h*tangent(i+1))


def steps(a,b,n):
    return [a+(b-a)*i/n for i in range(n+1)]


def signed_pow(x,e):
    return (1 if x>=0 else -1)*abs(x)**e


def shoulder(x):
    return fit(x,[(-2.45,.71),(-2.1,.83),(-1.55,.887),(-.85,.88),(.20,.879),
                  (.93,.884),(1.48,.854),(1.92,.806),(2.45,.692)])


def crown(x):
    return fit(x,[(-2.45,.851),(-2.04,.957),(-1.58,.986),(-.87,1.005),(.6,1.025),
                  (1.18,1.020),(1.57,1.007),(1.98,.981),(2.19,.944),(2.45,.802)])


def mould_width(x):
    return fit(x,[(-2.45,.849),(-2.10,.92),(-1.49,.958),(-.44,.939),(.65,.947),
                  (1.47,.961),(2.05,.928),(2.45,.846)])


def sweep(x, u, t):
    phase=max(0, (abs(x)-1.70)/(BODY_END-1.70))
    setback = .224*abs(u)**3.25 + .055*(t-.59)**2*4 + .023*(1-t)**3
    return x-(1 if x>0 else -1)*phase**2.2*setback


def depth(t):
    return fit(t,[(0,-.129),(.14,-.069),(.29,-.044),(.5,-.038),
                  (.71,-.011),(.86,.003),(1,0)])


def side_at(x,z,sgn,offset=0):
    t = min(1., max(0., (z-ROCKER)/(shoulder(x)-ROCKER)))
    return (sweep(x,1,t), sgn*(mould_width(x)+depth(t)+offset), z)


def top_at(x,u):
    h = (1-abs(u)**5.5)**.68
    hood=fit(x,[(.6,0),(1.12,.013),(1.71,.016),(2.13,.012),(2.40,0)])
    scallop=fit(abs(u),[(0,0),(.44,0),(.58,-.60),(.69,.56),(.77,1),(.91,.27),(1,0)])
    return (sweep(x,u,1),u*mould_width(x),shoulder(x)+(crown(x)-shoulder(x))*h+hood*scallop)


def arch(x):
    d=min(abs(x-w) for w in WHEELS)
    r=.363
    if d<r:
        return AXLE_Z+sqrt(max(0,r*r-d*d))
    return fit(d,[(r,AXLE_Z),(r+.012,.188),(r+.026,ROCKER)])


def end_at(end,u,t,out=0):
    x=end*BODY_END
    z=ROCKER+(top_at(x,u)[2]-ROCKER)*t
    return (sweep(x,u,t)+end*out, u*(mould_width(x)+depth(t)),z)


def make_body():
    group("01 • Coachwork")
    stations=set(steps(-BODY_END,BODY_END,112))
    for wheel in WHEELS:
        stations.update(wheel+.363*cos(a) for a in steps(0,pi,52))
        stations.update(wheel+s*d for s in (-1,1) for d in (.363,.368,.375,.38,.389))
    xs=sorted(stations)
    us=steps(-1,1,52)
    patch("Bonnet, rear deck and shoulder stamping",[[top_at(x,u) for u in us] for x in xs],pearl,True)
    for s,side in ((1,"Left"),(-1,"Right")):
        patch(side+" • integral quarter panels",[[side_at(x,arch(x)+(shoulder(x)-arch(x))*t,s)
            for t in steps(0,1,12)] for x in xs],pearl,s<0)
        patch(side+" • recessed wheelhouse return",[[side_at(x,arch(x),s),
               (sweep(x,1,0),s*.624,arch(x)-.018)] for x in xs],liner,s<0)
        for w,axle in zip(WHEELS,("front","rear")):
            points=[(w-.387,ROCKER+.017),(w-.373,.205)]
            points += [(w+.363*cos(t),AXLE_Z+.363*sin(t)) for t in steps(pi,0,64)]
            points += [(w+.373,.205),(w+.387,ROCKER+.017)]
            strip(side+" • "+axle+" arch rolled lip",[side_at(x,z,s,.0015) for x,z in points],.006,pearl)
        hood_points=[]
        for x in steps(.956,2.35,74):
            f=fit(x,[(.956,.81),(1.16,.758),(1.55,.717),(1.96,.718),(2.23,.763),(2.35,.865)])
            p=Vector(top_at(x,s*f))
            p.z+=.0018
            hood_points.append(p)
        strip(side+" • pressed hood shut line",hood_points,.0024,seam)
    patch("Undertray", [[(sweep(x,1,0),y,.094) for y in (-.63,.63)] for x in xs],liner)
    for end,label in ((1,"Front"),(-1,"Rear")):
        patch(label+" • bumper skin",[[end_at(end,u,t) for u in us] for t in steps(0,1,24)],pearl,end<0)


def canopy(q,t,sign=1,raise_by=0):
    # A continuous half-perimeter. q=0 is the windscreen midpoint, q=1 the rear glass.
    a=q*pi
    cx=-.315-.025*t
    long=1.555-.803*t
    wide=.859-.205*t
    x=cx+long*signed_pow(cos(a),.568)
    y=sign*(wide+raise_by)*signed_pow(sin(a),.58)
    z0=.945+.024*cos(a)-.022*sin(a)**2
    z1=1.452+.013*cos(a)-.019*sin(a)**2
    z=z0+(z1-z0)*t+.022*sin(pi*t)+raise_by*.34
    return (x+raise_by*cos(a),y,z)


def make_canopy():
    group("02 • Greenhouse")
    for s,label in ((1,"Left"),(-1,"Right")):
        # Dedicated windshield, side glass, backlight and pillar meshes share all edges.
        bands=[(0,.189,"windscreen",glass),(.189,.204,"A-pillar",pearl),
               (.204,.487,"front door glass",glass),(.487,.495,"B-pillar",pillar),
               (.495,.674,"rear door glass",glass),(.674,.684,"quarter-light mullion",pillar),
               (.684,.773,"quarter light",glass),(.773,.838,"sail panel",pearl),(.838,1,"backlight",glass)]
        for a,b,title,mat in bands:
            patch(label+" • "+title, [[canopy(q,t,s) for q in steps(a,b,max(4,round((b-a)*110)))]
                    for t in steps(0,1,10)],mat,s<0)
        strip(label+" • glazing bottom seal",[canopy(q,0,s,.003) for q in steps(0,1,124)],.013,pillar)
        strip(label+" • bright window surround",[canopy(q,.012,s,.006) for q in steps(.192,.832,80)],.0065,alloy)
        for q in (.184,.207,.486,.496,.774,.836):
            strip(label+" • glass frit "+str(q),[canopy(q,t,s,.002) for t in steps(.012,.989,18)],.0055,pillar)
        strip(label+" • roof edge",[canopy(q,1,s) for q in steps(0,1,120)],.014,pearl)
    ring=[canopy(q,1,1) for q in steps(0,1,96)] + [canopy(q,1,-1) for q in steps(1,0,96)[1:-1]]
    rows=[]
    for f in steps(1,.001,16):
        rows.append([(x*f-.34*(1-f),y*f,z+(1.487-z)*(1-f**3)) for x,y,z in ring+[ring[0]]])
    patch("Bonded panoramic roof • pearl cap",rows,pearl)
    for s,name in ((1,"Driver"),(-1,"Passenger")):
        blade=[canopy(q,.073+.040*q/.160,s,.005) for q in steps(.024,.160,30)]
        strip(name+" • wiper blade",blade,.0075,graphite)
        strip(name+" • wiper arm",[canopy(q,t,s,.006) for q,t in
              ((.150,.018),(.141,.041),(.121,.054),(.105,.076),(.092,.095))],.0055,brake)


def bez_path(nodes):
    """Rounded closed polygon, retaining long straight pressed panel edges."""
    points=[]
    for i,p in enumerate(nodes):
        v=Vector(p)
        a=v.lerp(Vector(nodes[i-1]),.10)
        b=v.lerp(Vector(nodes[(i+1)%len(nodes)]),.10)
        for t in steps(0,1,6):
            points.append(a*(1-t)**2+v*(2*t*(1-t))+b*t*t)
        dest=Vector(nodes[(i+1)%len(nodes)]).lerp(v,.10)
        points += [b.lerp(dest,t) for t in steps(.075,.925,12)]
    return points


def panel(name,outline,side,finish,offset=.004,rim=.003):
    contour=bez_path(outline)
    cx=sum(p.x for p in contour)/len(contour)
    cz=sum(p.y for p in contour)/len(contour)
    rows=[]
    for f in [.0001]+steps(.075,1,17):
        rows.append([side_at(cx+(p.x-cx)*f,cz+(p.y-cz)*f,side,offset+.002*(1-f))
                     for p in contour+[contour[0]]])
    patch(name,rows,finish,side>0)
    if rim:
        strip(name+" • gasket",[side_at(p.x,p.y,side,offset) for p in contour],rim,seam,True)


def make_doors():
    group("03 • Closures and lower trim")
    for s,label in ((1,"Left"),(-1,"Right")):
        front=[(.994,.867),(.998,.760),(1.010,.602),(1.008,.322),(.976,.229),
               (.906,.202),(-.088,.200),(-.120,.228),(-.122,.549),(-.118,.869)]
        rear=[(-.134,.869),(-.137,.547),(-.135,.233),(-.174,.202),(-.858,.202),
              (-.909,.238),(-.936,.349),(-.992,.462),(-1.094,.581),(-1.225,.694),(-1.377,.863)]
        for name,path in (("Front passenger door",front),("Rear passenger door",rear)):
            panel(label+" • "+name,path,s,pearl,rim=.0023)
        for i,x in enumerate((.002,-.998)):
            panel(label+f" • flush latch {i+1}",[(x-.078,.789),(x+.060,.789),
                  (x+.069,.799),(x+.061,.812),(x-.066,.813),(x-.077,.805)],s,alloy,.014,.0025)
        panel(label+" • rocker aerofoil",[(-.972,.159),(-.884,.133),(.972,.13),
              (1.04,.159),(.89,.190),(-.89,.186)],s,graphite,.017,.002)
        strip(label+" • rocker edge",[side_at(x,.169+.009*sin((x+1)*pi/2),s,.018)
              for x in steps(-.89,.96,65)],.005,alloy)
        # The thin, stamped character line runs through the doors under the belt.
    panel("Left • charge-port flap",[(-1.95,.735),(-1.76,.742),(-1.752,.830),(-1.943,.822)],1,pearl,.005,.002)


def wheel_point(cx,s,r,angle,lateral):
    return (cx+r*cos(angle),s*(.796+lateral),AXLE_Z+r*sin(angle))


def revolved(name,cx,s,profile,mat):
    rows=[[wheel_point(cx,s,r,a,lateral) for a in steps(0,2*pi,72)] for r,lateral in profile]
    return patch(name,rows,mat,s<0)


def spokes(name,cx,s):
    vertices=[]
    faces=[]
    for k in range(10):
        base=2*pi*k/10
        shape=[(.056,-.27),(.126,-.14),(.223,.035),(.224,.175),(.174,.145),(.072,.048)]
        start=len(vertices)
        for front in (False,True):
            for radius,angle in shape:
                depth=.113-(.225-radius)*.048+(.006 if front else -.003)
                vertices.append(wheel_point(cx,s,radius,base+angle,depth))
        n=len(shape)
        faces += [tuple(start+j for j in range(n)), tuple(start+n+j for j in reversed(range(n)))]
        for j in range(n):
            a,b=start+j,start+(j+1)%n
            faces.append((a,b,b+n,a+n))
    obj=mesh(name,vertices,faces,alloy,False)
    mod=obj.modifiers.new("Diamond-cut edge", 'BEVEL')
    mod.width=.0023
    mod.segments=3
    obj.modifiers.new("Facet normals","WEIGHTED_NORMAL")


def make_wheels():
    group("04 • Running gear")
    for x,axle in zip(WHEELS,("Front","Rear")):
        for s,side in ((1,"left"),(-1,"right")):
            w=axle+" "+side+" • "
            revolved(w+"all-season tire",x,s,[(.227,-.103),(.244,-.113),(.282,-.119),
                (.307,-.113),(.321,-.101),(.329,-.080),(.331,-.065),(.331,.065),
                (.329,.080),(.321,.101),(.308,.113),(.284,.119),(.259,.116),(.239,.109),(.227,.102)],rubber)
            for j,r in enumerate((.248,.284,.293)):
                revolved(w+f"sidewall relief {j+1}",x,s,[(r,.117),(r+.001,.1196),(r+.002,.117)],rubber)
            for j,d in enumerate((-.061,-.021,.021,.061)):
                revolved(w+f"tread channel {j+1}",x,s,[(.3312,d-.0015),(.3314,d+.0015)],liner)
            revolved(w+"painted alloy barrel",x,s,[(.226,-.095),(.232,-.092),(.232,.107),
                    (.227,.112),(.215,.104),(.208,.088),(.210,-.080)],graphite)
            revolved(w+"turned rim flange",x,s,[(.216,.109),(.219,.114),(.229,.1145),(.231,.110)],alloy)
            revolved(w+"brake rotor",x,s,[(.057,.080),(.177,.080),(.184,.076),(.184,.064),(.057,.062)],brake)
            for j,r in enumerate((.134,.153,.170)):
                revolved(w+f"rotor track {j+1}",x,s,[(r,.0801),(r+.0011,.0803)],graphite)
            spokes(w+"directional turbine blades",x,s)
            revolved(w+"hub cap",x,s,[(.067,.100),(.067,.113),(.058,.121),(.003,.125)],graphite)
            revolved(w+"center medallion",x,s,[(.026,.122),(.026,.126),(.001,.1265)],alloy)
            for j in range(5):
                ang=2*pi*j/5+pi/10
                bot=Vector(wheel_point(x,s,.045,ang,.121))
                verts=[bot+Vector((.0054*cos(a),s*.002,.0054*sin(a))) for a in steps(0,2*pi,6)[:-1]]
                mesh(w+f"hub fastener {j+1}",verts,[tuple(range(6))],brake,False)


def fascia_ribbon(name,end,points,mat,offset):
    """Curved front/rear ribbon with per-station upper/lower heights."""
    amin,amax=points[0][0],points[-1][0]
    rows=[]
    for u in steps(amin,amax,max(14,round((amax-amin)*96))):
        lo=fit(u,[(k[0],k[1]) for k in points])
        hi=fit(u,[(k[0],k[2]) for k in points])
        rows.append([end_at(end,u,t,offset) for t in steps(lo,hi,4)])
    return patch(name,rows,mat,end>0)


def make_lights():
    group("05 • Lighting and fascias")
    for s,side in ((1,"Left"),(-1,"Right")):
        def sym(records):
            return sorted((u*s,lo,hi) for u,lo,hi in records)
        housing=[(.19,.739,.748),(.30,.726,.795),(.55,.719,.822),(.76,.710,.842),
                 (.93,.68,.855),(1,.766,.836)]
        fascia_ribbon(side+" • projector enclosure",1,sym(housing),pillar,.005)
        fascia_ribbon(side+" • daytime light guide",1,sym([(.22,.744,.749),(.33,.767,.779),
                     (.57,.791,.805),(.78,.808,.822),(.947,.823,.838),(.989,.81,.82)]),daylight,.009)
        for i,u in enumerate((.48,.57,.66)):
            fascia_ribbon(side+f" • matrix projector {i+1}",1,sym([(u,.743,.760),
                       (u+.018,.749,.777),(u+.040,.75,.772)]),optic,.010)
        fascia_ribbon(side+" • amber repeater",1,sym([(.91,.744,.755),(.94,.748,.765),(.965,.758,.776)]),amber,.010)
        fascia_ribbon(side+" • air curtain",1,sym([(.743,.283,.29),(.821,.20,.362),(.904,.19,.504),
                      (.942,.25,.531),(.958,.32,.477)]),graphite,.005)
        for j in range(3):
            u=.836+j*.036
            fascia_ribbon(side+f" • vent vane {j+1}",1,sym([(u,.234,.248),(u+.018,.275,.30)]),brake,.008)
        # Side wraps and reflectors are placed on the same shaped stamping as the doors.
        end_low=end_at(1,s,.766)[2]
        end_high=end_at(1,s,.836)[2]
        panel(side+" • front lamp return",[(BODY_END,end_low),(BODY_END,end_high),(2.26,.654),
              (2.11,.692),(2.025,.700),(2.045,.680),(2.18,.644)],s,pillar,.005,0)
        strip(side+" • blade return",[side_at(x,z,s,.009) for x,z in
              [(BODY_END,end_high-.004),(2.39,.624),(2.27,.650),(2.14,.678),(2.045,.693)]],.005,daylight)
        panel(side+" • tail lamp return",[(-2.443,.672),(-2.432,.718),(-2.235,.771),(-2.074,.789),
              (-2.081,.769),(-2.272,.711)],s,red,.007,0)
        strip(side+" • rear light guide",[side_at(x,z,s,.011) for x,z in
              [(-2.44,.707),(-2.34,.728),(-2.23,.754),(-2.12,.771),(-2.08,.780)]],.006,redlight)
    fascia_ribbon("Front • lower cooling inlet",1,[(-.76,.211,.218),(-.62,.144,.325),
              (-.43,.134,.329),(0,.124,.315),(.43,.134,.329),(.62,.144,.325),(.76,.211,.218)],graphite,.006)
    for j in range(24):
        u=-.603+j*.052
        fascia_ribbon(f"Front • airfoil fin {j+1:02}",1,[(u,.181,.199),(u+.012,.175,.292)],liner,.009)
    fascia_ribbon("Front • chin blade",1,[(-.80,.101,.102),(-.64,.086,.116),(0,.071,.098),
                 (.64,.086,.116),(.80,.101,.102)],brake,.010)
    fascia_ribbon("Rear • smoked full-width lamp", -1,[(-1,.783,.848),(-.87,.721,.832),(-.59,.716,.803),
             (0,.739,.790),(.59,.716,.803),(.87,.721,.832),(1,.783,.848)],pillar,.006)
    fascia_ribbon("Rear • continuous red light guide",-1,[(-1,.798,.816),(-.89,.792,.813),(0,.764,.780),
             (.89,.792,.813),(1,.798,.816)],redlight,.009)
    fascia_ribbon("Rear • bumper diffuser",-1,[(-.97,.14,.17),(-.85,.08,.29),(-.60,.066,.27),
             (0,.064,.215),(.60,.066,.27),(.85,.08,.29),(.97,.14,.17)],graphite,.006)


def make_mirrors():
    group("06 • Exterior fittings")
    for s,side in ((1,"Left"),(-1,"Right")):
        strip(side+" • wing mirror stem",[(.753,s*.876,.976),(.716,s*.932,1.003),(.707,s*.989,1.03)],.022,graphite)
        rows=[]
        for x,cy,wy,cz,h in [(.800,1.006,.008,1.060,.009),(.785,1.012,.063,1.06,.035),
                (.744,1.020,.084,1.060,.048),(.670,1.021,.090,1.060,.049),
                (.597,1.018,.081,1.062,.043),(.585,1.016,.074,1.060,.037)]:
            rows.append([(x,s*(cy+wy*signed_pow(cos(t),.71)),cz+h*signed_pow(sin(t),.76))
                         for t in steps(0,2*pi,40)])
        patch(side+" • wing mirror shell",rows,pearl,s<0)
        shape=[(.584,s*(1.016+.074*signed_pow(cos(t),.71)),1.060+.037*signed_pow(sin(t),.76))
               for t in steps(0,2*pi,40)[:-1]]
        mesh(side+" • mirror glass",shape,[tuple(range(len(shape)))],optic)
        strip(side+" • mirror gasket",shape,.004,pillar,True)


def aim(obj, at):
    obj.rotation_euler=(Vector(at)-obj.location).to_track_quat("-Z","Y").to_euler()


def bone_for(obj):
    if any(c.name.startswith("04 •") for c in obj.users_collection):
        axle,side=obj.name.split(" • ")[0].split()
        return "Wheel_"+("Front" if axle=="Front" else "Back")+"_"+side.title()
    return "Vehicle_Base"


def rig_vehicle(objects):
    group("08 • Vehicle rig")
    skeleton=bpy.data.armatures.new("Meridian • suspension skeleton")
    arm=bpy.data.objects.new("Armature",skeleton)
    assembly.objects.link(arm)
    bpy.context.view_layer.objects.active=arm
    arm.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    root=skeleton.edit_bones.new("Vehicle_Base")
    root.head=(0,0,0)
    root.tail=(.35,0,0)
    for x,axle in zip(WHEELS,("Front","Back")):
        for s,side in ((1,"Left"),(-1,"Right")):
            bone=skeleton.edit_bones.new("Wheel_"+axle+"_"+side)
            bone.head=(x,s*.796,AXLE_Z)
            bone.tail=(x+.15,s*.796,AXLE_Z)
            bone.parent=root
    bpy.ops.object.mode_set(mode="OBJECT")
    arm.show_in_front=False
    skeleton.display_type="STICK"
    bpy.context.view_layer.update()
    # Root-level objects only: if an import brings a transform hierarchy, preserve it.
    for ob in objects:
        if ob.parent is None:
            transform=ob.matrix_world.copy()
            ob.parent=arm
            ob.parent_type="BONE"
            ob.parent_bone=bone_for(ob)
            ob.matrix_world=transform
    bpy.context.view_layer.update()
    return arm


def export_skeleton(arm, objects):
    """Bake evaluated surfaces into five material-preserving rigid skeletal meshes."""
    root="Vehicle_Base"
    parts={bone.name: [[],[],[],[],[]] for bone in arm.data.bones}
    graph=bpy.context.evaluated_depsgraph_get()
    for obj in objects:
        if obj.type not in {"CURVE","MESH"}:
            continue
        evaluated=obj.evaluated_get(graph)
        data=evaluated.to_mesh()
        verts,faces,mats,indices,smooth=parts[bone_for(obj)]
        first=len(verts)
        verts.extend(tuple(evaluated.matrix_world @ v.co) for v in data.vertices)
        slots=[]
        for mat in data.materials:
            if mat not in mats:
                mats.append(mat)
            slots.append(mats.index(mat))
        data.calc_loop_triangles()
        for tri in data.loop_triangles:
            faces.append(tuple(first+v for v in tri.vertices))
            indices.append(slots[tri.material_index])
            smooth.append(data.polygons[tri.polygon_index].use_smooth)
        evaluated.to_mesh_clear()
    bpy.ops.object.select_all(action="DESELECT")
    arm.select_set(True)
    temporary=[]
    for name,(verts,faces,mats,indices,smooth) in parts.items():
        ob=mesh("SK_Meridian_"+("Chassis" if name==root else name),verts,faces,mats[0])
        temporary.append(ob)
        for mat in mats[1:]:
            ob.data.materials.append(mat)
        ob.data.polygons.foreach_set("material_index",indices)
        ob.data.polygons.foreach_set("use_smooth",smooth)
        weights=ob.vertex_groups.new(name=name)
        weights.add(list(range(len(verts))),1,"REPLACE")
        deform=ob.modifiers.new("Rigid suspension weights","ARMATURE")
        deform.object=arm
        ob.parent=arm
        ob.select_set(True)
    bpy.context.view_layer.objects.active=arm
    bpy.ops.export_scene.fbx(filepath=str(OUT/"meridian-skeletal.fbx"),
                  use_selection=True,object_types={"ARMATURE","MESH"},add_leaf_bones=False,
                  use_armature_deform_only=True,bake_anim=False,axis_forward="-Y",axis_up="Z",
                  primary_bone_axis="X",secondary_bone_axis="Z",apply_unit_scale=True)
    for ob in temporary:
        data=ob.data
        bpy.data.objects.remove(ob,do_unlink=True)
        bpy.data.meshes.remove(data)


def create_studio():
    group("90 • Studio — excluded from GLB")
    ground=mesh("Cyclorama",[(-200,-200,-.0625),(200,-200,-.0625),(200,200,-.0625),(-200,200,-.0625)],
                [(0,1,2,3)],material("Studio • slate",(.19,.226,.245),.07,.49))
    def area(name,pos,focus,power,width,height,color):
        data=bpy.data.lights.new(name,"AREA")
        data.energy=power
        data.shape="RECTANGLE"
        data.size=width
        data.size_y=height
        data.color=color
        obj=bpy.data.objects.new(name,data)
        assembly.objects.link(obj)
        obj.location=pos
        aim(obj,focus)
    area("Silk • overhead",(-.8,1.6,6.7),(0,0,0),540,6,3,(1,.96,.92))
    area("Strip • side key",(-.7,4.6,2.9),(.2,0,.5),340,5.2,1.45,(.82,.91,1))
    area("Strip • rear rim",(-3.7,-2.9,4.1),(-.2,0,.7),820,3.9,2,(.78,.86,1))
    area("Silk • nose",(4.8,-2.4,3.3),(1,0,.7),570,3.2,3,(1,.91,.77))
    camera=bpy.data.cameras.new("Product camera")
    obj=bpy.data.objects.new("Product camera",camera)
    assembly.objects.link(obj)
    obj.location=(5.95,8.3,3.44)
    aim(obj,(0,0,.984))
    camera.lens=55
    scene.camera=obj
    world=bpy.data.worlds.new("Studio • soft ambient")
    world.use_nodes=True
    world.node_tree.nodes["Background"].inputs[0].default_value=(.51,.58,.66,1)
    world.node_tree.nodes["Background"].inputs[1].default_value=.36
    scene.world=world
    scene.render.engine="CYCLES"
    scene.cycles.samples=12
    scene.cycles.use_denoising=True
    scene.render.threads_mode="FIXED"
    scene.render.threads=2
    scene.render.resolution_x=960
    scene.render.resolution_y=600
    scene.render.resolution_percentage=100
    scene.render.image_settings.file_format="PNG"
    scene.render.filepath="//meridian-studio.png"
    scene.view_settings.view_transform="AgX"
    # Hide production lights and the studio sweep from modeling views.
    for obj in assembly.objects:
        if obj.type != "CAMERA":
            obj.hide_set(True)
    for screen in bpy.data.screens:
        for a in screen.areas:
            if a.type=="VIEW_3D":
                v=a.spaces.active
                v.region_3d.view_distance=6.25
                v.region_3d.view_location=(0,0,.87)
                v.region_3d.view_rotation=scene.camera.rotation_euler.to_quaternion()
                v.overlay.show_floor=False
                v.overlay.show_extras=False
                v.shading.color_type="MATERIAL"


def main():
    global scene,pearl,glass,pillar,seam,alloy,graphite,rubber,brake,liner,optic,daylight,amber,red,redlight
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene=bpy.context.scene
    scene.unit_settings.system="METRIC"
    pearl=material("01 • Glacier ceramic metallic",(.77,.834,.845),.30,.24,.34)
    glass=material("02 • Solar-control glazing",(.018,.039,.050),.47,.15,.25)
    pillar=material("03 • Piano black pillar",(.011,.017,.020),.40,.20,.28)
    seam=material("14 • Closure joint",(.090,.109,.116),.24,.40)
    alloy=material("04 • Bright machined aluminum",(.59,.66,.69),.91,.23)
    graphite=material("05 • Satin graphite",(.043,.055,.059),.62,.32)
    rubber=material("06 • Vulcanized sidewall",(.021,.024,.025),.02,.56)
    brake=material("07 • Titanium grey",(.22,.256,.267),.81,.33)
    liner=material("08 • Wheelhouse felt",(.012,.016,.017),0,.69)
    optic=material("09 • Optical silver",(.47,.66,.72),.82,.13,.43)
    daylight=material("10 • Ice-white LED guide",(.72,.92,1),.19,.18,.16,2.3)
    amber=material("11 • Amber repeater",(.92,.225,.018),.20,.22,.20,.63)
    red=material("12 • Garnet polymer",(.24,.011,.023),.39,.22,.36)
    redlight=material("13 • Red light guide",(.86,.025,.033),.23,.19,.13,1.2)
    make_body()
    make_canopy()
    make_doors()
    make_wheels()
    make_lights()
    make_mirrors()
    group("07 • Perception hardware")
    bpy.ops.import_scene.gltf(filepath=str(OUT.parent/"roof-carrier.glb"))
    imported=list(bpy.context.selected_objects)
    for ob in imported:
        for col in list(ob.users_collection):
            col.objects.unlink(ob)
        assembly.objects.link(ob)
    bpy.ops.object.select_all(action="DESELECT")
    for ob in scene.objects:
        ob.select_set(True)
    scene["asset_name"]="Meridian • autonomous research sedan"
    scene["coordinate_system"]="CARLA actor frame, converted to Blender (x, -y, z); metres"
    scene["wheel_radius_m"]=TYRE_R
    # Curves are exported as evaluated meshes. All studio objects are added afterwards.
    bpy.ops.export_scene.gltf(filepath=str(OUT/"meridian.glb"),export_format="GLB",
                              export_apply=True,export_extras=True,use_selection=True,export_animations=False)
    bpy.ops.object.select_all(action="DESELECT")
    components=list(scene.objects)
    arm=rig_vehicle(components)
    export_skeleton(arm,components)
    bpy.ops.object.select_all(action="DESELECT")
    create_studio()
    bpy.context.preferences.filepaths.save_version=0
    bpy.ops.wm.save_as_mainfile(filepath=str(OUT/"meridian.blend"),compress=True)
    if "--render" in sys.argv:
        prefs=bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type="OPTIX"
        prefs.refresh_devices()
        for device in prefs.devices:
            device.use=device.type=="OPTIX"
        if not any(d.use for d in prefs.devices):
            raise RuntimeError("No OptiX GPU available")
        scene.cycles.device="GPU"
        bpy.ops.render.render(write_still=True)


if __name__=="__main__":
    main()
