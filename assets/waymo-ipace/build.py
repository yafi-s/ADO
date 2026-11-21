"""Build the independently modeled fifth-generation Waymo Jaguar I-PACE exterior."""
from pathlib import Path
from math import atan2, cos, pi, sin, sqrt
import sys
import bpy
from mathutils import Vector, Matrix

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parent / "robotaxi" / "vehicle"))
import build_vehicle as util

mesh, strip, patch, fit, steps = util.mesh, util.strip, util.patch, util.fit, util.steps
material, group, sp = util.material, util.group, util.signed_pow
TAU = 2*pi
FRONT, REAR = 2.341, -2.341
AXLES = [(1.495,.8205,"Front"),(-1.495,.8305,"Back")]
RADIUS = .365
scene = None
parts = {}
anchors = {}
bone = "Vehicle_Base"


def tag(obj):
    obj["rig_part"] = bone
    return obj


def box(name, center, size, radius, mat):
    x,y,z = (n/2 for n in size)
    verts = [(i*x,j*y,k*z) for i,j,k in ((-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),(-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1))]
    faces = [(3,2,1,0),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)]
    ob = mesh(name,verts,faces,mat,False)
    ob.location = center
    bevel=ob.modifiers.new("Moulded radii","BEVEL")
    bevel.width=radius
    bevel.segments=5
    ob.modifiers.new("Planar normals","WEIGHTED_NORMAL")
    return ob


def shell(name, rings, mat, exponent=1):
    rows=[]
    for z,cx,cy,l,w in rings:
        rows.append([(cx+l*sp(cos(a),exponent),cy+w*sp(sin(a),exponent),z) for a in steps(0,TAU,80)])
    ob=patch(name,rows,mat)
    for i in (0,-1):
        rim=rows[i][:-1]
        mesh(name+(" cap" if i else " underside"),rim,[tuple(range(len(rim))) if i else tuple(reversed(range(len(rim))))],mat,False)
    return ob


def sensor_face(name,cx,y,z,width,height,depth,s,mat):
    # Rounded rectangle in the viewing plane, independently of its shallow depth.
    ring=[(width*.5*sp(cos(a),.45),height*.5*sp(sin(a),.46)) for a in steps(0,TAU,48)]
    rows=[[(cx+dx*t,y+s*h,z+dz*t) for dx,dz in ring] for t,h in ((.89,-depth*.5),(1,-depth*.15),(1,depth*.15),(.91,depth*.5))]
    patch(name,rows,mat,s>0)
    cap(name+" face",rows[-1][:-1],mat)
    return rows

def ellipsoid(name,center,size,mat):
    x,y,z=center
    a,b,c=size
    return shell(name,[(z+c*sin(t),x,y,max(.0001,a*cos(t)),max(.0001,b*cos(t))) for t in steps(-pi/2,pi/2,12)],mat)


def cap(name,pts,mat):
    return mesh(name,pts,[tuple(range(len(pts)))],mat,False)


def ink(name, pos, right, up, label, height, mat, spacing=1.15):
    data=bpy.data.curves.new(name,'FONT')
    data.body=label
    data.size=height
    data.align_x="CENTER"
    data.align_y="CENTER"
    data.space_character=spacing
    if label=="W":
        data.offset=.0018
    ob=bpy.data.objects.new(name,data)
    util.assembly.objects.link(ob)
    ob.data.materials.append(mat)
    r,u=Vector(right).normalized(),Vector(up).normalized()
    ob.matrix_world=Matrix((r,u,r.cross(u))).transposed().to_4x4()
    ob.location=pos
    bpy.context.view_layer.objects.active=ob
    ob.select_set(True)
    bpy.ops.object.convert(target="MESH")
    ob.select_set(False)
    return ob


# Station lines are in metres, rear bumper to front bumper. Side belt rises toward the hatch.
WIDTH=[(REAR,.817),(-2.09,.913),(-1.67,.949),(-1.36,.945),(-.50,.929),(.40,.928),(1.40,.947),(1.68,.947),(2.05,.914),(FRONT,.856)]
BELT=[(REAR,.971),(-2.05,1.15),(-1.49,1.209),(-1.34,1.19),(-.81,1.145),(-.26,1.104),(.57,1.063),(1.09,1.037),(1.47,1.045),(1.84,.994),(2.10,.911),(FRONT,.783)]
CENTRE=[(REAR,1.018),(-2.08,1.20),(-1.25,1.15),(0,1.08),(.88,1.047),(1.28,1.063),(1.65,1.005),(2.08,.904),(FRONT,.805)]
SILL=.257
ARCH_R=.395


def nose_x(x,u,t):
    k=max(0,(abs(x)-1.77)/(FRONT-1.77))**2
    return x-(1 if x>0 else -1)*k*(.151*(.60*abs(u)**4+.40*abs(u)**12)+.045*(t-.52)**2*4)


def side(x,z,s,offset=0):
    belt=fit(x,BELT)
    t=max(0,min(1,(z-SILL)/(belt-SILL)))
    # Pressed rocker, lower door scallop and shoulder crease. The sill rises behind the rear wheel.
    sculpt=fit(t,[(0,-.082),(.10,-.050),(.19,-.016),(.27,-.043),(.46,-.033),(.69,-.013),(.88,.002),(1,-.017)])
    return (nose_x(x,1,t), s*(fit(x,WIDTH)+sculpt+offset),z)


def bonnet_slot(u):
    f=min(1,abs(u)/.685)
    return 1.989-.075*f*f,.027*max(0,1-f**4)**.6

def top(x,u):
    a=abs(u)
    ridge=fit(a,[(0,0),(.46,-.004),(.61,-.012),(.73,.015),(.89,.017),(1,0)])
    hood=max(0,min(1,(x-.83)/.45))
    # I-PACE's transverse bonnet outlet is a depressed duct behind the leading edge.
    slot,half=bonnet_slot(u)
    # The opening faces aft. The forward lip masks its dark interior at road level.
    rear_dip=max(0,1-abs(x-(slot-half))/.19)**2
    recess=rear_dip*max(0,1-(a/.70)**8)*(1 if x<=slot else .13)
    ridge+=.010*max(0,1-abs(x-1.72)/.58)*max(0,1-abs(a-.81)/.16)
    # Clearance below the raked backlight avoids the white inner deck intersecting glass.
    backlight=max(0,min(1,(x+2.29-.13*a*a)/.075))*max(0,min(1,(-.92-x)/.25))
    inner=max(0,min(1,(.99-a)/.17))
    z=fit(x,CENTRE)+(fit(x,BELT)-fit(x,CENTRE))*a**4.2+hood*ridge-.028*recess-.130*backlight*inner
    return (nose_x(x,u,1),u*(fit(x,WIDTH)-.017),z)


def fascia(end,u,z,offset=0):
    x=FRONT if end>0 else REAR
    maxz=top(x,u)[2]
    t=max(0,min(1,(z-SILL)/(maxz-SILL)))
    width=abs(side(x,SILL+(fit(x,BELT)-SILL)*t,1)[1])
    return (nose_x(x,u,t)+end*offset,u*width,z)


def arch(x):
    d=min(abs(x-cx) for cx,_,_ in AXLES)
    if d<ARCH_R:
        return RADIUS+sqrt(max(0,ARCH_R**2-d*d))
    return fit(d,[(ARCH_R,RADIUS),(ARCH_R+.012,.304),(ARCH_R+.034,SILL)])


def body():
    group("01 | Painted exterior")
    xs=set(steps(REAR,FRONT,95))
    for cx,_,_ in AXLES:
        xs.update(cx+ARCH_R*cos(a) for a in steps(0,pi,48))
        for s in (-1,1):
            xs.update(cx+s*(ARCH_R+delta) for delta in (0,.005,.012,.02,.034))
    xs=sorted(xs)
    us=sorted(set(steps(-1,1,40)+[-.685,.685]))
    # Two structured surface grids meet the physical outlet edges exactly.
    # This preserves smooth taper and prevents floating black decals across the bonnet.
    aft_x=1.73
    patch("Cowl and rear deck stamping",[[top(x,u) for u in us] for x in sorted({*[v for v in xs if v<aft_x],aft_x})],pearl,True)
    for leading in (False,True):
        rows=[]
        for t in steps(0,1,24):
            row=[]
            for u in us:
                x,w=bonnet_slot(u)
                begin,end=(x+w,FRONT) if leading else (aft_x,x-w)
                row.append(top(begin+(end-begin)*t,u))
            rows.append(row)
        patch("Bonnet | forward brow" if leading else "Bonnet | contoured air exit",rows,pearl,True)
    rows=[[],[],[],[]]
    # Include both terminal points and use the body's exact lateral tessellation.
    outline=sorted({-.685,.685,*[v for v in us if abs(v)<.685]})
    for u in outline:
        x,w=bonnet_slot(u)
        aft=Vector(top(x-w,u));front=Vector(top(x+w,u))
        depth=.022*min(1,w/.018)
        under_aft=aft.copy();under_aft.z-=depth
        under_front=front.copy();under_front.z-=depth
        for arr,pt in zip(rows,(aft,under_aft,under_front,front)):
            arr.append(tuple(pt))
    patch("Bonnet | inset duct throat",rows,black)
    strip("Bonnet | formed leading lip",rows[-1],.0019,pearl)
    for s,l in ((1,"L"),(-1,"R")):
        patch(l+" | bodyside pressing",[[side(x,arch(x)+(fit(x,BELT)-arch(x))*t,s) for t in steps(0,1,16)] for x in xs],pearl,s<0)
        patch(l+" | wheelhouse inner lining",[[side(x,arch(x),s),(nose_x(x,1,0),s*.655,arch(x)-.018)] for x in xs],black,s<0)
        for cx,_,axle in AXLES:
            # Close the inboard face of each stamped wheelhouse. This backing
            # carries the chassis transform and is visible through wheel openings.
            radius=ARCH_R+.015
            lining=[(cx+radius,s*.648,.151),(cx+radius,s*.648,RADIUS)]
            lining.extend((cx+radius*cos(a),s*.648,RADIUS+radius*sin(a)) for a in steps(0,pi,58)[1:])
            lining.append((cx-radius,s*.648,.151))
            cap(l+" | "+axle+" inner wheelhouse shield",lining if s<0 else list(reversed(lining)),black)
            path=[(cx-ARCH_R-.026,SILL+.005),(cx-ARCH_R-.010,.309)]
            path += [(cx+ARCH_R*cos(a),RADIUS+ARCH_R*sin(a)) for a in steps(pi,0,48)]
            path += [(cx+ARCH_R+.010,.309),(cx+ARCH_R+.026,SILL+.005)]
            strip(l+" | "+axle+" arch shadow lip",[side(x,z,s,.001) for x,z in path],.0065,trim)
        points=[]
        for x in steps(1.07,2.285,72):
            u=fit(x,[(1.07,.79),(1.42,.69),(1.83,.67),(2.11,.73),(2.285,.84)])
            p=Vector(top(x,s*u));p.z+=.002
            points.append(p)
        strip(l+" | bonnet separation",points,.0023,joint)
    for sign,label in ((1,"nose"),(-1,"tail")):
        x=sign*FRONT
        patch(label+" | bumper",[[fascia(sign,u,SILL+(top(x,u)[2]-SILL)*t) for u in us] for t in steps(0,1,17)],pearl,sign<0)
    patch("Flat battery undertray",[[(nose_x(x,0,0),y,.165) for y in (-.68,.68)] for x in steps(-2.16,2.16,36)],black)
    for s,l in ((1,"L"),(-1,"R")):
        # Slender raised lower door tread sweeps into the rear haunch.
        shape=[(-1.08,.456),(-.84,.459),(-.56,.510),(-.41,.544),(.36,.528),(.60,.508),(.85,.450),(.97,.433),(.83,.432),(.51,.483),(-.40,.510),(-.77,.449)]
        side_panel(l+" | stamped lower door swage",shape,s,rocker,.004)
        path=[(-1.11,.255),(-.96,.224),(.87,.224),(1.12,.253),(.97,.285),(-1.03,.289)]
        side_panel(l+" | graphite sill",path,s,trim,.011)
        path=[(1.00,1.026),(.97,.91),(.969,.52),(.947,.350),(.901,.302),(.78,.29),(-.16,.292),(-.19,.322),(-.205,.65),(-.203,1.094)]
        strip(l+" | front door cut",[side(p.x,p.y,s,.004) for p in util.bez_path(path)],.0028,joint,True)
        path=[(-.207,1.095),(-.218,.66),(-.21,.336),(-.234,.296),(-.89,.298),(-.965,.35),(-1.013,.50),(-1.12,.65),(-1.28,.76),(-1.385,1.18)]
        strip(l+" | rear door cut",[side(p.x,p.y,s,.004) for p in util.bez_path(path)],.0028,joint,True)
        for n,x in enumerate((.023,-1.002)):
            z=fit(x,BELT)-.126
            side_panel(l+f" | flush handle {n+1}",[(x-.111,z-.011),(x+.09,z-.01),(x+.112,z+.007),(x+.102,z+.019),(x-.102,z+.018)],s,pearl,.010,.003)
        curve=[(-1.07,.47),(-.77,.49),(-.48,.514),(.33,.492),(.70,.449),(.93,.412)]
        strip(l+" | lower swage highlight",[side(x,z,s,.003) for x,z in curve],.0022,pearl)
        pos=side(-.86,.664,s,.012)
        ink(l+" | fleet door lettering",pos,(-s,0,0),(0,0,1),"WAYMO",.068,letter,1.57)
    side_panel("L | charging-port cover",[(1.023,.873),(1.041,.849),(1.10,.834),(1.182,.847),(1.25,.878),(1.277,.905),(1.278,.953),(1.25,.985),(1.202,.996),(1.07,.996),(1.027,.981),(1.008,.950)],1,pearl,.005,.0024)


def side_panel(name,outline,s,mat,offset=.005,rim=0):
    loop=util.bez_path(outline)
    cx=sum(p.x for p in loop)/len(loop)
    cz=sum(p.y for p in loop)/len(loop)
    rows=[[side(cx+(p.x-cx)*t,cz+(p.y-cz)*t,s,offset) for p in loop+[loop[0]]] for t in (.001,.35,.65,.85,1)]
    patch(name,rows,mat,s>0)
    if rim:
        strip(name+" surround",[side(p.x,p.y,s,offset) for p in loop],rim,joint,True)


# The greenhouse has independent longitudinal maps for the belt and header, enabling a
# long sloping windshield, short bonnet, long crossover roof and steep rear hatch.
BOTTOM_X=[(0,1.115),(.07,.965),(.42,-.205),(.70,-1.00),(.79,-1.36),(1,-2.087)]
TOP_X=[(0,.405),(.07,.32),(.42,-.38),(.67,-.91),(.75,-1.12),(.79,-1.23),(.83,-1.27),(1,-1.37)]
TOP_Z=[(0,1.450),(.12,1.501),(.4,1.524),(.60,1.512),(.73,1.48),(.84,1.448),(1,1.428)]
TOP_Y=[(0,.644),(.24,.692),(.55,.704),(.76,.674),(1,.612)]


def glazing(q,t,s,raise_by=0):
    xb=fit(q,BOTTOM_X)
    xt=fit(q,TOP_X)
    zb=fit(xb,BELT)-.006
    zt=fit(q,TOP_Z)
    yb=fit(xb,WIDTH)-.027
    yt=fit(q,TOP_Y)
    return (xb+(xt-xb)*t, s*(yb+(yt-yb)*t+.018*sin(pi*t)+raise_by), zb+(zt-zb)*t+.012*sin(pi*t)+raise_by*.35)


def wind(end,u,t,offset=0):
    q=0 if end==1 else 1
    xedge,yedge,zedge=glazing(q,t,1)
    bow=(1-abs(u)**2)
    return (xedge+end*(.154-.063*t)*bow+end*offset, u*yedge,zedge+(.016+.016*t)*bow+offset)


def canopy():
    group("02 | Glass and roof")
    for end,title in ((1,"Laminated windscreen"),(-1,"Heated tailgate backlight")):
        patch(title,[[wind(end,u,t) for u in steps(-1,1,42)] for t in steps(0,1,16)],glass,end<0)
        for t in (.016,.986):
            strip(title+" | ceramic frit",[wind(end,u,t,.002) for u in steps(-1,1,48)],.009,trim)
    for s,l in ((1,"L"),(-1,"R")):
        # Wrap the rear quarter with one continuous painted sail. A clipped,
        # tapered quarter-light and its weatherstrip sit in the sail panel.
        for q0,q1,label,mat in [(0,.029,"A pillar",pearl),(.029,.041,"A pillar lining",trim),(.041,.413,"front glass",glass),(.413,.438,"B pillar",trim),(.438,.671,"rear glass",glass),(.671,.682,"division bar",trim),(.682,1,"D pillar",pearl)]:
            patch(l+" | "+label,[[glazing(q,t,s) for q in steps(q0,q1,max(4,round((q1-q0)*100)))] for t in steps(0,1,14)],mat,s<0)
        outline=[(.682,.032),(.777,.040),(.799,.076),(.813,.15),(.820,.252),(.815,.349),(.805,.445),(.787,.565),(.758,.711),(.730,.82),(.704,.895),(.682,.934)]
        loop=util.bez_path(outline)
        def quarter(q,t,off):
            return glazing(q,t,s,off)
        mid=(.74,.40)
        patch(l+" | tapered quarter glass",[[quarter(mid[0]+(p.x-mid[0])*r,mid[1]+(p.y-mid[1])*r,.0035) for p in loop+[loop[0]]] for r in (.001,.3,.55,.78,1)],glass,s<0)
        strip(l+" | quarter window moulding",[quarter(p.x,p.y,.005) for p in loop],.0125,trim,True)
        strip(l+" | rear quarter bright perimeter",[quarter(p.x,p.y,.008) for p in loop if p.x>.690],.0037,alloy)
        for t,rad,mat in ((0,.008,trim),(.019,.003,alloy),(1,.009,pearl),(.973,.006,trim)):
            strip(l+" | side glass rail",[glazing(q,t,s,.003) for q in steps(.032,.682,65)],rad,mat)
        for q0 in (.041,.412,.438,.67,.682):
            strip(l+" | glass seal",[glazing(q0,t,s,.003) for t in steps(0,1,16)],.004,trim)
        for y0,y1 in ((.12,.83),):
            strip(l+" | wiper",[wind(1,s*u,.06+.054*(u-y0),.006) for u in steps(y0,y1,32)],.006,trim)
    rows=[]
    for q in steps(0,1,56):
        x,y,z=glazing(q,1,1)
        row=[]
        for u in steps(-1,1,32):
            bow=1-abs(u)**2
            xx=x+(.091*(1-q)**5-.091*q**5)*bow
            zz=z+(.032+.012*sin(pi*q))*bow
            row.append((xx,u*y,zz))
        rows.append(row)
    patch("Black panoramic roof",rows,roof)
    for s,l in ((1,"L"),(-1,"R")):
        strip(l+" | painted roof edge",[glazing(q,1,s,.001) for q in steps(0,1,84)],.017,pearl)
    spoiler=shell("Trailing roof spoiler",[(1.416,-1.438,0,.157,.599),(1.442,-1.46,0,.184,.653),(1.457,-1.465,0,.153,.638)],pearl,.47)
    strip("High stop light",[(-1.622+.043*u*u,.55*u,1.439-.008*u*u) for u in steps(-1,1,45)],.007,redled)


def interior():
    group("03 | Cabin silhouettes")
    box("Instrument panel",(.745,0,1.046),(.33,1.57,.11),.054,dash)
    for s in (-1,1):
        for x in (.018,-.76):
            b=box("Seat backrest",(x,s*.38,1.110),(.16,.417,.38),.077,interior_mat)
            b.rotation_euler.y=-.13
            box("Seat head restraint",(x-.034,s*.38,1.325),(.13,.229,.138),.051,interior_mat)
    strip("Steering wheel",[(.60+.045*sin(a),.395+.158*cos(a),1.126+.144*sin(a)) for a in steps(0,TAU,48)],.014,dash,True)
    box("Central instrument display",(.69,0,1.118),(.041,.30,.141),.014,trim)

def face_band(name,end,curve,mat,offset):
    """Conformal panel on a curved bumper, from left-to-right lower/upper edges."""
    rows=[]
    for u in steps(curve[0][0],curve[-1][0],max(12,int((curve[-1][0]-curve[0][0])*80))):
        a=fit(u,[(j[0],j[1]) for j in curve])
        b=fit(u,[(j[0],j[2]) for j in curve])
        rows.append([fascia(end,u,z,offset) for z in steps(a,b,4)])
    return patch(name,rows,mat,end>0)


def grille_ring(t):
    # Softly flattened upper edge and pinched lower corners.
    u=.531*sp(cos(t),.48)*(1+.042*sin(t))
    z=.613+.148*sp(sin(t),.58)
    return u,z


def front_rear():
    group("04 | Lamps, grilles and bumpers")
    contour=[grille_ring(a) for a in steps(0,TAU,112)[:-1]]
    patch("Recessed oval grille",[[fascia(1,u*k,.613+(z-.613)*k,.010) for u,z in contour+[contour[0]]] for k in (.001,.35,.7,1)],trim)
    strip("Satin grille surround",[fascia(1,u,z,.012) for u,z in contour],.009,alloy,True)
    # Build an open honeycomb lattice in a single editable mesh, clipped to the aperture.
    verts=[];faces=[]
    r=.021
    for j in range(-6,7):
        for i in range(-16,17):
            y=(i+(j%2)*.5)*sqrt(3)*r
            z=.62+j*1.5*r
            if (abs(y)/.424)**3.5+(abs(z-.613)/.128)**3.2>.85:
                continue
            base=len(verts)
            for rr in (r,r-.0020):
                for k in range(6):
                    a=k*pi/3+pi/2
                    verts.append(fascia(1,(y+rr*cos(a))/.838,z+rr*sin(a),.0175))
            for k in range(6):
                n=(k+1)%6
                faces.append((base+k,base+n,base+6+n,base+6+k))
    mesh("Grille | hexagonal bright edges",verts,faces,hexmat,False)
    ink("Nose | bonnet lettering",(2.304,0,.798),(0,1,0),(-.74,0,.67),"JAGUAR",.026,darkmetal,1.4)
    face_band("Lower cooling grille",1,[(-1,.288,.289),(-.82,.275,.310),(-.61,.273,.324),(-.42,.277,.398),(0,.274,.40),(.42,.277,.398),(.61,.273,.324),(.82,.275,.310),(1,.288,.289)],trim,.009)
    face_band("Front splitter edge",1,[(-1,.267,.278),(-.70,.252,.274),(0,.265,.278),(.70,.252,.274),(1,.267,.278)],trim,.013)
    for j in range(3):
        z=.300+.024*j
        strip("Cooling grille horizontal vane",[fascia(1,u,z+.003*u*u,.011) for u in steps(-.45,.45,32)],.002,hexmat)
    for s,l in ((1,"L"),(-1,"R")):
        sym=lambda vals: sorted((u*s,lo,hi) for u,lo,hi in vals)
        face_band(l+" | outer bumper intake",1,sym([(.803,.608,.616),(.824,.472,.63),(.851,.389,.635),(.869,.380,.637),(.899,.405,.632),(.922,.49,.625),(.928,.573,.613)]),trim,.008)
        frame=[(.533,.719,.724),(.576,.680,.752),(.69,.677,.764),(.82,.679,.777),(.95,.713,.779),(1.0,.745,.776)]
        face_band(l+" | front jewel lamp",1,sym(frame),trim,.008)
        for k,(u,z) in enumerate(((.624,.712),(.729,.722),(.823,.732))):
            path=[fascia(1,s*(u+.046*cos(a)),z+.029*sin(a),.0132) for a in steps(0,TAU,28)]
            strip(l+f" | projector bezel {k+1}",path,.0036,alloy,True)
            p=fascia(1,s*u,z,.0137)
            ellipsoid(l+" | projector optic",p,(.005,.019,.015),optic)
        border=[(.561,.735),(.589,.703),(.617,.695),(.761,.704),(.895,.728),(.977,.756)]
        strip(l+" | J-blade daytime running light",[fascia(1,s*u,z,.016) for u,z in border],.0038,led)
        side_panel(l+" | swept lamp housing",[(FRONT+.010,.745),(FRONT+.010,.776),(2.215,.814),(2.02,.867),(1.866,.890),(1.914,.837),(2.087,.778),(2.240,.744)],s,trim,.005)
        # Bridge both molded surfaces at the rounded corner, keeping the DRL uninterrupted.
        ue=.965
        seam_loop=[fascia(1,s*ue,.727,.010),fascia(1,s*ue,.777,.010),fascia(1,s,.776,.009),side(2.285,.797,s,.006),side(2.285,.759,s,.006),fascia(1,s,.745,.009)]
        cap(l+" | lamp corner transition",seam_loop,trim)
        strip(l+" | outer DRL joint",[fascia(1,s*.970,.754,.016),fascia(1,s*.993,.755,.013),side(FRONT,.752,s,.012)],.0052,led)
        strip(l+" | swept lamp blade",[side(x,z,s,.012) for x,z in [(FRONT,.752),(2.24,.758),(2.11,.785),(1.99,.830),(1.911,.859)]],.0052,led)
        strip(l+" | side amber marker",[side(x,z,s,.013) for x,z in [(1.884,.88),(1.937,.866),(1.984,.849)]],.006,amber)
        # The production lamps wrap as a single piece, rising onto the quarter.
        side_panel(l+" | tail lamp outer lens",[(REAR,.923),(REAR,.980),(-2.28,1.004),(-2.12,1.05),(-1.95,1.072),(-1.84,1.071),(-1.90,1.043),(-2.016,1.015),(-2.16,.976)],s,red,.007)
        strip(l+" | rear flank light signature",[side(x,z,s,.013) for x,z in [(REAR,.957),(-2.24,1.01),(-2.075,1.045),(-1.943,1.061),(-1.87,1.069)]],.0047,redled)
        face_band(l+" | dark rear lamp lens",-1,sym([(.363,.971,.974),(.39,.943,.981),(.47,.901,.979),(.60,.894,.977),(.68,.918,.975),(.83,.922,.979),(.94,.919,.981),(1,.922,.980)]),trim,.009)
        face_band(l+" | garnet rear light optic",-1,sym([(.39,.964,.969),(.45,.932,.966),(.48,.907,.965),(.60,.901,.963),(.68,.927,.966),(.88,.930,.971),(1,.936,.972)]),red,.013)
        upper=[(.38,.975),(.48,.976),(.64,.974),(.80,.976),(.96,.979),(1,.978)]
        lower=[(.425,.95),(.496,.915),(.591,.914),(.656,.933),(.78,.938),(1,.943)]
        for line in (upper,lower):
            strip(l+" | rear LED blade",[fascia(-1,s*u,z,.017) for u,z in line],.0043,redled)
        face_band(l+" | wrap transition",-1,sym([(.972,.920,.978),(1,.923,.980)]),red,.014)
        strip(l+" | lamp tip seal",[fascia(-1,s,.963,.009),side(REAR,.963,s,.012)],.008,red)
        face_band(l+" | rear bumper vertical vent",-1,sym([(.783,.592,.595),(.815,.577,.601),(.845,.550,.602),(.867,.510,.596),(.880,.498,.581),(.888,.519,.559)]),trim,.009)
    # Badge, full tailgate cut, broad lower triangular license recess and sculpted valance.
    hatch=[(-.706,1.018),(-.705,.908),(-.694,.712),(-.684,.655),(-.658,.628),(-.583,.611),(0,.607),(.583,.611),(.658,.628),(.684,.655),(.694,.712),(.705,.908),(.706,1.018)]
    for sgn,l in ((-1,"R"),(1,"L")):
        start=[(abs(u)*sgn,z) for u,z in hatch[8:]]
        pts=[]
        for (u,z),(v,h) in zip(start,start[1:]):
            pts += [fascia(-1,u+(v-u)*t,z+(h-z)*t,.014) for t in steps(0,1,8)]
        strip(l+" | tailgate reveal",pts,.003, joint)
    strip("Tailgate | lower shut line",[fascia(-1,u,.607+.017*abs(u/.64)**6,.011) for u in steps(-.64,.64,72)],.003,joint)
    face_band("Rear bumper | central license recess",-1,[(-.50,.402,.408),(-.44,.399,.470),(-.35,.400,.576),(-.31,.402,.593),(.31,.402,.593),(.35,.400,.576),(.44,.399,.470),(.50,.402,.408)],trim,.012)
    face_band("Rear | dark graphite valance",-1,[(-1,.270,.303),(-.91,.249,.401),(-.75,.243,.406),(0,.239,.401),(.75,.243,.406),(.91,.249,.401),(1,.270,.303)],trim,.013)
    meshrows=[]
    for u in steps(-.65,.65,55):
        z=.246+.037*max(0,1-(abs(u)/.58)**6)
        meshrows.append([fascia(-1,u,z,.015),fascia(-1,u,z-.055+.018*abs(u),.004)])
    patch("Rear | diffuser blade",meshrows,graphite)
    strip("Rear | diffuser bright edge",[row[0] for row in meshrows],.006,darkmetal)
    for u in (-.48,-.18,.18,.48):
        fin=box("Rear | undertray fin",(-2.23,u,.213),(.123,.014,.082),.005,trim)
        fin.rotation_euler.y=.16
    for sign in (-1,1):
        face_band("Rear | red bumper reflector",-1,sorted((u*sign,z0,z1) for u,z0,z1 in [(.62,.319,.325),(.68,.313,.327),(.81,.320,.338),(.84,.331,.340)]),red,.018)
    box("Rear fleet plate",(-2.352,0,.509),(.016,.290,.100),.008,white)
    ink("Rear fleet number",(-2.361,0,.513),(0,-1,0),(0,0,1),"X590  05",.038,letter)
    ink("Tailgate | marque script",(-2.329,0,.920),(0,-1,0),(0,0,1),"JAGUAR",.040,darkmetal,1.22)
    ink("Tailgate | model script",(-2.309,-.50,.670),(0,-1,0),(0,0,1),"I - P A C E",.021,darkmetal,1.10)



def wheels():
    global bone
    group("05 | Wheels and brakes")
    for cx,track,axle in AXLES:
        for s,l in ((1,"Left"),(-1,"Right")):
            bone="Wheel_"+axle+"_"+l
            name=axle+" "+l+" | "
            def point(r,a,h):
                # Fleet press wheels have a smaller graphite rim and deeper tire sidewall.
                if r<=.258:
                    r*=.882
                elif r<.365:
                    r=.258*.882+(r-.258)*(.365-.258*.882)/(.365-.258)
                return (cx+r*cos(a),s*(track+h),RADIUS+r*sin(a))
            def turn(label,prof,mat):
                ob=patch(name+label,[[point(r,a,h) for a in steps(0,TAU,72)] for r,h in prof],mat,s<0)
                tag(ob)
            turn("radial tire",[(.252,-.093),(.264,-.113),(.307,-.122),(.337,-.115),(.354,-.104),(.364,-.080),(.365,-.068),(.365,.068),(.364,.080),(.354,.104),(.337,.115),(.307,.122),(.264,.113),(.252,.093)],rubber)
            ribs=[]
            ribfaces=[]
            for n in range(88):
                for sidewall in (-1,1):
                    a=TAU*n/88+.021*sidewall
                    start=len(ribs)
                    ribs.extend([point(.354,a,sidewall*.106),point(.354,a+.004,sidewall*.106),point(.3642,a+.020,sidewall*.079),point(.365,a+.018,sidewall*.072)])
                    ribfaces.append((start,start+1,start+2,start+3))
            tag(mesh(name+"asymmetric shoulder grooves",ribs,ribfaces,tread,False))
            for j,r in enumerate((.277,.296,.320)):
                turn(f"moulding {j}",[(r,.120),(r+.001,.1225),(r+.002,.120)],rubber)
            turn("cast wheel barrel",[(.246,-.091),(.255,-.097),(.257,.099),(.253,.109),(.240,.111),(.232,.104),(.231,-.083)],graphite)
            turn("machined lip",[(.245,.109),(.251,.113),(.254,.110)],darkmetal)
            turn("disc brake",[(.055,.054),(.187,.054),(.191,.066),(.188,.070),(.055,.070)],brake)
            for rr in (.14,.167,.179):
                turn("disc wear ring",[(rr,.070), (rr+.0014,.0705)],darkmetal)
            cal=box(name+"fixed brake caliper",(cx-.151,s*(track+.074),RADIUS+.024),(.048,.037,.142),.015,brake)
            cal["rig_part"]="Vehicle_Base"
            verts=[];faces=[]
            for i in range(10):
                a=i*TAU/10
                for direction in (-1,1):
                    profile=[(.046,-.027),(.099,-.036),(.239,direction*.112-.016),(.249,direction*.115+.018),(.17,direction*.085+.030),(.064,.034)]
                    base=len(verts)
                    for h in (.083,.107):
                        for rr,theta in profile:
                            verts.append(point(rr,a+theta,h+.007*(1-rr/.25)))
                    faces.extend([tuple(base+k for k in range(6)),tuple(base+6+k for k in reversed(range(6)))])
                    for k in range(6):
                        n=(k+1)%6
                        faces.append((base+k,base+n,base+6+n,base+6+k))
            ob=tag(mesh(name+"forked graphite spokes",verts,faces,graphite,False))
            m=ob.modifiers.new("Machined spoke edges","BEVEL");m.width=.0017;m.segments=2
            ob.modifiers.new("Machined face normals","WEIGHTED_NORMAL")
            turn("hub medallion",[(.056,.106),(.058,.113),(.045,.122),(.029,.124),(.001,.124)],darkmetal)
            turn("oxblood centre cap",[(.021,.1241),(.021,.125),(.001,.125)],red)
            for i in range(5):
                a=TAU*i/5
                coords=[point(.040,a,.119)]
                mid=Vector(coords[0])
                v=[mid+Vector((.006*cos(t),0,.006*sin(t))) for t in steps(0,TAU,6)[:-1]]
                tag(cap(name+"wheel lug",v,alloy))
    bone="Vehicle_Base"


def sensors():
    group("06 | Fifth-generation Driver")
    # Four raised perimeter enclosures on body ends and a separate central nose instrument.
    for s,l in ((1,"L"),(-1,"R")):
        for cx,z,title in ((1.688,.952,"front"),(-2.075,.826,"rear")):
            outward=1 if cx>0 else -1
            # Main radome grows directly out of the fender. The outer IR glass bends
            # around the leading end; the rear units angle toward following traffic.
            def skin(label,sections,mat):
                rows=[]
                for t,w,h,cy,cz in sections:
                    rows.append([(cx+outward*t,s*(cy+w*sp(cos(a),.62)),z+cz+h*sp(sin(a),.62)) for a in steps(0,TAU,52)])
                patch(l+" "+title+" | "+label,rows,mat,s*outward<0)
                cap(l+" "+title+" | "+label+" closed end",rows[-1][:-1],mat)
            shape=[(-.231,.052,.050,.865,.001),(-.219,.078,.075,.878,.001),(-.184,.098,.099,.889,.004),(-.126,.105,.105,.890,.006),(.103,.107,.105,.891,.007),(.168,.102,.101,.890,.006),(.199,.086,.087,.893,.003),(.209,.063,.064,.896,.002)]
            skin("blended ceramic fairing",shape,pearl)
            # Rounded black instrument band sits slightly proud of painted shell.
            sensor_face(l+" "+title+" | side infrared window",cx+.003*outward,s*.991,z-.004,.326,.152,.024,s,sensor)
            sensor_face(l+" "+title+" | large peripheral lens",cx-.064*outward,s*1.006,z-.007,.136,.093,.007,s,deepopt)
            sensor_face(l+" "+title+" | short range lens",cx+.106*outward,s*1.006,z-.019,.049,.044,.007,s,deepopt)
            sensor_face(l+" "+title+" | aperture index",cx+.109*outward,s*1.007,z+.034,.044,.009,.006,s,graphite)
            # Terminal glass is visible head-on and from behind, as on the press car.
            rim=[]
            for a in steps(0,TAU,60):
                h=.075*sp(cos(a),.56)
                dz=.080*sp(sin(a),.53)
                at=cx+outward*(.220+.007*(1-(abs(h)/.075)**2))
                rim.append((at,s*(.902+h),z+dz))
            mid=(cx+outward*.229,s*.902,z)
            patch(l+" "+title+" | end-looking black optic",[[tuple(Vector(mid).lerp(Vector(pt),f)) for pt in rim] for f in (.001,.4,.75,1)],sensor,s*outward>0)
            strip(l+" "+title+" | enclosure border",rim,.0026,trim)
            frontx=cx+outward*.231
            ellipsoid(l+" "+title+" | end-looking lens",(frontx,s*.914,z+.012),(.008,.032,.025),deepopt)
            ellipsoid(l+" "+title+" | glint",(frontx+outward*.007,s*.924,z+.021),(.0018,.005,.003),optic)
            if cx<0:
                # Rear black sensor glass faces obliquely outward at the bumper corner.
                for i in range(12):
                    t=i/11
                    pt=(cx+outward*(.103+.105*t),s*(.991-.042*t**1.8),z-.001)
                    coords=[(pt[0],pt[1],pt[2]-v) for v in (.065-.014*t,-.066+.009*t)]
                    if i==0:
                        sweep=[]
                    sweep.append(coords)
                patch(l+" rear | swept corner glass",sweep,sensor,s<0)
            bx=cx+(-.225 if cx>0 else -.173)
            yy=s*(.928 if cx>0 else .850)
            zz=z+(.030 if cx>0 else .078)
            shell(l+" "+title+" | auxiliary sensor pedestal",[(zz-.100,bx,yy,.066,.047),(zz-.094,bx,yy,.071,.051),(zz+.072,bx,yy,.062,.052),(zz+.104,bx,yy,.066,.053),(zz+.114,bx,yy,.056,.049)],white,.59)
            sensor_face(l+" "+title+" | low auxiliary window",bx,yy+s*.052,zz-.068,.090,.040,.008,s,sensor)
            sensor_face(l+" "+title+" | vertical aperture",bx,yy+s*.052,zz+.007,.066,.094,.008,s,auxoptic)
            shell(l+" "+title+" | auxiliary pale-blue collar",[(zz+.065,bx,yy,.064,.055),(zz+.075,bx,yy,.064,.055)],auxoptic,.67)
        # Body-colour mirror, dark undershell and separate reflective face.
        strip(l+" | mirror pedestal",[(.665,s*.868,1.07),(.639,s*.935,1.087),(.604,s*.981,1.107)],.019,trim)
        shell(l+" | wing mirror lower",[(1.071,.609,s*.987,.066,.065),(1.085,.609,s*.987,.097,.077),(1.112,.603,s*.987,.100,.0825)],trim,.61)
        shell(l+" | wing mirror cap",[(1.102,.603,s*.987,.103,.0825),(1.133,.603,s*.987,.098,.081),(1.165,.601,s*.984,.077,.070),(1.176,.60,s*.981,.040,.044)],pearl,.62)
        box(l+" | aluminised mirror",(.507,s*.985,1.122),(.009,.133,.057),.004,alloy)
    box("Nose | forward sensor",(2.346,0,.612),(.091,.105,.192),.03,sensor)
    sensor_face("Nose | side sensor channel",2.346,.051,.627,.052,.077,.006,1,deepopt)
    p=(2.394,0,.640)
    ellipsoid("Nose | primary optical receiver",p,(.006,.030,.024),deepopt)
    ellipsoid("Nose | nested return optic",(2.399,0,.641),(.003,.013,.011),optic)
    box("Nose | blue perimeter status strip",(2.393,0,.586),(.004,.063,.007),.0015,blue)
    shell("Nose | lower collar",[(.461,2.349,0,.033,.043),(.482,2.349,0,.038,.046),(.559,2.349,0,.038,.046),(.566,2.349,0,.035,.042)],sensor,.71)
    # Carrier feet and the shallow aerodynamic pearl/black roof tray.
    for s,l in ((1,"L"),(-1,"R")):
        for x in (.115,-.788):
            foot=box(l+" | tray stanchion",(x,s*.599,1.541),(.146,.115,.165),.035,trim)
            foot.rotation_euler.x=s*.20
        rail=[(.22,s*.57,1.59),(-.33,s*.584,1.596),(-.90,s*.58,1.574),(-1.18,s*.549,1.518)]
        strip(l+" | carrier rail",rail,.029,trim)
        for tip in (rail[0],rail[-1]):
            ellipsoid(l+" | rounded rail end",tip,(.031,.029,.029),trim)
    shell("Roof | carrier shadow",[(1.578,-.239,0,.548,.394),(1.592,-.239,0,.606,.467),(1.613,-.239,0,.617,.486)],trim,.56)
    shell("Roof | broad faired carrier",[(1.592,-.239,0,.583,.441),(1.606,-.239,0,.618,.485),(1.667,-.239,0,.625,.494),(1.691,-.239,0,.604,.484),(1.706,-.239,0,.537,.442),(1.723,-.239,0,.414,.369)],pearl,.55)
    # Wide rounded black forward face extends onto the tray flanks.
    for s,l in ((1,"L"),(-1,"R")):
        rr=[]
        for a in steps(0,pi*.74,62):
            f=fit(a,[(0,1),(.63,1),(.97,.88),(1.23,.52),(1.55,.21),(pi*.74,.14)])
            rr.append([(-.239+.630*sp(cos(a),.55),s*.497*sp(sin(a),.55),1.662+v*.055*f) for v in (-1,1)])
        patch(l+" | carrier black glass wrap",rr,sensor,s<0)
    for n in range(5):
        y=n*.171-.342
        def tray_x(v):
            return -.239+.630*max(.001,1-(abs(v)/.497)**(2/.55))**(.55/2)
        width=.127 if abs(n-2)==2 else .111
        shape=[(y+width*.5*sp(cos(a),.44),1.656+.036*sp(sin(a),.52)) for a in steps(0,TAU,42)[:-1]]
        pts=[(tray_x(v)+.0025,v,z) for v,z in shape]
        patch("Roof | recessed carrier camera pocket "+str(n),[[(tray_x(y+(v-y)*f)+.003,y+(v-y)*f,1.656+(z-1.656)*f) for v,z in shape+[shape[0]]] for f in (.001,.48,1)],black)
        strip("Roof | camera pocket rim",pts,.0025,graphite,True)
        if n in (1,2,3):
            ellipsoid("Roof | carrier lens "+str(n),(tray_x(y)+.004,y,1.656),(.005,.016,.016),optic)
    def central(name,levels,mat):
        return shell(name,[(z,-.176,0,r,r) for z,r in levels],mat)
    central("Roof | turret footing",[(1.705,.310),(1.724,.307),(1.75,.281)],white)
    central("Roof | cobalt status band",[(1.744,.285),(1.756,.283),(1.773,.278)],blue)
    central("Roof | panoramic camera crown",[(1.769,.279),(1.790,.277),(1.869,.238),(1.883,.220)],sensor)
    for i in range(8):
        base=TAU*i/8
        def crownpoint(a,z,offset):
            r=fit(z,[(1.769,.279),(1.790,.277),(1.869,.238),(1.883,.220)])+offset
            return (-.176+r*cos(a),r*sin(a),z)
        contour=[(base+.227*sp(cos(a),.53),1.823+.032*sp(sin(a),.51)) for a in steps(0,TAU,42)]
        rows=[[crownpoint(base+(a-base)*f,1.823+(z-1.823)*f,.002) for a,z in contour] for f in (.001,.35,.70,1)]
        patch("Roof | recessed camera crown window "+str(i),rows,deepopt)
        strip("Roof | crown optical seal "+str(i),[crownpoint(a,z,.0026) for a,z in contour],.002,trim)
        for j,da in enumerate((-.068,.081)):
            a=base+da
            ring=[crownpoint(a+.051*cos(t),1.824+.013*sin(t),.0037) for t in steps(0,TAU,30)]
            strip("Roof | paired camera retainer "+str(i)+"."+str(j),ring,.0017,graphite)
            ellipsoid("Roof | optical element",crownpoint(a,1.824,.0040),(.006,.006,.0075),optic)
    central("Roof | 360 lidar dark drum",[(1.878,.146),(1.891,.146),(1.996,.131),(2.030,.126)],sensor)
    central("Roof | lidar cap",[(2.024,.128),(2.034,.126),(2.039,.118)],white)
    for i in range(4):
        a=i*pi/2
        right=(-sin(a),cos(a),0)
        ink("Roof | illuminated display "+str(i),(-.176+.142*cos(a),.142*sin(a),1.958),right,(0,0,1),"W",.108,teal,1)
    for key,xyz,yaw in [("roof_lidar",(-.176,0,1.958),0),("roof_camera_front",(.399,0,1.656),0),("perimeter_front_left",(1.688,1.006,.952),90),("perimeter_front_right",(1.688,-1.006,.952),-90),("perimeter_rear_left",(-2.075,1.006,.826),90),("perimeter_rear_right",(-2.075,-1.006,.826),-90)]:
        ob=bpy.data.objects.new("sensor_origin_"+key,None)
        util.assembly.objects.link(ob)
        ob.empty_display_type="ARROWS";ob.empty_display_size=.06
        ob.location=xyz
        ob.rotation_euler.z=yaw*pi/180
        ob["purpose"]="Visual reference point in metres"
        anchors[key]=xyz

def rig(components):
    group("07 | Wheel skeleton")
    data=bpy.data.armatures.new("I-PACE suspension")
    arm=bpy.data.objects.new("Armature",data)
    util.assembly.objects.link(arm)
    bpy.context.view_layer.objects.active=arm
    arm.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    base=data.edit_bones.new("Vehicle_Base")
    base.head=(0,0,0)
    base.tail=(.30,0,0)
    for x,track,axle in AXLES:
        for s,l in ((1,"Left"),(-1,"Right")):
            b=data.edit_bones.new("Wheel_"+axle+"_"+l)
            b.head=(x,s*track,RADIUS)
            b.tail=(x+.145,s*track,RADIUS)
            b.parent=base
    bpy.ops.object.mode_set(mode="OBJECT")
    data.display_type="STICK"
    bpy.context.view_layer.update()
    for ob in components:
        m=ob.matrix_world.copy()
        ob.parent=arm
        ob.parent_type="BONE"
        ob.parent_bone=ob.get("rig_part","Vehicle_Base")
        ob.matrix_world=m
    bpy.context.view_layer.update()
    return arm


def export(arm,components):
    # Bake a material-preserving runtime mesh for each rigid body. Keep source surfaces editable.
    pools={b.name: [[],[],[],[],[]] for b in arm.data.bones}
    graph=bpy.context.evaluated_depsgraph_get()
    for obj in components:
        if obj.type not in {"MESH","CURVE"}:
            continue
        ev=obj.evaluated_get(graph)
        me=ev.to_mesh()
        verts,faces,mats,ids,smooth=pools[obj.parent_bone]
        base=len(verts)
        verts.extend(tuple(ev.matrix_world @ v.co) for v in me.vertices)
        slots=[]
        for ma in me.materials:
            if ma not in mats:
                mats.append(ma)
            slots.append(mats.index(ma))
        me.calc_loop_triangles()
        for tri in me.loop_triangles:
            faces.append(tuple(base+i for i in tri.vertices))
            ids.append(slots[tri.material_index])
            smooth.append(me.polygons[tri.polygon_index].use_smooth)
        ev.to_mesh_clear()
    bpy.ops.object.select_all(action="DESELECT")
    arm.select_set(True)
    tmp=[]
    for key,(verts,faces,mats,ids,smooth) in pools.items():
        obj=mesh("SK_IPACE_"+("Chassis" if key=="Vehicle_Base" else key),verts,faces,mats[0])
        tmp.append(obj)
        for ma in mats[1:]:
            obj.data.materials.append(ma)
        obj.data.polygons.foreach_set("material_index",ids)
        obj.data.polygons.foreach_set("use_smooth",smooth)
        weight=obj.vertex_groups.new(name=key)
        weight.add(list(range(len(verts))),1,"REPLACE")
        mod=obj.modifiers.new("Rigid skin","ARMATURE")
        mod.object=arm
        obj.parent=arm
        obj.select_set(True)
    bpy.context.view_layer.objects.active=arm
    bpy.ops.export_scene.fbx(filepath=str(OUT/"waymo-ipace.fbx"),use_selection=True,
        object_types={"ARMATURE","MESH"},add_leaf_bones=False,use_armature_deform_only=True,
        bake_anim=False,axis_forward="-Y",axis_up="Z",primary_bone_axis="X",secondary_bone_axis="Z",apply_unit_scale=True)
    for ob in components:
        if ob.type=="EMPTY":
            ob.select_set(True)
    bpy.ops.export_scene.gltf(filepath=str(OUT/"waymo-ipace.glb"),export_format="GLB",use_selection=True,
        export_extras=True,export_animations=False,export_skins=True,export_yup=True,export_all_influences=False)
    for ob in tmp:
        data=ob.data
        bpy.data.objects.remove(ob,do_unlink=True)
        bpy.data.meshes.remove(data)
    bpy.ops.object.select_all(action="DESELECT")


def main():
    global scene,pearl,white,trim,rocker,glass,roof,joint,alloy,hexmat,darkmetal,interior_mat,dash,letter,sensor,deepopt,auxoptic,black,graphite,brake,rubber,tread,led,red,redled,optic,amber,blue,teal
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene=bpy.context.scene
    util.scene=scene
    scene.unit_settings.system="METRIC"
    pearl=material("Fuji white | clearcoat",(.82,.855,.855),.15,.248,.45)
    rocker=material("Lower door return",(.747,.780,.787),.17,.27,.37)
    white=material("Ceramic sensor radome",(.84,.865,.875),.08,.28,.21)
    trim=material("Piano black trim",(.012,.014,.016),.32,.24,.32)
    glass=material("Smoked solar glass",(.051,.079,.097),.20,.095,.29)
    bs=glass.node_tree.nodes.get("Principled BSDF")
    bs.inputs["Transmission Weight"].default_value=.53
    bs.inputs["IOR"].default_value=1.45
    roof=material("Bonded black roof glass",(.020,.033,.044),.46,.16,.4)
    joint=material("Panel separation",(.10,.111,.118),.1,.43)
    alloy=material("Polished aluminium",(.50,.55,.57),.89,.21)
    hexmat=material("Grille graphite metal",(.082,.094,.102),.65,.34)
    darkmetal=material("Wheel diamond-cut charcoal",(.18,.19,.19),.86,.25)
    interior_mat=material("Graphite upholstery",(.078,.080,.079),.02,.76)
    dash=material("Dark cabin",(.027,.030,.033),.05,.57)
    letter=material("Fleet graphite lettering",(.069,.074,.078),.06,.43)
    sensor=material("IR transparent black polymer",(.010,.013,.016),.27,.19,.48)
    deepopt=material("Perimeter sapphire cover",(.006,.009,.011),.30,.12,.35)
    auxoptic=material("Auxiliary optical polymer",(.24,.31,.355),.37,.23,.22)
    black=material("Deep aperture",(.007,.009,.012),.05,.52)
    graphite=material("Gloss anthracite alloy",(.039,.042,.043),.69,.29,.16)
    brake=material("Ventilated steel",(.094,.103,.108),.75,.42)
    rubber=material("Carbon tyre compound",(.017,.018,.019),.0,.68)
    tread=material("Shoulder groove rubber",(.009,.010,.011),0,.72)
    led=material("Ice-white light pipe",(.80,.91,1),.18,.20,.1,1.5)
    red=material("Garnet lens",(.27,.009,.016),.3,.19,.4)
    redled=material("Rear lamp filament",(.70,.022,.021),.17,.23,.2,.55)
    amber=material("Amber marker",(.66,.20,.021),.33,.21)
    optic=material("Blue optical glass",(.039,.075,.112),.64,.13,.57)
    blue=material("Cobalt status collar",(.010,.141,.72),.12,.24,.20,.12)
    teal=material("Turquoise rider display",(.015,.79,.74),.08,.24,.1,.73)
    body();canopy();interior();front_rear();wheels();sensors()
    bpy.context.view_layer.update()
    components=list(scene.objects)
    arm=rig(components)
    if "--preview" not in sys.argv:
        export(arm,components)
    util.create_studio()
    studio=util.assembly
    studio.name="90 | Photography studio"
    floor=studio.objects["Cyclorama"]
    floor.location.z=.062
    floor.data.materials[0].node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value=(.160,.192,.219,1)
    # Physical curved sweep keeps the background continuous above the floor horizon.
    forward=Vector((-0.71,-0.704,0)).normalized()
    across=Vector((-forward.y,forward.x,0))
    profile=[(5.3,0)]
    for a in steps(0,pi/2,32):
        profile.append((5.5+3.7*sin(a),3.7*(1-cos(a))))
    profile.append((9.2,15))
    sweep=patch("Curved studio backdrop",[[tuple(forward*d+across*w+Vector((0,0,z-.001))) for w in (-35,35)] for d,z in profile],floor.data.materials[0],True)
    sweep.hide_set(True)
    scene.camera.location=(8.62,8.43,2.89)
    util.aim(scene.camera,(0,0,.97))
    scene.camera.data.lens=74
    scene.render.resolution_x=1080
    scene.render.resolution_y=656
    scene.cycles.samples=28
    scene.cycles.device="CPU"
    scene.cycles.use_denoising=True
    scene.render.threads=2
    scene.render.filepath="//studio.png"
    scene["asset_name"]="Waymo Jaguar I-PACE | fifth-generation exterior study"
    scene["wheel_radius_m"]=RADIUS
    scene["axis_convention"]="metres; Blender +X forward, +Y left, +Z up; axle midpoint on ground"
    scene["reference_limitation"]="Exterior visual reconstruction derived from public photographs, mechanically approximated."
    bpy.context.preferences.filepaths.save_version=0
    if "--preview" in sys.argv:
        import tempfile
        scene.render.resolution_percentage=63
        scene.cycles.samples=3
        scene.render.filepath=str(Path(tempfile.gettempdir())/"ipace-check.png")
    else:
        bpy.ops.wm.save_as_mainfile(filepath=str(OUT/"waymo-ipace.blend"),compress=True)
    if "--preview" in sys.argv or "--render" in sys.argv:
        bpy.ops.render.render(write_still=True)


if __name__=="__main__":
    main()
