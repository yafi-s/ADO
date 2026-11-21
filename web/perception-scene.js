import * as T from './vendor/city-view/three-view.js';
import {buildRoadUser, disposeRoadUserGeometry, plausibleSize, TRACK_TONE} from './road-users.js';
import {ObjectMemory, ReplayMemory, laneAxes} from './object-memory.mjs';
import {SignalHeads, townFeatures} from './signal-heads.js';
import {cornerReturns} from './sensor-returns.mjs';

const V = (x = 0, y = 0, z = 0) => new T.Vector3(x, y, z);
const finite = Number.isFinite;
const coord = (p, n = 2) => Array.isArray(p) && p.length >= n && p.slice(0,n).every(finite);
const clamp = (v, min, max) => Math.min(max, Math.max(min, v));
const rad = Math.PI / 180;
const shortest = (start, end) => Math.atan2(Math.sin(end-start), Math.cos(end-start));
const C = value => new T.Color(value);
const paint = (color, properties = {}) => new T.MeshStandardMaterial({color, roughness:.62, metalness:.1, ...properties});
const ink = color => new T.MeshBasicMaterial({color, toneMapped:false});
function add(parent, geometry, material, x = 0, y = 0, z = 0) {
  const item = new T.Mesh(geometry, material);
  item.position.set(x,y,z);
  parent.add(item);
  return item;
}
function release(root) {
  const geometries = new Set(), materials = new Set(), textures = new Set();
  root?.traverse(obj => {
    if (obj.geometry) geometries.add(obj.geometry);
    if (obj.material) for (const material of [].concat(obj.material)) materials.add(material);
  });
  for (const material of materials) for (const value of Object.values(material)) if (value?.isTexture) textures.add(value);
  for (const item of [...textures, ...materials, ...geometries]) item.dispose();
  root?.removeFromParent();
}

// Two display palettes. Night keeps the original values; day is a light studio
// with clay-grey road users. Linear RGB triples feed the shaders directly, hex
// strings are sRGB. Only colour changes between them, never what is drawn.
const PALETTES = {
  night: {
    sky:'#223340', floorFar:[.019,.034,.044], floorNear:[.032,.050,.061], grid:[.004,.008,.008],
    roadEdge:[.066,.088,.105], roadMid:[.071,.095,.112], marking:[.44,.54,.58], centre:[.65,.47,.16], kerb:[.20,.29,.32], curb:'#455661',
    pathFar:[.024,.35,.38], pathNear:[.16,.81,.68],
    wallDark:[.026,.041,.058], wallLit:[.074,.102,.126], wallEdge:[.030,.052,.066],
    shadow:'#061012', shadowOpacity:.57, lidar:'#8cced2', radar:'#e6b979',
    vehicle:'#7792b8', rider:'#f2b45e', person:'#f4aa51', unknown:'#8597a4',
    glass:'#233d52', lamp:'#e6ecff', tail:'#e9877c', tyre:'#232f39', hub:'#637885', glow:.025, rim:.14, exposure:1.04,
    signalBody:'#2a3644', signalOff:'#141c26', red:'#ff453a', amber:'#ffb01f', green:'#2fe37a'
  },
  day: {
    sky:'#edf1f3', floorFar:'#dde5e7', floorNear:'#e3e9e9', grid:[-.014,-.014,-.014],
    roadEdge:'#eff2f4', roadMid:'#f2f4f6', marking:'#8493a0', centre:'#bd8f41', kerb:'#acb8bf', curb:'#c6d2d4',
    pathFar:'#7fa8f5', pathNear:'#1f5fe8',
    wallDark:'#cdd2da', wallLit:'#e9ecf0', wallEdge:[-.035,-.033,-.03],
    shadow:'#1d2633', shadowOpacity:.34, lidar:'#5578b4', radar:'#d9641a',
    vehicle:'#8e9da9', rider:'#bf762e', person:'#c96022', unknown:'#9daab2',
    glass:'#505f71', lamp:'#f6f7f9', tail:'#c97875', tyre:'#4b5560', hub:'#828f9b', glow:0, rim:0, exposure:1.0,
    signalBody:'#5d6674', signalOff:'#39414d', red:'#e5352b', amber:'#e39a0c', green:'#17a85a'
  }
};
const TONES = ['sky','floorFar','floorNear','grid','roadEdge','roadMid','marking','centre','kerb','curb','pathFar','pathNear',
  'wallDark','wallLit','wallEdge','shadow','lidar','radar'];
const displayMode = () => document.documentElement.dataset.display === 'day' ? 'day' : 'night';
const tone = (target, value) => Array.isArray(value) ? target.setRGB(...value) : target.set(value);

const worldVertex = `
varying vec3 vWorld;
void main(){
  vec4 p = modelMatrix * vec4(position,1.);
  vWorld = p.xyz;
  gl_Position = projectionMatrix * viewMatrix * p;
}`;
const endColor = '\n#include <colorspace_fragment>\n';
const haze = `
vec3 aerial(vec3 color, vec3 p) {
  float dist = distance(cameraPosition, p);
  return mix(color, uSky, smoothstep(48., 210., dist)*.96);
}`;
function floorMaterial(tones) {
  return new T.ShaderMaterial({
    uniforms:{uSky:{value:tones.sky}, uEgo:{value:V()}, uFar:{value:tones.floorFar}, uNear:{value:tones.floorNear},
      uGrid:{value:tones.grid}}, vertexShader:worldVertex,
    fragmentShader:`
      uniform vec3 uSky,uEgo,uFar,uNear,uGrid; varying vec3 vWorld;
      ${haze}
      void main(){
        vec2 g = vWorld.xz / 5.;
        vec2 dd = abs(fract(g + .5) - .5) / max(fwidth(g),vec2(.0001));
        float line = 1.-smoothstep(0.,1.,min(dd.x,dd.y));
        float proximity = 1.-smoothstep(9.,78.,distance(vWorld.xz,uEgo.xz));
        vec3 base = mix(uFar,uNear, proximity);
        base += uGrid*line*proximity;
        gl_FragColor = vec4(aerial(base,vWorld),1.);${endColor}
      }`
  });
}

// Each polyline is a surveyed driving-lane centre. Index the waypoints to locate
// their shared boundaries and separate interior markings from the outside curb.
function roadIndex(roads) {
  const cells = new Map(), key = (x,z) => Math.floor(x/6)+','+Math.floor(z/6);
  roads.forEach((lane,id) => lane.forEach((p,i) => {
    const a=lane[Math.max(0,i-1)], b=lane[Math.min(lane.length-1,i+1)];
    const len=Math.hypot(b[0]-a[0],b[1]-a[1])||1;
    const k=key(p[0],p[1]); if (!cells.has(k)) cells.set(k,[]);
    cells.get(k).push([id,p,(b[0]-a[0])/len,(b[1]-a[1])/len]);
  }));
  const near = (x,z,id,test) => {
    for (let i=-1;i<=1;i++) for (let j=-1;j<=1;j++) {
      for (const entry of cells.get(key(x+i*6,z+j*6)) || []) {
        if (entry[0]!==id && (entry[1][0]-x)**2+(entry[1][1]-z)**2<2.7 && test(entry)) return true;
      }
    }
    return false;
  };
  return {
    // Another lane runs alongside this edge, so the line between them is a lane divider.
    beside: (x,z,id) => near(x,z,id,()=>true),
    // Another lane crosses or peels away here, which only happens inside a junction.
    crossing: (x,z,id,dx,dz) => near(x,z,id,entry=>Math.abs(entry[2]*dx+entry[3]*dz)<.975)
  };
}
// A ribbon along each lane centre. `reach` continues every lane past both ends so
// neighbouring segments overlap instead of leaving a seam. Markings stop inside junctions.
// `paints[i]` describes how lane i's two edges are painted, when the map records it.
function surfaceGeometry(lines, halfWidth, elevation = 0, index = null, reach = 0, paints = null) {
  const xyz=[],uv=[],inside=[],calm=[],paint=[],broken=[],indices=[];
  lines.forEach((source,id) => {
    const measuredWidth=paints?.[id]?.width;
    const laneHalf=finite(measuredWidth)&&measuredWidth>0?measuredWidth/2:halfWidth;
    const heading=(i)=>{
      const a=source[Math.max(0,i-1)], b=source[Math.min(source.length-1,i+1)];
      const len=Math.hypot(b[0]-a[0],b[1]-a[1])||1;
      return [(b[0]-a[0])/len,(b[1]-a[1])/len];
    };
    let junction=source.map(()=>0);
    if(index){
      const hit=source.map((p,i)=>index.crossing(p[0],p[1],id,...heading(i)));
      junction=source.map((_,i)=>hit.slice(Math.max(0,i-3),i+4).some(Boolean)?1:0);
    }
    let line=source;
    if(reach>0){
      const [sx,sz]=heading(0), [ex,ez]=heading(source.length-1), first=source[0], last=source[source.length-1];
      line=[[first[0]-sx*reach,first[1]-sz*reach,first[2]],...source,[last[0]+ex*reach,last[1]+ez*reach,last[2]]];
      junction=[1,...junction,1];
    }
    const offset=xyz.length/3;
    let s=0;
    line.forEach(([x,z,y],i) => {
      const a=line[Math.max(0,i-1)], b=line[Math.min(line.length-1,i+1)];
      const dx=b[0]-a[0], dz=b[1]-a[1], len=Math.hypot(dx,dz)||1;
      const nx=-dz/len, nz=dx/len;
      if(i) s+=Math.hypot(x-a[0],z-a[1]);
      for(const sign of [-1,1]) {
        xyz.push(x+sign*laneHalf*nx, y+elevation,z+sign*laneHalf*nz);
        uv.push(sign,s);
        inside.push(index?.beside(x+sign*laneHalf*2*nx,z+sign*laneHalf*2*nz,id)?1:0);
        calm.push(junction[i]);
        // -1: not recorded, so the shader guesses. 0: unpainted. 1: white. 2: yellow.
        const edge=paints?.[id]?paints[id][sign>0?'plus':'minus']:undefined;
        paint.push(edge===undefined?-1:!edge?0:edge.color==='yellow'?2:1);
        broken.push(edge?.line==='broken'?1:0);
      }
      if(i) {const k=offset+i*2;indices.push(k-2,k-1,k,k,k-1,k+1);}
    });
  });
  const geo=new T.BufferGeometry();
  geo.setAttribute('position',new T.Float32BufferAttribute(xyz,3));
  geo.setAttribute('uv',new T.Float32BufferAttribute(uv,2));
  geo.setAttribute('inner',new T.Float32BufferAttribute(inside,1));
  geo.setAttribute('calm',new T.Float32BufferAttribute(calm,1));
  geo.setAttribute('paint',new T.Float32BufferAttribute(paint,1));
  geo.setAttribute('broken',new T.Float32BufferAttribute(broken,1));
  geo.setIndex(indices);
  return geo;
}
const ribbonVertex = `
attribute float inner, calm, paint, broken;
varying vec2 vLane; varying vec3 vWorld; varying float vInner, vCalm, vPaint, vBroken;
void main(){
  vLane=uv; vInner=inner; vCalm=calm; vPaint=paint; vBroken=broken;
  vec4 p=modelMatrix*vec4(position,1.); vWorld=p.xyz;
  gl_Position=projectionMatrix*viewMatrix*p;
}`;
function streetMaterial(tones) {
  return new T.ShaderMaterial({
    uniforms:{uSky:{value:tones.sky}, uEdge:{value:tones.roadEdge}, uMid:{value:tones.roadMid}, uMark:{value:tones.marking},
      uCentre:{value:tones.centre}, uKerb:{value:tones.kerb}},
    side:T.DoubleSide,vertexShader:ribbonVertex,
    fragmentShader:`uniform vec3 uSky,uEdge,uMid,uMark,uCentre,uKerb; varying vec2 vLane; varying vec3 vWorld;
      varying float vInner,vCalm,vPaint,vBroken;
      ${haze}
      void main(){
        float a=abs(vLane.x), aa=min(.12,fwidth(a)*.7+.001);
        float lane=smoothstep(.96-aa,.96+aa,a);
        float d=mod(vLane.y,7.);
        float dash=smoothstep(.05,.25,d)*(1.-smoothstep(2.65,2.85,d));
        // Paint the edge as the map records it. Without a record, dash the line between two lanes.
        float recorded=step(-.5,vPaint);
        float strength=mix(mix(.6,dash,step(.58,vInner)), step(.5,vPaint)*mix(1.,dash,step(.5,vBroken)), recorded);
        float marking=strength*lane*(1.-smoothstep(.05,.7,vCalm));
        vec3 base=mix(uEdge,uMid,1.-a);
        vec3 color=mix(base,mix(uMark,uCentre,step(1.5,vPaint)),marking*.85);
        // An unpainted outer edge meets the kerb: a fine line keeps the carriageway's outline crisp.
        float kerb=recorded*(1.-step(.5,vPaint))*(1.-step(.58,vInner))*lane*(1.-smoothstep(.05,.7,vCalm));
        color=mix(color,uKerb,kerb*.8);
        gl_FragColor=vec4(aerial(color,vWorld),1.);${endColor}
      }`
  });
}
// Marked crosswalks, as the road description outlines them. These towns paint a crossing as two
// lines across the road, one along each long side of its outline, and so does this.
function crossingGeometry(crossings) {
  const xyz=[],across=[],indices=[];
  for(const corners of crossings){
    if(corners.length!==4||!corners.every(p=>coord(p,3)))continue;
    const [a,b,c]=corners, first=Math.hypot(b[0]-a[0],b[1]-a[1]), second=Math.hypot(c[0]-b[0],c[1]-b[1]);
    // Metres from one long side, and the crossing's width, at each corner.
    const along=first>=second, width=along?second:first, offset=xyz.length/3;
    corners.forEach((p,i)=>{
      xyz.push(p[0],p[2]+.016,p[1]);
      across.push((along?i>=2:i===0||i===3)?width:0,width);
    });
    indices.push(offset,offset+1,offset+2,offset,offset+2,offset+3);
  }
  const geometry=new T.BufferGeometry();
  geometry.setAttribute('position',new T.Float32BufferAttribute(xyz,3));
  geometry.setAttribute('across',new T.Float32BufferAttribute(across,2));
  geometry.setIndex(indices);
  return geometry;
}
function crossingMaterial(tones) {
  return new T.ShaderMaterial({
    uniforms:{uSky:{value:tones.sky},uMark:{value:tones.marking}},transparent:true,depthWrite:false,side:T.DoubleSide,
    vertexShader:`attribute vec2 across; varying vec2 vAcross; varying vec3 vWorld;
      void main(){vAcross=across;vec4 p=modelMatrix*vec4(position,1.);vWorld=p.xyz;gl_Position=projectionMatrix*viewMatrix*p;}`,
    fragmentShader:`uniform vec3 uSky,uMark; varying vec2 vAcross; varying vec3 vWorld;
      ${haze}
      void main(){
        float d=min(vAcross.x,vAcross.y-vAcross.x), w=clamp(fwidth(vAcross.x),.001,.3);
        float line=1.-smoothstep(.26-w,.26+w,d);
        gl_FragColor=vec4(aerial(uMark,vWorld),line*.8);${endColor}
      }`
  });
}
function pathMaterial(tones) {
  return new T.ShaderMaterial({
    uniforms:{uSky:{value:tones.sky},uStation:{value:0},uOverview:{value:0},uPreviewSpan:{value:0},uFar:{value:tones.pathFar},uNear:{value:tones.pathNear}},
    transparent:true,side:T.DoubleSide,depthWrite:false,vertexShader:ribbonVertex,
    fragmentShader:`uniform vec3 uSky,uFar,uNear; uniform float uStation,uOverview,uPreviewSpan;
      varying vec2 vLane; varying vec3 vWorld;
      ${haze}
      void main() {
        float along=vLane.y-uStation, a=abs(vLane.x);
        float width=mix(.97,.58,smoothstep(2.,72.,along));
        if(uPreviewSpan>0.)width*=.37;
        float aa=clamp(fwidth(a),.005,.13);
        float mask=1.-smoothstep(width-.115-aa,width+aa,a);
        float edge=smoothstep(width-.18-aa,width-.035+aa,a);
        float arm=abs(mod(vLane.y+.68*a+4.75,9.5)-4.75);
        float chevron=(1.-smoothstep(.07,.12+fwidth(vLane.y)*.6,arm))*(1.-smoothstep(.34,.45,a));
        float nearEgo=exp(-max(0.,along)/33.);
        float ahead=smoothstep(-4.,-.4,along)*(1.-smoothstep(65.,108.,along));
        float energy=mix(ahead*(.18+.82*nearEgo),.32+.68*exp(-abs(along)/42.),uOverview);
        float opacity=energy*mask*(.52+.29*edge+.17*chevron);
        // The surrounding destination route resumes beyond the selected maneuver.
        if(uPreviewSpan>0.)opacity*=.58*max(1.-smoothstep(-6.,-2.,along),smoothstep(uPreviewSpan-1.,uPreviewSpan+3.,along));
        vec3 color=mix(uFar,uNear,.17*nearEgo+max(edge,chevron)*.81);
        gl_FragColor=vec4(aerial(color,vWorld),opacity);${endColor}
      }`
  });
}
function architecture(boxes, tones, egoPosition) {
  const material=new T.ShaderMaterial({
    uniforms:{uSky:{value:tones.sky},uEgo:{value:egoPosition},uOverview:{value:0},
      uDark:{value:tones.wallDark},uLit:{value:tones.wallLit},uRim:{value:tones.wallEdge}},
    transparent:true,
    vertexShader:`varying vec3 vWorld,vLocal,vNormal;
      void main(){ vLocal=position;vec4 p=instanceMatrix*vec4(position,1.);
        vNormal=normalize(mat3(instanceMatrix)*normal);
        vWorld=p.xyz; gl_Position=projectionMatrix*viewMatrix*p; }`,
    fragmentShader:`uniform vec3 uSky,uEgo,uDark,uLit,uRim; uniform float uOverview; varying vec3 vWorld,vLocal,vNormal;
      ${haze}
      void main(){
        vec3 ax=abs(vLocal);
        float sideEdge=min(.5-ax.x,.5-ax.z);
        float sideDist=max(.5-ax.x,.5-ax.z);
        float e=vNormal.y>.6?sideEdge:min(.5-ax.y,sideDist);
        float edge=1.-smoothstep(.001, .007+fwidth(e),e);
        float light=clamp(.4+dot(vNormal,normalize(vec3(-.4,1.,-.25)))*.5,0.,1.);
        vec3 c=mix(uDark,uLit,light)+edge*uRim;
        float fade=mix(smoothstep(8.,20.,distance(vWorld.xz,uEgo.xz)),1.,uOverview);
        if(fade<.015) discard;
        gl_FragColor=vec4(aerial(c,vWorld),fade);${endColor}
      }`
  });
  const result=new T.InstancedMesh(new T.BoxGeometry(),material,boxes.length);
  const obj=new T.Object3D();
  boxes.forEach((box,i)=>{
    const [x,y,z]=box.center, [sx,sy,sz]=box.extent;
    obj.position.set(x,z,y); obj.scale.set(sx*2,sz*2,sy*2);obj.rotation.y=-(box.yaw||0)*rad;
    obj.updateMatrix();result.setMatrixAt(i,obj.matrix);
  });
  return result;
}

function projectedPatch(color, solid = false) {
  return new T.ShaderMaterial({
    uniforms:{uColor:{value:color?.isColor?color:C(color)},uFresh:{value:1},uAlpha:{value:1},uStrength:{value:.57},uSelected:{value:0},uLastSeen:{value:0}},
    transparent:true, depthWrite:false,side:T.DoubleSide,
    vertexShader:`varying vec2 vUv;void main(){vUv=uv*2.-1.;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}`,
    fragmentShader:`uniform vec3 uColor;uniform float uFresh,uAlpha,uStrength,uSelected,uLastSeen;varying vec2 vUv;
      void main() {
        float d=length(vUv);
        ${solid ? 'float opacity=(1.-smoothstep(.38,1.,d))*uStrength;' : `
          float border=abs(length(max(abs(vUv)-vec2(.50,.34),0.))-.22);
          float footprint=(1.-smoothstep(.013,.036+.035*uSelected+fwidth(border),border));
          float dash=step(.27,fract((atan(vUv.y,vUv.x)+3.142)*3.18));
          float opacity=footprint*mix(mix(.54*(.35+.65*uFresh),.9*dash,uLastSeen)*uAlpha,1.,uSelected);`}
        gl_FragColor=vec4(mix(uColor,vec3(.1,.92,.75),uSelected),opacity);${endColor}
      }`
  });
}
function patch(parent,length,width,mat,height) {
  const p=add(parent,new T.PlaneGeometry(length,width),mat,0,height,0);
  p.rotation.x=-Math.PI/2;
  return p;
}
function cloud(radar, color) {
  const geometry=new T.BufferGeometry();
  const material=new T.ShaderMaterial({
    transparent:true,depthWrite:false,
    uniforms:{uPixel:{value:1},uColor:{value:color}},
    vertexShader:`uniform float uPixel;void main(){
      vec4 view=modelViewMatrix*vec4(position,1.);
      gl_Position=projectionMatrix*view;
      gl_PointSize=uPixel*clamp(${radar?'66.':'46.'}/max(1.,-view.z),${radar?'3.9':'1.9'},${radar?'8.':'4.5'});
    }`,
    fragmentShader:`uniform vec3 uColor;void main(){
      float d=length(gl_PointCoord*2.-1.);
      float a=${radar?'(1.-smoothstep(.09,.25,abs(d-.68)))*.70 + (1.-smoothstep(.04,.25,d))*.86':'(1.-smoothstep(.35,1.,d))*.66'};
      gl_FragColor=vec4(uColor,a);${endColor}
    }`
  });
  const points=new T.Points(geometry,material);
  points.frustumCulled=false;
  points.renderOrder=4;
  return points;
}
function fillReturns(points,returns,height) {
  const values=Array.isArray(returns)?returns:[];
  let attr=points.geometry.getAttribute('position');
  if(!attr||attr.count<values.length) {
    attr=new T.BufferAttribute(new Float32Array(Math.max(32,values.length)*3),3);
    attr.setUsage(T.DynamicDrawUsage);
    // Keep one reusable GPU buffer as long as capacity is sufficient.
    points.geometry.dispose();points.geometry.setAttribute('position',attr);
  }
  let n=0;
  for(const p of values)if(coord(p,height==null?3:2))attr.setXYZ(n++,p[0],height??p[2],p[1]);
  points.geometry.setDrawRange(0,n);attr.needsUpdate=true;
}

function makeTrail(color) {
  const geo=new T.BufferGeometry(),positions=new T.BufferAttribute(new Float32Array(30*3),3),alphas=new T.BufferAttribute(new Float32Array(30),1);
  positions.setUsage(T.DynamicDrawUsage);alphas.setUsage(T.DynamicDrawUsage);
  geo.setAttribute('position',positions);geo.setAttribute('opacity',alphas);geo.setDrawRange(0,0);
  const material=new T.ShaderMaterial({
    uniforms:{uColor:{value:C(color)},uAlpha:{value:1}},transparent:true,depthWrite:false,
    vertexShader:`attribute float opacity;varying float vOpacity;void main(){vOpacity=opacity;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}`,
    fragmentShader:`uniform vec3 uColor;uniform float uAlpha;varying float vOpacity;void main(){gl_FragColor=vec4(uColor,vOpacity*uAlpha);${endColor}}`
  });
  const item=new T.Line(geo,material);item.frustumCulled=false;
  return item;
}
// Samples are world-space measurements, so ego motion cannot produce a trail.
// Only draw travelled ground, bounded by the tracker's current speed estimate.
function reviseTrail(trail, history, now, speed) {
  const geometry=trail.geometry;
  geometry.setDrawRange(0,0);
  if(history.length<2||!finite(speed)||speed<=0)return;
  const first=history[0],last=history.at(-1);
  if(last.t-first.t<.14||Math.hypot(last.x-first.x,last.z-first.z)<Math.max(.20,speed*.09))return;
  const positions=geometry.attributes.position,alphas=geometry.attributes.opacity;
  const limit=Math.min(13,speed*1.15);
  let length=0,count=0;
  function vertex(x,y,z,age) {
    positions.setXYZ(count,x,y,z);
    alphas.setX(count++,.52*clamp(1-age/1.65,0,1)*clamp(1-length/limit,0,1));
  }
  vertex(last.x,last.y,last.z,now-last.t);
  for(let i=history.length-2;i>=0&&length<limit;i--){
    const a=history[i+1],b=history[i];
    const distance=Math.hypot(a.x-b.x,a.z-b.z);
    if(distance<.001)continue;
    const ratio=Math.min(1,(limit-length)/distance);
    length+=distance*ratio;
    vertex(a.x+(b.x-a.x)*ratio,a.y+(b.y-a.y)*ratio,a.z+(b.z-a.z)*ratio,
      now-(a.t+(b.t-a.t)*ratio));
  }
  positions.needsUpdate=alphas.needsUpdate=true;
  geometry.setDrawRange(0,count);
}

// What a town's road description records beyond lane centres: lane directions, edge paint and
// traffic signals. Written by crosswalk/map_features.py, one file per town.
const endsKey=(a,b)=>`${a[0].toFixed(2)},${a[1].toFixed(2)}|${b[0].toFixed(2)},${b[1].toFixed(2)}`;

// A traffic signal is drawn as a symbol, half again as large as life so its lamps read from the
// follow camera. Its position and facing are the map's. A lamp is lit only from a recorded state.
const LAMPS=['red','amber','green'];
const SIGNAL={scale:1.5, post:3.3, tall:5.2, reach:1.5};
function signalParts(look) {
  const lit=color=>new T.MeshBasicMaterial({color,toneMapped:false});
  const glow=color=>new T.ShaderMaterial({
    uniforms:{uColor:{value:C(color)}},transparent:true,depthWrite:false,
    // A soft disc that always faces the viewer.
    vertexShader:`varying vec2 vUv;void main(){vUv=position.xy*2.;
      vec4 centre=modelViewMatrix*vec4(0.,0.,0.,1.);centre.xy+=position.xy*1.9;gl_Position=projectionMatrix*centre;}`,
    fragmentShader:`uniform vec3 uColor;varying vec2 vUv;void main(){
      float d=length(vUv);gl_FragColor=vec4(uColor,exp(-d*d*4.5)*.62*step(d,1.));${endColor}}`
  });
  return {
    body:paint(look.signalBody,{roughness:.7,metalness:.2}), off:paint(look.signalOff,{roughness:.6}),
    lit:Object.fromEntries(LAMPS.map(name=>[name,lit(look[name])])),
    glow:Object.fromEntries(LAMPS.map(name=>[name,glow(look[name])])),
    pole:new T.CylinderGeometry(.06,.08,1,10), arm:new T.CylinderGeometry(.045,.045,1,8),
    housing:new T.RoundedBoxGeometry(.3,1.2,.42,2,.07), lamp:new T.SphereGeometry(.13,16,10), halo:new T.PlaneGeometry(1,1)
  };
}
function buildSignal(parts, record, height) {
  const group=new T.Group();
  // Local +X points at the traffic the signal governs, which approaches from that side, and
  // local +Z to that traffic's left, where the lanes it governs lie.
  group.position.set(record.post[0],height,record.post[1]);
  group.rotation.y=-(record.governs_yaw+180)*rad;
  const reach=finite(record.reach)?clamp(record.reach,-14,14):SIGNAL.reach;
  // A long mast stands taller, so its head clears a bus.
  const top=Math.abs(reach)>5?SIGNAL.tall:SIGNAL.post;
  add(group,parts.pole,parts.body,0,top/2,0).scale.y=top;
  // An arm carries the head from the roadside post out over the lanes it governs.
  const arm=add(group,parts.arm,parts.body,0,top-.05,reach/2);
  arm.rotation.x=Math.PI/2;arm.scale.y=Math.abs(reach);
  const head=new T.Group();
  head.position.set(0,top-.1,reach);head.scale.setScalar(SIGNAL.scale);
  group.add(head);
  add(head,parts.housing,parts.body,0,0,0);
  const lamps=LAMPS.map((name,i)=>{
    const lamp=add(head,parts.lamp,parts.off,.12,.38-.38*i,0);
    lamp.scale.set(.45,1,1);
    return lamp;
  });
  const halo=add(head,parts.halo,parts.glow.red,.2,0,0);
  halo.visible=false;halo.renderOrder=5;
  // Where the head hangs, in world coordinates: `reach` to the left of the governed traffic.
  const yaw=record.governs_yaw*rad;
  return {id:record.id,x:record.post[0]+reach*Math.sin(yaw),z:record.post[1]-reach*Math.cos(yaw),
    yaw:record.governs_yaw,post:record.post,group,lamps,halo,state:null};
}
function showSignal(parts, signal, state) {
  if(signal.state===state)return;
  signal.state=state;
  signal.lamps.forEach((lamp,i)=>{lamp.material=LAMPS[i]===state?parts.lit[state]:parts.off;});
  signal.halo.visible=Boolean(state);
  if(state){signal.halo.material=parts.glow[state];signal.halo.position.y=.38-.38*LAMPS.indexOf(state);}
}
// The simulator names lamp states Red, Yellow and Green.
const lampState=value=>({red:'red',yellow:'amber',amber:'amber',green:'green'})[String(value).toLowerCase()]||null;

// The camera turns with the vehicle after this lag, so the scene does not swing with every steering input.
const CAMERA_SECONDS=.32;
const ease=(seconds,over)=>1-Math.exp(-Math.max(0,seconds)/over);

export class PerceptionScene {
  constructor(canvas) {
    this.canvas=canvas;
    this.trackBadge=document.createElement('span');
    this.trackBadge.className='scene-track-id';this.trackBadge.hidden=true;
    this.trackBadge.setAttribute('aria-hidden','true');
    canvas.after(this.trackBadge);
    this.badgePosition=V();this.selectedTrackId=null;
    this.renderer=new T.WebGLRenderer({canvas,antialias:true,alpha:false,powerPreference:'low-power'});
    this.renderer.toneMapping=T.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure=1.04;
    this.tones=Object.fromEntries(TONES.map(name=>[name,new T.Color()]));
    this.sky=this.tones.sky;
    this.scene=new T.Scene();
    this.scene.background=this.sky;
    this.scene.fog=new T.Fog(this.sky,48,210);
    this.camera=new T.PerspectiveCamera(47,1,.15,1800);
    this.ego=new T.Group();
    this.scene.add(this.ego);
    const environment=new T.RoomEnvironment(), pmrem=new T.PMREMGenerator(this.renderer);
    this.environment=pmrem.fromScene(environment,.055);
    environment.dispose();pmrem.dispose();
    this.scene.environment=this.environment.texture;
    this.scene.environmentIntensity=.72;
    this.scene.add(new T.HemisphereLight(0xf4f8ff,0x526577,1.32));
    const key=new T.DirectionalLight(0xfff5dd,2.15);
    key.position.set(-20,40,-22);this.scene.add(key);
    const rim=new T.DirectionalLight(0xacd7ed,1.05);rim.position.set(17,14,20);this.scene.add(rim);
    this.floor=add(this.scene,new T.PlaneGeometry(6000,6000),floorMaterial(this.tones));
    this.floor.rotation.x=-Math.PI/2;
    this.floor.position.y=-.10;
    this.ego.visible=false;
    this.shadow=patch(this.ego,5.58,2.75,projectedPatch(this.tones.shadow,true),.029);
    this.returnGroup=new T.Group();this.scene.add(this.returnGroup);
    this.lidar=cloud(false,this.tones.lidar);this.radar=cloud(true,this.tones.radar);this.returnGroup.add(this.lidar,this.radar);
    this.surfaces=cloud(false,this.tones.lidar);this.cornerRadar=cloud(true,this.tones.radar);
    this.returnGroup.add(this.surfaces,this.cornerRadar);
    this.trackGroup=new T.Group();this.scene.add(this.trackGroup);
    this.measuredSignals=new SignalHeads(this.scene);
    // The planner supplies a new short preview each tick. Refill one ribbon so
    // video playback never has to compile a new material or allocate a GPU buffer.
    const strip=Array.from({length:43},(_,i)=>[i,0,0]);
    this.preview=new T.Mesh(surfaceGeometry([strip],1.22),pathMaterial(this.tones));
    this.preview.frustumCulled=false;this.preview.renderOrder=1;this.preview.visible=false;
    this.preview.material.uniforms.uOverview.value=.72;
    this.preview.geometry.attributes.position.setUsage(T.DynamicDrawUsage);
    this.preview.geometry.attributes.uv.setUsage(T.DynamicDrawUsage);
    this.previewSpan=0;this.scene.add(this.preview);
    this.layers={returns:true,tracks:true,path:true};
    this.tracks=new Map();
    // Which objects to draw, and where, is decided by the object memory from the recorded tracks.
    this.memory=new ObjectMemory({
      laneAxis:(x,z,hint,driven)=>this.laneAxis?this.laneAxis(x,z,hint,driven):null,
      laneDistance:(x,z)=>this.laneAxis?this.laneAxis.distance(x,z):0,
      groundAt:(x,z,estimate)=>this.laneAxis?.ground(x,z,estimate)
    });
    this.resources=0;
    this.view='follow';this.yaw=0;this.fromYaw=0;this.toYaw=0;this.cameraYaw=0;
    this.from=V();this.to=V();this.at=V();
    this.target=V();this.offset=V();
    this.orbit=0;this.pitch=0;this.zoom=1;this.dirty=true;
    this.bindings=new AbortController();
    const opts={signal:this.bindings.signal};
    this.setPalette(displayMode());
    addEventListener('ado:display',()=>this.setPalette(displayMode()),opts);
    let drag=null;
    canvas.addEventListener('pointerdown',e=>{
      if(e.button!==0||!e.isPrimary)return;
      drag={x:e.clientX,y:e.clientY,id:e.pointerId};
      canvas.setPointerCapture?.(e.pointerId);
    },opts);
    canvas.addEventListener('pointermove',e=>{
      if(!drag||e.pointerId!==drag.id)return;
      this.orbit-=(e.clientX-drag.x)*.007;
      this.pitch=clamp(this.pitch+(e.clientY-drag.y)*.006,-.24,.74);
      Object.assign(drag,{x:e.clientX,y:e.clientY});this.dirty=true;
    },opts);
    for(const name of ['pointerup','pointercancel','lostpointercapture'])canvas.addEventListener(name,()=>{drag=null;},opts);
    canvas.addEventListener('wheel',e=>{
      e.preventDefault();this.zoom=clamp(this.zoom*Math.exp(clamp(e.deltaY,-120,120)*.0015),.65,2.25);
      this.dirty=true;
    },{...opts,passive:false});
    canvas.addEventListener('dblclick',()=>this.setView(this.view),opts);
    this.resize();
    this.observer=new ResizeObserver(()=>this.resize());this.observer.observe(canvas);
    this.animate = now => {
      if(this.disposed)return;
      this.raf=requestAnimationFrame(this.animate);
      const interpolating=this.duration>0 && now<this.started+this.duration+35;
      const settling=this.view==='follow' && Math.abs(shortest(this.cameraYaw,this.yaw))>.0006;
      if(document.hidden||(!this.dirty&&!interpolating&&!settling))return;
      if(!this.dirty&&now-(this.lastDraw||0)<24)return;
      this.sample(now);
      // The camera turns with the vehicle after a short lag, so the scene does not swing with every steering input.
      const elapsed=clamp((now-(this.lastDraw||now))/1000,0,.1);
      this.cameraYaw=this.snapCamera?this.yaw:this.cameraYaw+shortest(this.cameraYaw,this.yaw)*ease(elapsed,CAMERA_SECONDS);
      this.snapCamera=false;
      this.placeCamera();
      this.positionTrackBadge();
      this.renderer.render(this.scene,this.camera);
      this.lastDraw=now;this.dirty=false;
    };
    this.raf=requestAnimationFrame(this.animate);
  }

  // Recolour the scene for the night or day display. Geometry, tracks and rules stay as they are.
  setPalette(name) {
    if(this.disposed)return;
    const look=PALETTES[name]||PALETTES.night;
    if(look===this.look)return;
    this.look=look;
    for(const key of TONES)tone(this.tones[key],look[key]);
    const custom=name==='night'&&getComputedStyle(this.canvas).getPropertyValue('--perception-sky').trim();
    if(custom)this.sky.set(custom);
    this.scene.fog.color.copy(this.sky);
    this.curbs?.color.copy(this.tones.curb);
    this.renderer.toneMappingExposure=look.exposure;
    this.shadow.material.uniforms.uStrength.value=look.shadowOpacity;
    // Signals and track shapes carry their own materials, so build them again in the new colours.
    // What is remembered does not depend on the colours, so the memory is left alone.
    this.buildSignals();
    this.redraw();
  }
  // Draw the current frame again from scratch, after something it is drawn with has changed.
  redraw(){
    const frame=this.frame;
    if(frame){
      this.clearTracks();this.frame=null;
      if(this.replayFrames)this.updateReplay(this.replayFrames,this.replayIndex,this.fraction);
      else this.update(frame);
    }
    this.dirty=true;
  }

  async loadCar(url) {
    if(this.disposed)return;
    if(url===this.carURL&&this.car)return;
    const version=++this.resources;
    if(this.car){release(this.car);this.car=null;this.brakeLamps=null;}
    this.carURL=null;this.dirty=true;
    const gltf=await new T.GLTFLoader().loadAsync(url);
    const car=gltf.scene;
    if(this.disposed||version!==this.resources){release(car);return;}
    const bounds=new T.Box3().setFromObject(car,true);
    if(bounds.isEmpty()){release(car);throw new Error('The vehicle model is empty.');}
    car.position.y-=bounds.min.y;
    // The model names its rear lamp material. Its glow follows the recorded brake input.
    const lamps=new Set();
    car.traverse(part=>{for(const material of [].concat(part.material||[]))if(/rear lamp/i.test(material.name))lamps.add(material);});
    for(const lamp of lamps)lamp.emissive.setRGB(1,.06,.045);
    this.brakeLamps=[...lamps];
    this.carURL=url;this.car=car;
    this.ego.add(car);this.showBrake(this.frame?.brake);this.dirty=true;
  }
  showBrake(brake) {
    const level=finite(brake)?clamp(brake,0,1):0;
    for(const lamp of this.brakeLamps||[])lamp.emissiveIntensity=.3+3.2*level;
  }

  setMap(town) {
    if(this.disposed)return;
    this.town=town||null;this.features=null;
    this.buildMap();
    // The town's signals and lane paint arrive separately. The scene is complete without them.
    const name=town?.producer?.map, request=this.featureRequest=(this.featureRequest||0)+1;
    if(typeof name!=='string'||!/^[\w-]+$/.test(name))return;
    townFeatures(name).then(data=>{
      if(this.disposed||request!==this.featureRequest||data?.map!==name)return;
      this.features=data;this.buildMap();
    });
  }
  buildMap() {
    const town=this.town, features=this.features;
    if(this.map){release(this.map);this.map=null;this.buildings=null;this.curbs=null;}
    this.signals=[];this.signalParts=null;   // released with the map
    this.laneAxis=null;
    this.elevation=new Map();
    this.mapKey=town?.producer?.map;
    const lanes=(town?.roads||[]).map(l=>(Array.isArray(l)?l:l.points||[]).filter(p=>coord(p,3))).filter(l=>l.length>1);
    if(lanes.length){
      this.map=new T.Group();
      // Match each lane to the road description by its end points.
      const described=new Map((features?.lanes||[]).map(lane=>[endsKey(lane.a,lane.b),lane]));
      const records=lanes.map(lane=>described.get(endsKey(lane[0],lane.at(-1))));
      const paints=features?records:null;
      const streets=new T.Mesh(surfaceGeometry(lanes,1.75,.008,roadIndex(lanes),1.1,paints),streetMaterial(this.tones));
      const sidewalks=ink(this.tones.curb);sidewalks.side=T.DoubleSide;
      this.curbs=sidewalks;
      // A sidewalk band beside every lane; under the street surface it also closes junction corners.
      const curbs=new T.Mesh(surfaceGeometry(lanes,3.75,-.028,null,2.4),sidewalks);
      this.map.add(curbs,streets);
      this.laneAxis=laneAxes(lanes,records.map(record=>record?.forward));
      for(const lane of lanes)for(const p of lane){
        const key=Math.floor(p[0]/8)+','+Math.floor(p[1]/8);
        if(!this.elevation.has(key))this.elevation.set(key,[]);
        this.elevation.get(key).push(p);
      }
      const boxes=(town.buildings||[]).filter(b=>coord(b.center,3)&&coord(b.extent,3)&&b.extent.every(v=>v>0));
      if(boxes.length){
        this.buildings=architecture(boxes,this.tones,this.at);
        this.buildings.material.uniforms.uOverview.value=Number(this.view==='overview');
        this.map.add(this.buildings);
      }
      const crossings=Array.isArray(features?.crosswalks)?features.crosswalks:[];
      if(crossings.length)this.map.add(new T.Mesh(crossingGeometry(crossings),crossingMaterial(this.tones)));
      this.buildSignals();
      this.map.visible=Boolean(this.localized);
      this.scene.add(this.map);
    }
    this.measuredSignals.setMasts(features?.signals,(x,z)=>this.height(x,z));
    this.pathKey=null;
    // Headings follow the mapped lanes, so what is remembered is worked out again with this map.
    this.memory.reset();
    this.replayMemory=null;
    this.redraw();
  }
  // The town's signals, in the current palette.
  buildSignals() {
    for(const signal of this.signals||[])release(signal.group);
    for(const part of Object.values(this.signalParts||{})){
      for(const item of part.isMaterial||part.isBufferGeometry?[part]:Object.values(part))item.dispose();
    }
    this.signals=[];this.signalParts=null;
    const posts=(this.features?.signals||[]).filter(record=>coord(record.post)&&finite(record.governs_yaw));
    if(!this.map||!posts.length)return;
    this.signalParts=signalParts(this.look);
    this.signals=posts.map(record=>buildSignal(this.signalParts,record,this.height(record.post[0],record.post[1])));
    for(const signal of this.signals)this.map.add(signal.group);
  }
  // Legacy logs may explicitly record an observer state for a mapped pole. Without a
  // per-fixture observation its head stays unlit, including during scripted signal events.
  lightSignals(frame,ctx) {
    this.measuredSignals.update(ctx.located?frame:null);
    if(Array.isArray(frame.signals)||frame.perception_version==='surround-v1'){
      // A surveyed pole can carry multiple real heads. Their measured positions replace the
      // single schematic mast used by recordings with no fixture observations.
      for(const signal of this.signals||[])signal.group.visible=false;
      return;
    }
    if(!this.signals?.length)return;
    const states=new Map(), listed=frame.observer_map?.signals;
    if(Array.isArray(listed)){
      for(const entry of listed){
        let match=entry?.id==null?null:this.signals.find(signal=>String(signal.id)===String(entry.id));
        if(!match&&finite(entry?.x)&&finite(entry?.y)){
          let distance=6;
          for(const signal of this.signals){
            const d=Math.hypot(signal.post[0]-entry.x,signal.post[1]-entry.y);
            if(d<distance){distance=d;match=signal;}
          }
        }
        if(match)states.set(match,lampState(entry.state));
      }
    }
    const cos=Math.cos(ctx.degrees*rad), sin=Math.sin(ctx.degrees*rad), follow=this.view==='follow';
    for(const signal of this.signals){
      showSignal(this.signalParts,signal,states.get(signal)??null);
      // A signal the vehicle is passing shrinks away, so it does not loom in front of the follow camera.
      const ahead=cos*(signal.x-ctx.x)+sin*(signal.z-ctx.z);
      const size=follow&&ctx.located?clamp((ahead-2)/3,0,1):1;
      signal.group.visible=size>0;signal.group.scale.setScalar(Math.max(size,.001));
    }
  }

  height(x,z){
    if(!this.elevation)return 0;
    let distance=Infinity,h=0;
    const ix=Math.floor(x/8),iz=Math.floor(z/8);
    for(let i=-1;i<=1;i++)for(let j=-1;j<=1;j++)for(const p of this.elevation.get((ix+i)+','+(iz+j))||[]){
      const d=(p[0]-x)**2+(p[1]-z)**2;
      if(d<distance){distance=d;h=p[2];}
    }
    return h;
  }

  setPath(points) {
    const path=Array.isArray(points)?points.filter(p=>coord(p)):[];
    const key=JSON.stringify(path);
    if(this.pathKey===key)return;
    this.pathKey=key;
    if(this.path){release(this.path);this.path=null;}
    this.stations=[];
    const waypoints=[];
    for(const [x,z,h] of path){
      const p=V(x,finite(h)?h:this.height(x,z),z);
      if(!waypoints.length||p.distanceToSquared(waypoints.at(-1))>.0001)waypoints.push(p);
    }
    if(waypoints.length<2)return;
    const curve=new T.CatmullRomCurve3(waypoints);
    const samples=curve.getPoints(clamp(Math.ceil(curve.getLength()/1.4),waypoints.length-1,3500));
    let s=0;
    samples.forEach((p,i)=>{
      if(i)s+=Math.hypot(p.x-samples[i-1].x,p.z-samples[i-1].z);
      this.stations.push([p.x,p.z,s]);
    });
    const lines=samples.map(p=>[p.x,p.z,p.y]);
    this.path=new T.Mesh(surfaceGeometry([lines],1.22,.051),pathMaterial(this.tones));
    this.path.renderOrder=1;this.path.visible=this.layers.path;
    this.scene.add(this.path);
    this.path.material.uniforms.uOverview.value=Number(this.view==='overview');
  }

  setPreview(points) {
    this.previewSpan=0;
    if(Array.isArray(points)&&points.length>1&&points.every(p=>coord(p))){
      const waypoints=[];
      for(const [x,y] of points){
        const p=V(x,this.height(x,y),y);
        if(!waypoints.length||p.distanceToSquared(waypoints.at(-1))>.0001)waypoints.push(p);
      }
      if(waypoints.length>1){
        const {position,uv}=this.preview.geometry.attributes;
        const samples=new T.CatmullRomCurve3(waypoints).getPoints(position.count/2-1);
        for(let i=0;i<samples.length;i++){
          const p=samples[i],a=samples[Math.max(0,i-1)],b=samples[Math.min(samples.length-1,i+1)];
          const dx=b.x-a.x,dz=b.z-a.z,length=Math.hypot(dx,dz)||1;
          this.previewSpan+=Math.hypot(p.x-a.x,p.z-a.z);
          for(const [side,sign] of [[0,-1],[1,1]]){
            position.setXYZ(i*2+side,p.x-sign*1.22*dz/length,p.y+.075,p.z+sign*1.22*dx/length);
            uv.setXY(i*2+side,sign,this.previewSpan);
          }
        }
        position.needsUpdate=true;uv.needsUpdate=true;
      }
    }
    this.showPaths();
  }

  showPaths() {
    this.preview.visible=this.layers.path&&this.previewSpan>0;
    if(this.path){
      this.path.visible=this.layers.path&&(this.view==='overview'||!this.previewSpan);
      this.path.material.uniforms.uPreviewSpan.value=this.previewSpan;
    }
  }

  station(x,z) {
    let dist=Infinity,station=0;
    for(let i=1;i<(this.stations?.length||0);i++){
      const [ax,az,sa]=this.stations[i-1],[bx,bz,sb]=this.stations[i];
      const dx=bx-ax,dz=bz-az,ds=sb-sa;
      const u=clamp(((x-ax)*dx+(z-az)*dz)/(ds*ds||1),0,1);
      const d=(ax+dx*u-x)**2+(az+dz*u-z)**2;
      if(d<dist){dist=d;station=sa+u*ds;}
    }
    return station;
  }

  dropTrack(id) {
    const track=this.tracks.get(id);
    if(!track)return;
    track.group.removeFromParent();release(track.foot);release(track.trail);
    for(const material of track.materials)material.dispose();   // the shape's geometry is shared
    this.tracks.delete(id);
  }
  clearTracks(){
    for(const id of this.tracks.keys())this.dropTrack(id);
  }
  setSelectedTrack(id){
    this.selectedTrackId=id;
    this.dirty=true;
  }
  setRiderAssociations(riders){ this.riders=riders; }
  sourceGroups(frame){
    if(!frame||frame.frame!==this.frame?.frame||frame.run_id!==this.frame?.run_id)return [];
    return [...this.tracks.values()].filter(t=>t.sourceIds?.length>1)
      .map(t=>({key:t.id,primary:t.sourceId,ids:t.sourceIds,label:t.label}));
  }

  positionTrackBadge(){
    let selected;
    for(const track of this.tracks.values()){
      const on=this.selectedTrackId!=null&&(track.sourceId===this.selectedTrackId||track.sourceIds?.includes(this.selectedTrackId));
      track.foot.material.uniforms.uSelected.value=on?1:0;
      if(on)selected=track;
    }
    this.trackBadge.hidden=true;
    if(!selected||!this.layers.tracks||!this.frame)return;
    this.badgePosition.copy(selected.group.position);
    this.badgePosition.y+=selected.group.scale.y+.18;
    this.camera.updateMatrixWorld();
    this.badgePosition.project(this.camera);
    const {x,y,z}=this.badgePosition;
    if(z< -1||z>1||Math.abs(x)>1||Math.abs(y)>1)return;
    this.trackBadge.style.left=`${(x+1)*50}%`;
    this.trackBadge.style.top=`${(1-y)*50}%`;
    this.trackBadge.textContent='#'+this.selectedTrackId;
    this.trackBadge.hidden=false;
  }
  makeTrack(id,label) {
    const group=new T.Group(), shape=buildRoadUser(label,this.look);
    group.add(shape.group);
    const color=this.look[TRACK_TONE[label]];
    const foot=patch(group,1.30,1.55,projectedPatch(color),.034);
    foot.renderOrder=1;
    const trail=makeTrail(color);
    trail.renderOrder=1;
    this.trackGroup.add(group,trail);
    const track={id,label,group,foot,trail,materials:shape.materials,history:[],from:V(),to:V(),fromAngle:0,toAngle:0,
      size:V(),fromSize:V(),toSize:V(),opacity:-1,fromOpacity:0,toOpacity:0};
    this.tracks.set(id,track);return track;
  }

  updateReplay(frames,index,fraction) {
    const frame=frames[index];
    if(!frame)return;
    if(this.frame===frame && this.replayFrames===frames && this.replayMemory) {
      if(fraction!==this.fraction){this.fraction=fraction;this.dirty=true;}
      return;
    }
    const discontinuity=!this.frame || this.replayFrames!==frames || this.replayIndex+1!==index;
    this.replayFrames=frames;this.replayIndex=index;this.fraction=fraction;
    this.replayMemory??=new ReplayMemory(this.memory.deps);
    const heightAt=(x,z)=>this.height(x,z);
    const current=this.replayMemory.at(frames,index,heightAt);
    const next=index+1<frames.length?this.replayMemory.at(frames,index+1,heightAt):null;
    this.update(frame,{current,next:next&&!next.ctx.reset?next:current,discontinuity});
  }

  update(frame,replay=null) {
    if(this.disposed)return;
    const now=performance.now();
    if(!replay){this.replayFrames=null;this.replayMemory=null;this.fraction=null;}
    if(!frame){
      this.frame=null;this.ego.visible=false;this.returnGroup.visible=false;this.localized=false;
      this.measuredSignals.update(null);
      if(this.map)this.map.visible=false;
      this.setPath(null);this.setPreview(null);this.clearTracks();this.duration=0;this.memory.ctx=null;
      this.dirty=true;return;
    }
    const previous=this.frame;
    if(!replay && previous && frame.frame!=null && frame.frame===previous.frame && frame.run_id===previous.run_id &&
       frame.sim_time===previous.sim_time && frame.scenario===previous.scenario && frame.seed===previous.seed)return;
    this.sample(now);
    if(!previous)this.memory.ctx=null;
    const {ctx,objects}=replay?.current??this.memory.step(frame,(px,pz)=>this.height(px,pz),now/1000);
    const {t,dt,x,z,degrees,h,located}=ctx;
    const reset=ctx.reset||Boolean(replay?.discontinuity);
    this.currentCtx=ctx;
    const m=frame.observer_map;
    if(reset)this.clearTracks();
    this.frame=frame;this.time=t;this.localized=located;
    this.ego.visible=true;this.returnGroup.visible=this.layers.returns;
    if(this.map)this.map.visible=located;
    this.from.copy(this.at);this.fromYaw=this.yaw;
    this.to.set(x,h,z);this.toYaw=-degrees*rad;
    if(reset){this.from.copy(this.to);this.fromYaw=this.toYaw;this.snapCamera=true;}
    this.started=now;this.duration=reset||replay?0:clamp(dt*1000,25,160);
    if(replay){
      this.from.set(x,h,z);this.fromYaw=-degrees*rad;
      const next=replay.next.ctx;
      this.to.set(next.x,next.h,next.z);this.toYaw=-next.degrees*rad;
    }
    this.setPath(located?m.path:null);
    this.setPreview(located?frame.plan_xy:null);
    this.stationFrom=reset?this.station(x,z):(this.path?.material.uniforms.uStation.value||0);
    this.stationTo=this.station(x,z);
    if(replay){this.stationFrom=this.stationTo;this.stationTo=this.station(this.to.x,this.to.z);}
    // Range samples remain fixed at their acquisition pose while the vehicle moves between frames.
    this.returnGroup.position.set(x,h,z);this.returnGroup.rotation.y=-degrees*rad;
    this.showBrake(frame.brake);
    if(Array.isArray(frame.lidar_xyz)){
      fillReturns(this.lidar,frame.lidar_xyz,null);fillReturns(this.surfaces,[],null);
    }else{
      fillReturns(this.lidar,frame.lidar,.085);
      fillReturns(this.surfaces,(frame.tracks||[]).flatMap(track=>track.surface_xyz||[]),null);
    }
    fillReturns(this.radar,frame.radar,.105);fillReturns(this.cornerRadar,cornerReturns(frame),null);
    this.lightSignals(frame,ctx);
    const drawn=new Set();
    const nextObjects=new Map(replay?.next.objects.map(o=>[o.key,o]));
    const nativeTracks=new Map((frame.tracks??[]).map(track=>[track.id,track]));
    for(const object of objects){
      const inferred=object.label==='person'?this.riders?.get(object.sourceId)?.label:null;
      const label=inferred??(Object.hasOwn(TRACK_TONE,object.label)?object.label:'unknown');
      drawn.add(object.key);
      let track=this.tracks.get(object.key), carried=null;
      // The memory settled on another class for it: swap the symbol where it stands.
      if(track&&track.label!==label){carried=track.opacity;this.dropTrack(object.key);track=null;}
      const fresh=!track;
      track=track||this.makeTrack(object.key,label);
      track.sourceId=object.sourceId;
      track.sourceIds=object.sourceIds;
      track.from.copy(track.group.position);track.to.set(object.x,object.y,object.z);
      track.fromAngle=track.group.rotation.y;track.toAngle=-object.heading*rad;
      // The symbol stays within sizes its class can have; the ground outline keeps the estimate itself.
      track.fromSize.copy(track.group.scale);
      track.size.set(object.length,object.height,object.width);
      plausibleSize(label,track.size,track.toSize);
      track.foot.scale.set(track.size.x/track.toSize.x,track.size.z/track.toSize.z,1);
      track.fromOpacity=track.opacity;track.toOpacity=object.opacity;
      if(fresh||object.turned)track.fromAngle=track.toAngle;
      if(fresh){
        track.from.copy(track.to);track.fromSize.copy(track.toSize);track.group.scale.copy(track.toSize);
        track.fromOpacity=carried??(reset?object.opacity:0);
      }
      if(replay){
        const following=nextObjects.get(object.key);
        track.from.copy(track.to);track.fromAngle=track.toAngle;track.fromSize.copy(track.toSize);
        track.fromOpacity=object.opacity;
        if(following&&following.label===object.label){
          track.to.set(following.x,following.y,following.z);
          if(!following.turned)track.toAngle=-following.heading*rad;
          plausibleSize(label,track.size.set(following.length,following.height,following.width),track.toSize);
          track.toOpacity=following.opacity;
        }
      }
      // A remembered object keeps a fainter outline than one measured in this frame.
      track.foot.material.uniforms.uFresh.value=object.held?.18:1;
      if(frame.perception_version==='surround-v1'&&object.held&&TRACK_TONE[label]==='vehicle'){
        const sources=(object.sourceIds??[object.sourceId]).map(id=>nativeTracks.get(id)).filter(Boolean);
        const age=sources.length?Math.min(...sources.map(source=>source.age_s??0)):1;
        track.lastSeen=clamp((age-.35)/.55,0,1);
      }else track.lastSeen=0;
      track.foot.material.uniforms.uLastSeen.value=track.lastSeen;
      const history=track.history;
      if(!object.moving)history.length=0;
      while(history.length&&(t-history[0].t>1.65||history.length>=30))history.shift();
      const last=history.at(-1);
      if(object.moving&&(!last||Math.hypot(last.x-object.x,last.z-object.z)>.09)){
        history.push({t,x:object.x,y:this.height(object.x,object.z)+.065,z:object.z});
      }
      reviseTrail(track.trail,history,t,object.moving?object.speed:0);
    }
    for(const key of this.tracks.keys())if(!drawn.has(key))this.dropTrack(key);
    this.dirty=true;
  }

  sample(now){
    if(!this.frame)return;
    const t=this.fraction??(this.duration?clamp((now-this.started)/this.duration,0,1):1);
    this.at.lerpVectors(this.from,this.to,t);
    this.yaw=this.fromYaw+shortest(this.fromYaw,this.toYaw)*t;
    this.ego.position.copy(this.at);this.ego.rotation.y=this.yaw;
    this.floor.material.uniforms.uEgo.value.copy(this.at);
    if(this.path)this.path.material.uniforms.uStation.value=this.stationFrom+(this.stationTo-this.stationFrom)*t;
    for(const tr of this.tracks.values()){
      tr.group.position.lerpVectors(tr.from,tr.to,t);
      tr.group.scale.lerpVectors(tr.fromSize,tr.toSize,t);
      tr.group.rotation.y=tr.fromAngle+shortest(tr.fromAngle,tr.toAngle)*t;
      const opacity=tr.fromOpacity+(tr.toOpacity-tr.fromOpacity)*t;
      if(opacity!==tr.opacity||tr.lastSeen!==tr.previousLastSeen){
        tr.opacity=opacity;
        tr.previousLastSeen=tr.lastSeen;
        for(const material of tr.materials)material.opacity=opacity*(1-.86*tr.lastSeen);
        tr.foot.material.uniforms.uAlpha.value=opacity;
        tr.trail.material.uniforms.uAlpha.value=opacity;
      }
    }
  }
  placeCamera(){
    const overview=this.view==='overview';
    // Follow looks straight down the road from behind and above the vehicle, which sits low in the
    // picture so most of it shows what lies ahead.
    const vertical=overview?1.13:clamp(.36+this.pitch,.19,1.24);
    const bearing=(overview?Math.PI*.5:Math.PI)+this.orbit;
    const portrait=this.camera.aspect<.9;
    const distance=(overview?74:portrait?27:22)*this.zoom;
    const pivot=overview?0:portrait?7.2:7.0;
    this.target.set(pivot,.6,0);
    this.offset.set(Math.cos(bearing)*Math.cos(vertical)*distance,Math.sin(vertical)*distance,
      Math.sin(bearing)*Math.cos(vertical)*distance);
    if(!overview){
      this.target.applyAxisAngle(PerceptionScene.up,this.yaw);
      this.offset.applyAxisAngle(PerceptionScene.up,this.cameraYaw);
    }
    this.target.add(this.at);
    this.camera.position.copy(this.target).add(this.offset);
    this.camera.lookAt(this.target);
  }
  setView(view){
    if(view!=='follow'&&view!=='overview')return;
    this.view=view;this.orbit=this.pitch=0;this.zoom=1;this.snapCamera=true;
    this.showPaths();
    if(this.path)this.path.material.uniforms.uOverview.value=Number(view==='overview');
    if(this.buildings)this.buildings.material.uniforms.uOverview.value=Number(view==='overview');
    if(this.frame&&this.currentCtx)this.lightSignals(this.frame,this.currentCtx);
    this.dirty=true;
  }
  setLayer(layer,enabled){
    if(!Object.hasOwn(this.layers,layer))return;
    this.layers[layer]=Boolean(enabled);
    const group={returns:this.returnGroup,tracks:this.trackGroup,path:this.path}[layer];
    if(group)group.visible=this.layers[layer];
    if(layer==='path')this.showPaths();
    this.dirty=true;
  }
  resize(){
    if(this.disposed)return;
    const w=this.canvas.clientWidth||this.canvas.parentElement?.clientWidth||640;
    const h=this.canvas.clientHeight||this.canvas.parentElement?.clientHeight||400;
    const ratio=Math.min(window.devicePixelRatio||1,w<650?1.4:1.6,Math.sqrt(2200000/(w*h)));
    this.renderer.setPixelRatio(ratio);
    this.renderer.setSize(w,h,false);
    for(const points of [this.lidar,this.radar,this.surfaces,this.cornerRadar])points.material.uniforms.uPixel.value=ratio;
    const style=getComputedStyle(this.canvas);
    const offsetX=Number(style.getPropertyValue('--perception-offset-x'))||0;
    const offsetY=Number(style.getPropertyValue('--perception-offset-y'))||0;
    this.camera.aspect=w/Math.max(1,h);
    if(offsetX||offsetY)this.camera.setViewOffset(w,h,w*offsetX,h*offsetY,w,h);
    else this.camera.clearViewOffset();
    this.dirty=true;
  }
  dispose(){
    if(this.disposed)return;
    this.disposed=true;this.resources++;
    cancelAnimationFrame(this.raf);this.bindings.abort();this.observer?.disconnect();
    this.clearTracks();
    this.measuredSignals.dispose();
    this.trackBadge.remove();
    release(this.scene);disposeRoadUserGeometry();this.environment.dispose();this.renderer.dispose();
  }
}
PerceptionScene.up=V(0,1,0);
