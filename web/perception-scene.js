import * as T from './vendor/city-view/three-view.js';

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
function floorMaterial(sky) {
  return new T.ShaderMaterial({
    uniforms:{uSky:{value:sky}, uEgo:{value:V()}}, vertexShader:worldVertex,
    fragmentShader:`
      uniform vec3 uSky,uEgo; varying vec3 vWorld;
      ${haze}
      void main(){
        vec2 g = vWorld.xz / 5.;
        vec2 dd = abs(fract(g + .5) - .5) / max(fwidth(g),vec2(.0001));
        float line = 1.-smoothstep(0.,1.,min(dd.x,dd.y));
        float proximity = 1.-smoothstep(9.,78.,distance(vWorld.xz,uEgo.xz));
        vec3 base = mix(vec3(.013,.024,.035),vec3(.025,.042,.054), proximity);
        base += vec3(.012,.024,.024)*line*proximity;
        gl_FragColor = vec4(aerial(base,vWorld),1.);${endColor}
      }`
  });
}

// Each polyline is a surveyed driving-lane centre. Index the waypoints to locate
// their shared boundaries and separate interior markings from the outside curb.
function roadIndex(roads) {
  const cells = new Map(), key = (x,z) => Math.floor(x/6)+','+Math.floor(z/6);
  roads.forEach((lane,id) => lane.forEach(p => {
    const k=key(p[0],p[1]); if (!cells.has(k)) cells.set(k,[]);
    cells.get(k).push([id,p]);
  }));
  return (x,z,id) => {
    for (let i=-1;i<=1;i++) for (let j=-1;j<=1;j++) {
      for (const [other,p] of cells.get(key(x+i*6,z+j*6)) || []) {
        if (other!==id && (p[0]-x)**2+(p[1]-z)**2<2.7) return true;
      }
    }
    return false;
  };
}
function surfaceGeometry(lines, halfWidth, elevation = 0, edges = null) {
  const xyz=[],uv=[],inside=[],indices=[];
  lines.forEach((line,id) => {
    const offset=xyz.length/3;
    let s=0;
    line.forEach(([x,z,y],i) => {
      const a=line[Math.max(0,i-1)], b=line[Math.min(line.length-1,i+1)];
      const dx=b[0]-a[0], dz=b[1]-a[1], len=Math.hypot(dx,dz)||1;
      const nx=-dz/len, nz=dx/len;
      if(i) s+=Math.hypot(x-a[0],z-a[1]);
      for(const sign of [-1,1]) {
        xyz.push(x+sign*halfWidth*nx, y+elevation,z+sign*halfWidth*nz);
        uv.push(sign,s);
        inside.push(edges?.(x+sign*4*nx,z+sign*4*nz,id)?1:0);
      }
      if(i) {const k=offset+i*2;indices.push(k-2,k-1,k,k,k-1,k+1);}
    });
  });
  const geo=new T.BufferGeometry();
  geo.setAttribute('position',new T.Float32BufferAttribute(xyz,3));
  geo.setAttribute('uv',new T.Float32BufferAttribute(uv,2));
  geo.setAttribute('inner',new T.Float32BufferAttribute(inside,1));
  geo.setIndex(indices);
  return geo;
}
const ribbonVertex = `
attribute float inner;
varying vec2 vLane; varying vec3 vWorld; varying float vInner;
void main(){
  vLane=uv; vInner=inner;
  vec4 p=modelMatrix*vec4(position,1.); vWorld=p.xyz;
  gl_Position=projectionMatrix*viewMatrix*p;
}`;
function streetMaterial(sky) {
  return new T.ShaderMaterial({
    uniforms:{uSky:{value:sky}},side:T.DoubleSide,vertexShader:ribbonVertex,
    fragmentShader:`uniform vec3 uSky; varying vec2 vLane; varying vec3 vWorld; varying float vInner;
      ${haze}
      void main(){
        float a=abs(vLane.x), aa=min(.12,fwidth(a)*.7+.001);
        float lane=smoothstep(.967-aa,.967+aa,a);
        float d=mod(vLane.y,7.);
        float dash=smoothstep(.05,.25,d)*(1.-smoothstep(2.65,2.85,d));
        float marking=mix(.52,dash,step(.58,vInner))*lane;
        vec3 base=mix(vec3(.041,.058,.077),vec3(.052,.073,.092),1.-a);
        vec3 color=mix(base,vec3(.32,.43,.49),marking*.85);
        gl_FragColor=vec4(aerial(color,vWorld),1.);${endColor}
      }`
  });
}
function pathMaterial(sky) {
  return new T.ShaderMaterial({
    uniforms:{uSky:{value:sky},uStation:{value:0},uOverview:{value:0}},
    transparent:true,side:T.DoubleSide,depthWrite:false,vertexShader:ribbonVertex,
    fragmentShader:`uniform vec3 uSky; uniform float uStation,uOverview;
      varying vec2 vLane; varying vec3 vWorld;
      ${haze}
      void main() {
        float along=vLane.y-uStation, a=abs(vLane.x);
        float width=mix(.97,.58,smoothstep(2.,72.,along));
        float aa=clamp(fwidth(a),.005,.13);
        float mask=1.-smoothstep(width-.115-aa,width+aa,a);
        float edge=smoothstep(width-.18-aa,width-.035+aa,a);
        float arm=abs(mod(vLane.y+.68*a+4.75,9.5)-4.75);
        float chevron=(1.-smoothstep(.07,.12+fwidth(vLane.y)*.6,arm))*(1.-smoothstep(.34,.45,a));
        float nearEgo=exp(-max(0.,along)/33.);
        float ahead=smoothstep(-4.,-.4,along)*(1.-smoothstep(65.,108.,along));
        float energy=mix(ahead*(.18+.82*nearEgo),.32+.68*exp(-abs(along)/42.),uOverview);
        float opacity=energy*mask*(.52+.29*edge+.17*chevron);
        vec3 color=mix(vec3(.028,.38,.42),vec3(.24,.84,.76),.17*nearEgo+max(edge,chevron)*.81);
        gl_FragColor=vec4(aerial(color,vWorld),opacity);${endColor}
      }`
  });
}
function architecture(boxes, sky, egoPosition) {
  const material=new T.ShaderMaterial({
    uniforms:{uSky:{value:sky},uEgo:{value:egoPosition},uOverview:{value:0}},
    transparent:true,
    vertexShader:`varying vec3 vWorld,vLocal,vNormal;
      void main(){ vLocal=position;vec4 p=instanceMatrix*vec4(position,1.);
        vNormal=normalize(mat3(instanceMatrix)*normal);
        vWorld=p.xyz; gl_Position=projectionMatrix*viewMatrix*p; }`,
    fragmentShader:`uniform vec3 uSky,uEgo; uniform float uOverview; varying vec3 vWorld,vLocal,vNormal;
      ${haze}
      void main(){
        vec3 ax=abs(vLocal);
        float sideEdge=min(.5-ax.x,.5-ax.z);
        float sideDist=max(.5-ax.x,.5-ax.z);
        float e=vNormal.y>.6?sideEdge:min(.5-ax.y,sideDist);
        float edge=1.-smoothstep(.001, .007+fwidth(e),e);
        float light=clamp(.4+dot(vNormal,normalize(vec3(-.4,1.,-.25)))*.5,0.,1.);
        vec3 c=mix(vec3(.023,.037,.052),vec3(.066,.092,.112),light)+edge*vec3(.014,.029,.037);
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
    uniforms:{uColor:{value:C(color)},uPulse:{value:0},uFresh:{value:1}},
    transparent:true, depthWrite:false,side:T.DoubleSide,
    vertexShader:`varying vec2 vUv;void main(){vUv=uv*2.-1.;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}`,
    fragmentShader:`uniform vec3 uColor;uniform float uPulse,uFresh;varying vec2 vUv;
      void main() {
        float d=length(vUv);
        ${solid ? 'float opacity=(1.-smoothstep(.38,1.,d))*.57;' : `
          float border=abs(length(max(abs(vUv)-vec2(.50,.34),0.))-.22);
          float footprint=(1.-smoothstep(.013,.036+fwidth(border),border))*.54;
          float ring=(1.-smoothstep(.015,.045,abs(d-(.65+.32*uPulse))))*(1.-uPulse)*step(.00001,uPulse);
          float opacity=footprint*(.35+.65*uFresh)+.24*ring;`}
        gl_FragColor=vec4(uColor,opacity);${endColor}
      }`
  });
}
function patch(parent,length,width,mat,height) {
  const p=add(parent,new T.PlaneGeometry(length,width),mat,0,height,0);
  p.rotation.x=-Math.PI/2;
  return p;
}
function shapedCabin() {
  // Cross-sections taper the nose, rake the windscreen and round the roof shoulders.
  const positions=[], indices=[];
  for (const [x,bottom,top,span] of [[-.46,.39,.47,.36],[-.30,.41,.77,.405],[-.19,.42,.91,.39],
      [.10,.42,.92,.39],[.31,.42,.63,.406],[.43,.40,.48,.34]]) {
    for (const [z,y] of [[-span,bottom],[-span*.94,top-.09],[-span*.74,top],[span*.74,top],
        [span*.94,top-.09],[span,bottom]]) positions.push(x,y,z);
  }
  for(let i=0;i<5;i++)for(let j=0;j<6;j++) {
    const a=i*6+j,b=i*6+(j+1)%6;
    indices.push(a,b,a+6,b,b+6,a+6);
  }
  for(const off of [0,30])for(let j=1;j<5;j++)indices.push(off,off+j,off+j+1);
  const geo=new T.BufferGeometry();geo.setAttribute('position',new T.Float32BufferAttribute(positions,3));geo.setIndex(indices);
  geo.computeVertexNormals();
  return geo;
}
const colors = {
  car:0x9caefa, bus:0x9caefa, truck:0x9caefa, motorcycle:0xe3bc83,
  bicycle:0xe3bc83, person:0xff9d9b, unknown:0x9cafbf
};
function segment(parent,start,end,radius,mat) {
  const a=V(...start),b=V(...end),delta=b.clone().sub(a);
  const m=add(parent,new T.CylinderGeometry(radius,radius*.85,delta.length(),8),mat);
  m.position.copy(a).add(b).multiplyScalar(.5);
  m.quaternion.setFromUnitVectors(V(0,1,0),delta.normalize());
}
function form(label) {
  const root=new T.Group(),color=colors[label]||colors.unknown;
  const body=paint(color,{roughness:.4,emissive:color,emissiveIntensity:.14});
  const glass=paint(0x485a8c,{metalness:.32,roughness:.31,side:T.DoubleSide});
  const light=ink(0xcdd6ff), dark=paint(0x28364c);
  const rounded=(w,h,d,r,mat,x,y,z)=>add(root,new T.RoundedBoxGeometry(w,h,d,2,r),mat,x,y,z);
  if(label==='person') {
    // Symbolic pedestrian, mapped to the estimated track envelope.
    const head=add(root,new T.SphereGeometry(.17,16,12),body,.03,.885,0);
    head.scale.set(1.16,.68,1.13);
    rounded(.51,.34,.72,.12,body,0,.58,0);
    for(const sign of [-1,1]){
      segment(root,[0,.46,sign*.155],[sign*.065,.08,sign*.20],.105,body);
      segment(root,[0,.698,sign*.31],[-.022,.427,sign*.43],.069,body);
    }
  } else if(label==='bicycle'||label==='motorcycle') {
    for(const sign of [-1,1]){
      const wheel=add(root,new T.CylinderGeometry(.15,.15,.11,20),dark,sign*.34,.17,0);
      wheel.rotation.x=Math.PI/2;
      segment(root,[sign*.34,.19,0],[0,.43,0],.045,body);
    }
    rounded(.48,.22,.46,.07,body,0,.39,0);
    segment(root,[0,.44,0],[-.055,.73,0],.115,body);
    add(root,new T.SphereGeometry(.124,12,10),body,.008,.86,0);
    segment(root,[0,.68,0],[.27,.53,0],.055,body);
  } else if(label==='car'||label==='truck'||label==='bus') {
    rounded(.99,.36,.98,.08,body,0,.34,0);
    if(label==='car') {
      add(root,shapedCabin(),glass);
      rounded(.32,.057,.61,.027,body,-.045,.918,0);
    } else if(label==='truck') {
      rounded(.65,.63,.92,.045,body,-.165,.665,0);
      rounded(.30,.40,.90,.055,glass,.31,.575,0);
    } else {
      rounded(.96,.63,.94,.053,body,0,.655,0);
      rounded(.92,.31,.95,.045,glass,.006,.71,0);
    }
    for(const x of [-.302,.305])for(const z of [-.467,.467]){
      const wheel=add(root,new T.CylinderGeometry(.115,.115,.081,14),dark,x,.185,z);
      wheel.rotation.x=Math.PI/2;
      wheel.scale.x=.89;
      wheel.scale.z=1.38;
    }
    for(const sign of [-1,1]){
      rounded(.018,.045,.17,.008,light,.492,.414,sign*.305);
      rounded(.018,.045,.15,.007,ink(0xc0a7c6),-.491,.412,sign*.32);
    }
  } else rounded(.97,.97,.97,.075,body,0,.5,0);
  return root;
}

function cloud(radar) {
  const geometry=new T.BufferGeometry();
  const material=new T.ShaderMaterial({
    transparent:true,depthWrite:false,
    uniforms:{uPixel:{value:1},uColor:{value:C(radar?0xe6b979:0x8cced2)}},
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
  for(const p of values)if(coord(p))attr.setXYZ(n++,p[0],height,p[1]);
  points.geometry.setDrawRange(0,n);attr.needsUpdate=true;
}

function makeTrail(color) {
  const geo=new T.BufferGeometry(),positions=new T.BufferAttribute(new Float32Array(30*3),3),alphas=new T.BufferAttribute(new Float32Array(30),1);
  positions.setUsage(T.DynamicDrawUsage);alphas.setUsage(T.DynamicDrawUsage);
  geo.setAttribute('position',positions);geo.setAttribute('opacity',alphas);geo.setDrawRange(0,0);
  const material=new T.ShaderMaterial({
    uniforms:{uColor:{value:C(color)}},transparent:true,depthWrite:false,
    vertexShader:`attribute float opacity;varying float vOpacity;void main(){vOpacity=opacity;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}`,
    fragmentShader:`uniform vec3 uColor;varying float vOpacity;void main(){gl_FragColor=vec4(uColor,vOpacity);${endColor}}`
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

function styleConfidence(track, confidence, coasting) {
  const c=finite(confidence)?clamp(confidence,0,1):0;
  const x=clamp((c-.16)/.66,0,1);
  const opacity=coasting ? .14+.29*c : .23+.77*x*x*(3-2*x);
  for(const material of track.materials){
    const transparent=opacity<.995;
    if(material.transparent!==transparent){
      material.transparent=transparent;
      material.depthWrite=!transparent;
      material.needsUpdate=true;
    }
    material.opacity=opacity;
  }
  track.foot.material.uniforms.uFresh.value=coasting ? .18 : c;
}

export class PerceptionScene {
  constructor(canvas) {
    this.canvas=canvas;
    this.renderer=new T.WebGLRenderer({canvas,antialias:true,alpha:false,powerPreference:'low-power'});
    this.renderer.toneMapping=T.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure=1.04;
    const custom=getComputedStyle(canvas).getPropertyValue('--perception-sky').trim();
    this.sky=C(custom||'#192633');
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
    this.floor=add(this.scene,new T.PlaneGeometry(6000,6000),floorMaterial(this.sky));
    this.floor.rotation.x=-Math.PI/2;
    this.floor.position.y=-.10;
    this.ego.visible=false;
    this.shadow=patch(this.ego,5.58,2.75,projectedPatch(0x061012,true),.029);
    this.returnGroup=new T.Group();this.ego.add(this.returnGroup);
    this.lidar=cloud(false);this.radar=cloud(true);this.returnGroup.add(this.lidar,this.radar);
    this.trackGroup=new T.Group();this.scene.add(this.trackGroup);
    this.layers={returns:true,tracks:true,path:true};
    this.tracks=new Map();this.templates=new Map();
    this.resources=0;
    this.view='follow';this.yaw=0;this.fromYaw=0;this.toYaw=0;
    this.from=V();this.to=V();this.at=V();
    this.target=V();this.offset=V();
    this.orbit=0;this.pitch=0;this.zoom=1;this.dirty=true;
    this.bindings=new AbortController();
    const opts={signal:this.bindings.signal};
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
      const pulsing=this.layers.tracks && now<this.lastPulse+680;
      if(document.hidden||(!this.dirty&&!interpolating&&!pulsing))return;
      if(!this.dirty&&now-(this.lastDraw||0)<24)return;
      this.sample(now);this.placeCamera();
      this.renderer.render(this.scene,this.camera);
      this.lastDraw=now;this.dirty=false;
    };
    this.raf=requestAnimationFrame(this.animate);
  }

  async loadCar(url) {
    if(this.disposed)return;
    if(url===this.carURL&&this.car)return;
    const version=++this.resources;
    if(this.car){release(this.car);this.car=null;}
    this.carURL=null;this.dirty=true;
    const gltf=await new T.GLTFLoader().loadAsync(url);
    const car=gltf.scene;
    if(this.disposed||version!==this.resources){release(car);return;}
    const bounds=new T.Box3().setFromObject(car,true);
    if(bounds.isEmpty()){release(car);throw new Error('The vehicle model is empty.');}
    car.position.y-=bounds.min.y;
    this.carURL=url;this.car=car;
    this.ego.add(car);this.dirty=true;
  }

  setMap(town) {
    if(this.disposed)return;
    if(this.map){release(this.map);this.map=null;this.buildings=null;}
    this.elevation=new Map();
    this.mapKey=town?.producer?.map;
    const lanes=(town?.roads||[]).map(l=>(Array.isArray(l)?l:l.points||[]).filter(p=>coord(p,3))).filter(l=>l.length>1);
    if(lanes.length){
      this.map=new T.Group();
      const streets=new T.Mesh(surfaceGeometry(lanes,2.02,.008,roadIndex(lanes)),streetMaterial(this.sky));
      const sidewalks=ink(0x3d4d58);sidewalks.side=T.DoubleSide;
      const curbs=new T.Mesh(surfaceGeometry(lanes,2.65,-.028),sidewalks);
      this.map.add(curbs,streets);
      for(const lane of lanes)for(const p of lane){
        const key=Math.floor(p[0]/8)+','+Math.floor(p[1]/8);
        if(!this.elevation.has(key))this.elevation.set(key,[]);
        this.elevation.get(key).push(p);
      }
      const boxes=(town.buildings||[]).filter(b=>coord(b.center,3)&&coord(b.extent,3)&&b.extent.every(v=>v>0));
      if(boxes.length){
        this.buildings=architecture(boxes,this.sky,this.at);
        this.map.add(this.buildings);
      }
      this.map.visible=Boolean(this.localized);
      this.scene.add(this.map);
    }
    this.pathKey=null;
    if(this.frame)this.setPath(this.localized?this.frame.observer_map.path:null);
    this.dirty=true;
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
    this.path=new T.Mesh(surfaceGeometry([lines],1.22,.051),pathMaterial(this.sky));
    this.path.renderOrder=1;this.path.visible=this.layers.path;
    this.scene.add(this.path);
    this.path.material.uniforms.uOverview.value=Number(this.view==='overview');
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
    for(const material of track.materials)material.dispose();
    this.tracks.delete(id);
  }
  clearTracks(){
    for(const id of this.tracks.keys())this.dropTrack(id);
  }
  makeTrack(id,label) {
    if(!this.templates.has(label))this.templates.set(label,form(label));
    const group=new T.Group(), body=this.templates.get(label).clone(true);
    // Each track owns its opacity; geometry stays shared between matching classes.
    const copies=new Map();
    body.traverse(part=>{
      if(!part.isMesh)return;
      const base=part.material;
      if(!copies.has(base))copies.set(base,base.clone());
      part.material=copies.get(base);
    });
    group.add(body);
    const color=colors[label]||colors.unknown;
    const foot=patch(group,1.30,1.55,projectedPatch(color),.034);
    const trail=makeTrail(color);
    this.trackGroup.add(group,trail);
    const track={id,label,group,foot,trail,materials:[...copies.values()],history:[],from:V(),to:V(),fromAngle:0,toAngle:0,pulse:-Infinity};
    this.tracks.set(id,track);return track;
  }

  update(frame) {
    if(this.disposed)return;
    const now=performance.now();
    if(!frame){
      this.frame=null;this.ego.visible=false;this.localized=false;
      if(this.map)this.map.visible=false;
      this.setPath(null);this.clearTracks();this.duration=0;
      this.dirty=true;return;
    }
    const previous=this.frame;
    if(previous && frame.frame!=null && frame.frame===previous.frame && frame.run_id===previous.run_id &&
       frame.sim_time===previous.sim_time && frame.scenario===previous.scenario)return;
    this.sample(now);
    const t=finite(frame.sim_time)?frame.sim_time:finite(frame.video_time)?frame.video_time:now/1000;
    const dt=t-this.time;
    const m=frame.observer_map, located=coord(m?.ego,3);
    const [x,z,degrees]=located?m.ego:[0,0,0];
    const h=located?(finite(m.ego_z)?m.ego_z:this.height(x,z)):0;
    const reset=!previous||frame.run_id!==previous.run_id||frame.scenario!==previous.scenario||
      previous.seed!==frame.seed||located!==this.localized||dt<=0||dt>.8||
      (finite(previous.frame)&&finite(frame.frame)&&frame.frame<=previous.frame)||
      (located&&Math.hypot(this.to.x-x,this.to.z-z)>9);
    if(reset)this.clearTracks();
    this.frame=frame;this.time=t;this.localized=located;
    this.ego.visible=true;
    if(this.map)this.map.visible=located;
    this.from.copy(this.at);this.fromYaw=this.yaw;
    this.to.set(x,h,z);this.toYaw=-degrees*rad;
    if(reset){this.from.copy(this.to);this.fromYaw=this.toYaw;}
    this.started=now;this.duration=reset?0:clamp(dt*1000,25,160);
    this.setPath(located?m.path:null);
    this.stationFrom=reset?this.station(x,z):(this.path?.material.uniforms.uStation.value||0);
    this.stationTo=this.station(x,z);
    fillReturns(this.lidar,frame.lidar,.085);fillReturns(this.radar,frame.radar,.105);
    const seen=new Set(),cos=Math.cos(degrees*rad),sin=Math.sin(degrees*rad);
    for(const sample of Array.isArray(frame.tracks)?frame.tracks:[]){
      if(sample?.id==null||![sample.x,sample.y,sample.z,sample.length,sample.width,sample.height].every(finite)||
         sample.length<=0||sample.width<=0||sample.height<=0)continue;
      const label=Object.hasOwn(colors,sample.label)?sample.label:'unknown';
      const id=sample.id;
      if(seen.has(id))continue;
      const coast=sample.coasting===true;
      const sources=Array.isArray(sample.sources)?sample.sources:[];
      const ranged=sources.includes('lidar')||sources.includes('radar');
      // A 2D class score cannot establish depth. A coast can continue a
      // previously range-supported track; a seek clears that provenance.
      if(!ranged&&(!coast||sources.length||this.tracks.get(id)?.label!==label))continue;
      seen.add(id);
      if(this.tracks.get(id)?.label!==label)this.dropTrack(id);
      const fresh=!this.tracks.has(id),track=this.tracks.get(id)||this.makeTrack(id,label);
      const tx=x+cos*sample.x-sin*sample.y,tz=z+sin*sample.x+cos*sample.y;
      // Track Z is the estimated centre. Anchor the shape at its measured base.
      const base=h+sample.z-sample.height/2;
      track.from.copy(track.group.position);track.to.set(tx,base,tz);
      track.fromAngle=track.group.rotation.y;
      track.toAngle=-(degrees+(finite(sample.heading)?sample.heading:0))*rad;
      if(fresh){track.from.copy(track.to);track.fromAngle=track.toAngle;}
      track.group.scale.set(sample.length,sample.height,sample.width);
      styleConfidence(track,sample.confidence,coast);
      if(!coast&&(fresh||track.coast)&&now-track.pulse>680){track.pulse=now;this.lastPulse=now;}
      track.coast=coast;
      const history=track.history;
      const moving=located&&finite(sample.speed_mps)&&sample.speed_mps>(label==='person' ? .30 : .55);
      if(!moving)history.length=0;
      while(history.length&&(t-history[0].t>1.65||history.length>=30))history.shift();
      const last=history.at(-1);
      if(moving&&!coast&&(!last||Math.hypot(last.x-tx,last.z-tz)>.09)){
        history.push({t,x:tx,y:this.height(tx,tz)+.065,z:tz});
      }
      reviseTrail(track.trail,history,t,moving?sample.speed_mps:0);
    }
    for(const id of this.tracks.keys())if(!seen.has(id))this.dropTrack(id);
    this.dirty=true;
  }

  sample(now){
    if(!this.frame)return;
    const t=this.duration?clamp((now-this.started)/this.duration,0,1):1;
    this.at.lerpVectors(this.from,this.to,t);
    this.yaw=this.fromYaw+shortest(this.fromYaw,this.toYaw)*t;
    this.ego.position.copy(this.at);this.ego.rotation.y=this.yaw;
    this.floor.material.uniforms.uEgo.value.copy(this.at);
    if(this.path)this.path.material.uniforms.uStation.value=this.stationFrom+(this.stationTo-this.stationFrom)*t;
    for(const tr of this.tracks.values()){
      tr.group.position.lerpVectors(tr.from,tr.to,t);
      tr.group.rotation.y=tr.fromAngle+shortest(tr.fromAngle,tr.toAngle)*t;
      const age=(now-tr.pulse)/640;
      tr.foot.material.uniforms.uPulse.value=age>0&&age<1?age:0;
    }
  }
  placeCamera(){
    const overview=this.view==='overview';
    const vertical=overview?1.13:clamp(.46+this.pitch,.19,1.24);
    const bearing=(overview?Math.PI*.5:Math.PI+.32)+this.orbit;
    const portrait=this.camera.aspect<.9;
    const distance=(overview?74:portrait?18.2:14.7)*this.zoom;
    const pivot=overview?0:portrait?3.2:4.15;
    this.target.set(pivot,.6,0);
    this.offset.set(Math.cos(bearing)*Math.cos(vertical)*distance,Math.sin(vertical)*distance,
      Math.sin(bearing)*Math.cos(vertical)*distance);
    if(!overview){
      this.target.applyAxisAngle(PerceptionScene.up,this.yaw);
      this.offset.applyAxisAngle(PerceptionScene.up,this.yaw);
    }
    this.target.add(this.at);
    this.camera.position.copy(this.target).add(this.offset);
    this.camera.lookAt(this.target);
  }
  setView(view){
    if(view!=='follow'&&view!=='overview')return;
    this.view=view;this.orbit=this.pitch=0;this.zoom=1;
    if(this.path)this.path.material.uniforms.uOverview.value=Number(view==='overview');
    if(this.buildings)this.buildings.material.uniforms.uOverview.value=Number(view==='overview');
    this.dirty=true;
  }
  setLayer(layer,enabled){
    if(!Object.hasOwn(this.layers,layer))return;
    this.layers[layer]=Boolean(enabled);
    const group={returns:this.returnGroup,tracks:this.trackGroup,path:this.path}[layer];
    if(group)group.visible=this.layers[layer];
    this.dirty=true;
  }
  resize(){
    if(this.disposed)return;
    const w=this.canvas.clientWidth||this.canvas.parentElement?.clientWidth||640;
    const h=this.canvas.clientHeight||this.canvas.parentElement?.clientHeight||400;
    const ratio=Math.min(window.devicePixelRatio||1,w<650?1.4:1.6,Math.sqrt(2200000/(w*h)));
    this.renderer.setPixelRatio(ratio);
    this.renderer.setSize(w,h,false);
    this.lidar.material.uniforms.uPixel.value=this.radar.material.uniforms.uPixel.value=ratio;
    this.camera.aspect=w/Math.max(1,h);this.camera.updateProjectionMatrix();this.dirty=true;
  }
  dispose(){
    if(this.disposed)return;
    this.disposed=true;this.resources++;
    cancelAnimationFrame(this.raf);this.bindings.abort();this.observer?.disconnect();
    this.clearTracks();
    for(const template of this.templates.values())release(template);
    this.templates.clear();
    release(this.scene);this.environment.dispose();this.renderer.dispose();
  }
}
PerceptionScene.up=V(0,1,0);
