import * as T from './vendor/city-view/three-view.js';
import {measuredHeads, cameraLamp, LAMP_COLOURS} from './lamp-observations.mjs';

const maps = new Map();
export function townFeatures(name) {
  if (typeof name !== 'string' || !/^[\w-]+$/.test(name)) return Promise.resolve(null);
  if (!maps.has(name)) {
    const url = new URL(`./media/map-features/${encodeURIComponent(name)}.json`,import.meta.url);
    maps.set(name,fetch(url).then(response=>response.ok?response.json():null).catch(()=>null));
  }
  return maps.get(name);
}

// Camera-facing traffic heads remain legible from both follow and overhead views.
// A separate symbol sits exactly at every surveyed luminous-head centre.
export class SignalHeads {
  constructor(parent) {
    this.group = new T.Group();
    parent.add(this.group);
    this.geometry = new T.PlaneGeometry(1, 1);
    this.heads = new Map();
    this.materials = new Map();
    this.masts = new Map();
    this.posts = new Map();
    this.tube = new T.CylinderGeometry(.065,.065,1,8);
    this.steel = new T.MeshStandardMaterial({color:'#778591',roughness:.82});
    this.up = new T.Vector3(0,1,0);
    this.delta = new T.Vector3();
  }
  setMasts(records,heightAt) {
    this.posts.clear();
    for(const record of records??[]){
      if (!Array.isArray(record.post)||record.post.length<2||!record.post.slice(0,2).every(Number.isFinite))continue;
      const [x,y]=record.post;
      this.posts.set(String(record.id),new T.Vector3(x,heightAt(x,y),y));
    }
  }
  material(state) {
    if (this.materials.has(state)) return this.materials.get(state);
    const material = new T.ShaderMaterial({
      uniforms: {uColor: {value: new T.Color(LAMP_COLOURS[state])},
        uLamp: {value: ['red','yellow','green'].indexOf(state)}},
      transparent: true, depthWrite: false,
      vertexShader: `varying vec2 vUv;
        void main(){vUv=uv;vec4 p=modelViewMatrix*vec4(0.,0.,0.,1.);
          p.xy+=position.xy*vec2(.78,1.56);gl_Position=projectionMatrix*p;}`,
      fragmentShader: `varying vec2 vUv;uniform vec3 uColor;uniform int uLamp;
        void main(){
          vec2 p=(vUv-.5)*vec2(1.,2.);
          vec2 q=abs(p)-vec2(.31,.78);
          float edge=length(max(q,0.))+min(max(q.x,q.y),0.)-.13;
          float aa=max(fwidth(edge),.008);
          vec3 color=mix(vec3(.022,.034,.05),vec3(.20,.24,.28),smoothstep(-.05,-.01,edge));
          for(int i=0;i<3;i++){
            float d=length(p-vec2(0.,.51-.51*float(i)));
            vec3 lamp=i==uLamp?uColor:vec3(.056,.075,.093);
            color=mix(color,lamp,1.-smoothstep(.18-aa,.18+aa,d));
          }
          gl_FragColor=vec4(color,1.-smoothstep(-aa,aa,edge));
          #include <colorspace_fragment>
        }`
    });
    this.materials.set(state, material);
    return material;
  }
  update(frame) {
    const active = new Set();
    const farthest = new Map();
    for (const head of measuredHeads(frame)) {
      active.add(head.id);
      let marker = this.heads.get(head.id);
      const material = this.material(cameraLamp(head));
      if (!marker) {
        marker = new T.Mesh(this.geometry, material);
        this.group.add(marker); this.heads.set(head.id, marker);
      }
      marker.material = material;
      marker.position.set(head.x, head.z, head.y);
      const key=String(head.map_id), base=this.posts.get(key);
      if(base){
        const span=Math.hypot(base.x-head.x,base.z-head.y);
        if(span<=18 && (!farthest.has(key)||span>farthest.get(key).span))farthest.set(key,{head,span,base});
      }
    }
    for (const [id,marker] of this.heads) if (!active.has(id)) {
      marker.removeFromParent(); this.heads.delete(id);
    }
    // Join the surveyed roadside post to its outermost measured head. All sibling
    // heads share this single arm. No mounting geometry is inferred without a post.
    for(const [key,{head,base}] of farthest){
      let mast=this.masts.get(key);
      if(!mast){
        mast={post:new T.Mesh(this.tube,this.steel),arm:new T.Mesh(this.tube,this.steel)};
        this.group.add(mast.post,mast.arm);this.masts.set(key,mast);
      }
      const top=head.z+.56;
      mast.post.position.set(base.x,(base.y+top)/2,base.z);
      mast.post.scale.y=Math.max(.01,top-base.y);
      this.delta.set(head.x-base.x,0,head.y-base.z);
      const length=this.delta.length();
      mast.arm.visible=length>.1;
      if(mast.arm.visible){
        mast.arm.position.set((base.x+head.x)/2,top,(base.z+head.y)/2);
        mast.arm.scale.set(.78,length,.78);
        mast.arm.quaternion.setFromUnitVectors(this.up,this.delta.normalize());
      }
    }
    for(const [key,mast] of this.masts)if(!farthest.has(key)){
      mast.arm.removeFromParent();mast.post.removeFromParent();this.masts.delete(key);
    }
  }
  dispose() {
    this.group.removeFromParent();
    this.heads.clear(); this.geometry.dispose();this.masts.clear();this.tube.dispose();this.steel.dispose();
    for (const material of this.materials.values()) material.dispose();
    this.materials.clear();
  }
}
