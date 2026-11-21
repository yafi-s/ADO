// Offline front-view evaluation. The display emits objects before the actor references are read.
import fs from 'node:fs';
import {createHash} from 'node:crypto';
import {parseArgs} from 'node:util';
import {audit,match} from './audit_city_perception.mjs';
import {bearingObservations,frontBearingCue} from '../web/lamp-observations.mjs';

const {positionals:[directory,output],values}=parseArgs({allowPositionals:true,options:{
  paired:{type:'string'},reference:{type:'string'},review:{type:'string'},examples:{type:'string'},
  'memory-revision':{type:'string'}}});
if(!directory||!output)throw Error('Pass capture folder and report file');
const offsets=JSON.parse(fs.readFileSync(new URL('../results/perception/town03-body-offsets.json',import.meta.url)));
const reference=values.reference??'results/perception/town03-static-vehicles.json';
const paired=values.paired?new Map(fs.readFileSync(values.paired,'utf8').trim().split('\n').map(line=>{
  const row=JSON.parse(line);return [row.frame,row.boxes];
})):null;
const rad=Math.PI/180;
const roadUser=label=>['person','car','bus','truck'].includes(label);
const near=p=>Math.hypot(p.x,p.y);
const ahead=p=>p.x>1.3&&Math.abs(p.y)<p.x-1.3&&near(p)<=100;
const band=d=>d<30?'0–30 m':d<50?'30–50 m':'50–100 m';
const area=b=>Math.max(0,b[2]-b[0])*Math.max(0,b[3]-b[1]);
function projected(a,frame) {
  const e=frame.observer_map,theta=(a.yaw-e.ego[2])*rad,c=Math.cos(theta),s=Math.sin(theta);
  const offset=offsets[a.blueprint]?.offset_xyz??[0,0,a.role==='parked-map-vehicle'||a.label==='person'?0:a.dimensions[1]/2];
  const x=a.x+offset[0]*c-offset[1]*s-1.3, y=a.y+offset[0]*s+offset[1]*c;
  const z=a.z+offset[2]-e.ego_z-1.65;
  const [l,h,w]=a.dimensions, pixels=[];
  for(const dx of [-l/2,l/2])for(const dy of [-w/2,w/2])for(const dz of [-h/2,h/2]){
    const depth=x+dx*c-dy*s;
    if(depth<.5)return null;
    pixels.push([.5+.5*(y+dx*s+dy*c)/depth,.5-(640/360)*.5*(z+dz)/depth]);
  }
  const us=pixels.map(p=>p[0]),vs=pixels.map(p=>p[1]);
  const bounds=[Math.max(0,Math.min(...us)),Math.max(0,Math.min(...vs)),
    Math.min(1,Math.max(...us)),Math.min(1,Math.max(...vs))];
  return area(bounds)>0?bounds:null;
}
function overlap(a,b){
  const intersection=area([Math.max(a[0],b[0]),Math.max(a[1],b[1]),Math.min(a[2],b[2]),Math.min(a[3],b[3])]);
  return intersection/(area(a)+area(b)-intersection);
}
const actorsSeen=new Map(), bands={},classBands={}, extras={}, samples=[];
const count=(stage,a,field)=>{
  const r=band(near(a));
  for(const level of [bands[stage]??={},(classBands[stage]??={})[a.label]??={}]){
    (level[r]??={opportunities:0,matched:0})[field]++;
  }
};
const unmatched=(stage,box,frame,range=null)=>{
  (extras[stage]??={total:0,by_class:{}}).total++;
  const kinds=extras[stage].by_class;kinds[box.label]=(kinds[box.label]??0)+1;
  if(values.examples)samples.push({stage,frame:frame.frame,label:box.label,box:box.box??null,range_m:range});
};
const stat=(a)=>actorsSeen.get(a.id);
function hit(stage,a,frame,details){
  const row=stat(a), r=near(a);
  count(stage,a,'matched');
  row.first[stage]??={frame:frame.frame,time_s:frame.sim_time,range_m:+r.toFixed(2),...details};
}
function scoreBoxes(stage,boxes,actors,frame){
  const observations=boxes.filter(b=>roadUser(b.label)),candidates=[];
  for(const a of actors){
    count(stage,a,'opportunities');
    for(const b of observations){
      if(a.label!==b.label)continue;
      const iou=overlap(a.pixels,b.box);
      if(iou>=.25)candidates.push({a,b,iou});
    }
  }
  const used=new Set(),found=new Set();
  for(const {a,b,iou} of candidates.sort((l,r)=>r.iou-l.iou)){
    if(used.has(b)||found.has(a))continue;
    used.add(b);found.add(a);
    hit(stage,a,frame,{iou:+iou.toFixed(3),box:b.box.map(n=>+n.toFixed(4)),source:b.inference_layout,
      ...(b.range_supported!==undefined?{range_supported:b.range_supported}: {})});
  }
  for(const box of observations)if(!used.has(box))unmatched(stage,box,frame);
}
let acquisitions=0;
const result=await audit(directory,{staticFile:reference,memoryRevision:values['memory-revision'],barrierReviewFile:values.review,onFrame({frame,actors,display}){
  const eligible=actors.filter(a=>roadUser(a.label)&&ahead(a)).map(a=>({...a,pixels:projected(a,frame)})).filter(a=>a.pixels);
  for(const a of eligible){
    if(!actorsSeen.has(a.id))actorsSeen.set(a.id,{id:a.id,role:a.role,label:a.label,first:{},first_reference_m:+near(a).toFixed(2),minimum_m:near(a)});
    stat(a).minimum_m=Math.min(stat(a).minimum_m,near(a));
  }
  if(frame.camera_inference_age_s?.front===0){
    acquisitions++;
    scoreBoxes('native_optical',frame.camera_detections?.front??[],eligible,frame);
    if(!values['memory-revision']){
      const cue=frontBearingCue(bearingObservations(frame,null,['front']));
      scoreBoxes('native_front_callout',cue?[cue]:[],eligible,frame);
    }
    if(paired){
      const boxes=paired.get(frame.frame);
      if(!boxes)throw Error('Missing paired front inference '+frame.frame);
      scoreBoxes('same_jpeg_full',boxes.filter(b=>b.inference_layout==='single'),eligible,frame);
      scoreBoxes('same_jpeg_multiscale',boxes,eligible,frame);
    } else {
      scoreBoxes('native_full_only',frame.camera_detections.front.filter(b=>b.inference_layout!=='front-horizon-2x'),eligible,frame);
    }
  }
  const predictions=display.filter(p=>roadUser(p.label)&&p.opacity>=.5);
  for(const a of eligible)count('display',a,'opportunities');
  const found=new Set();
  for(const label of ['person','car','truck','bus']){
    const pairs=match(actors.filter(a=>a.label===label),predictions.filter(p=>p.label===label),3);
    for(const {actor,prediction,distance} of pairs){
      found.add(prediction);
      const a=eligible.find(a=>a.id===actor.id);
      if(a)hit('display',a,frame,{error_m:+distance.toFixed(3),source:prediction.sourceId});
    }
  }
  for(const p of predictions)if(ahead(p)&&!found.has(p))unmatched('display',p,frame,+near(p).toFixed(2));
}});
const sha=path=>createHash('sha256').update(fs.readFileSync(path)).digest('hex');
const report={schema:1,frames:result.frames,front_acquisitions:acquisitions,journal_sha256:result.sha256,
  registered_parked_sha256:sha(reference),memory_sha256_lf:result.memory_sha256_lf,paired_optics_sha256:values.paired?sha(values.paired):null,
  method:'Front camera sector, reference center within 100 m and projected body in image; geometric opportunities include occluded actors. Optical requires exact class and one-to-one projected-box IoU >= 0.25. Unmatched optical boxes have no assigned distance. 3D requires exact class, >=0.5 opacity and a unique center within 3 m. Distances are horizontal ego-to-reference-center range.',
  bands,class_bands:classBands,unmatched:extras,encounters:[...actorsSeen.values()].map(a=>({...a,minimum_m:+a.minimum_m.toFixed(2)}))};
fs.writeFileSync(output,JSON.stringify(report,null,2)+'\n');
if(values.examples)fs.writeFileSync(values.examples,samples.map(x=>JSON.stringify(x)).join('\n')+'\n');
