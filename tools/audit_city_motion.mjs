// Evaluate each 10 Hz pose that the tracker and the rider display emitted.
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {parseArgs} from 'node:util';
import {audit,match} from './audit_city_perception.mjs';
import {plausibleSize} from '../web/road-users.js';

const rad=Math.PI/180, round=x=>Number(x.toFixed(3));
const wrap=x=>((x+540)%360+360)%360-180;
const family=x=>x==='person'?'person':['motorcycle','bicycle','car','truck','bus'].includes(x)?'wheels':x;
const rig=JSON.parse(fs.readFileSync(new URL('../results/surround-rig/crossing-left/frame.json',import.meta.url))).cameras;
const shapes=JSON.parse(fs.readFileSync(new URL('../results/perception/town03-body-offsets.json',import.meta.url)));
const signature=p=>({id:p.id,label:p.label,world:p.world.map(round),heading:round(p.heading),
  dimensions:p.dimensions.map(round),foot:round(p.foot),opacity:round(p.opacity??1),status:p.status,cameras:p.cameras});
const where=(r,p)=>({frame:r.frame,time_s:r.time_s,range_m:r.range_m,...p});

// The observer reports the actor transform and extents, without the bounding-box offset.
// Use a loose projected center and apparent-height gate only to identify plausible RGB candidates.
function imageCandidates(a,f) {
  const views=f.camera_detections??(f.detection_age_s===0?{front:f.detections}:{}), hits=[];
  for(const [name,boxes] of Object.entries(views)) {
    const m=rig[name]; if(!m) continue;
    const h=m.yaw*rad,dx=a.x-m.x,dy=a.y-m.y,depth=dx*Math.cos(h)+dy*Math.sin(h);
    if(depth<1) continue;
    const fx=.5/Math.tan(m.hfov*rad/2),fy=fx*m.width/m.height;
    const u=.5+fx*(-dx*Math.sin(h)+dy*Math.cos(h))/depth;
    const centerZ=a.z+(shapes[a.blueprint]?.offset_xyz[2]??(a.role==='parked-map-vehicle'||a.type==='pedestrian'?0:a.dimensions[1]/2));
    const v=.5-fy*(centerZ-(f.observer_map.ego_z??0)-m.z)/depth, height=fy*a.dimensions[1]/depth;
    if(u<0||u>1||v<0||v>1) continue;
    for(const b of boxes??[]) {
      if(family(a.label)!==family(b.label)||!b.box) continue;
      const [l,t,r,bottom]=b.box,ratio=(bottom-t)/height;
      if(l-.012<=u&&u<=r+.012&&t-.04<=v&&v<=bottom+.04&&ratio>=.35&&ratio<=2.5)
        hits.push({camera:name,label:b.label,score:round(b.score),range_supported:b.range_supported??null,track_id:b.track_id??null});
    }
  }
  return hits;
}
function metric(values) {
  if (!values.length) return null;
  const sorted=values.sort((a,b)=>a[0]-b[0]), at=q=>round(sorted[Math.ceil(q*sorted.length)-1][0]);
  const worst=sorted.at(-1);
  return {n:sorted.length,median:at(.5),p95:at(.95),max:round(worst[0]),worst_frames:worst[1],worst_time_s:worst[2]};
}
function footprint(center,heading,dimensions) {
  const a=heading*rad,c=Math.cos(a),s=Math.sin(a),l=dimensions[0]/2,w=dimensions[2]/2;
  return [[l,w],[-l,w],[-l,-w],[l,-w]].map(([x,y])=>[center[0]+x*c-y*s,center[1]+x*s+y*c]);
}
export function layerStats(rows,key) {
  const steps={translation_m:[],heading_degrees:[],extent_m:[],length_m:[],height_m:[],width_m:[],
    position_error_m:[],absolute_heading_error_deg:[],absolute_extent_error_m:[],footprint_corner_rmse_m:[],
    foot_height_error_m:[],vertical_step_error_m:[],actor_step_m:[]};
  const ids=new Set(),switches=[],duplicateFrames=[],footprintDuplicates=[],runs=[];
  let prev=null,hits=0,physicalTravel=0,symbolTravel=0;
  for(const r of rows) {
    const p=r[key], dup=r[key+'_duplicates'];
    if(dup.length) duplicateFrames.push({frame:r.frame,time_s:r.time_s,matched:p?.id,extra_ids:dup});
    if(p&&r[key+'_footprint_extras']?.length)footprintDuplicates.push({frame:r.frame,time_s:r.time_s,
      matched:p.id,extra_ids:r[key+'_footprint_extras']});
    if(!p) {prev=null;continue;}
    ids.add(p.id);hits++;
    if(!prev||prev.p.id!==p.id||prev.r.frame+1!==r.frame)runs.push({id:p.id,start:r.frame,end:r.frame});
    else runs.at(-1).end=r.frame;
    steps.position_error_m.push([Math.hypot(p.world[0]-r.world[0],p.world[1]-r.world[1]),[r.frame],r.time_s]);
    if(Number.isFinite(r.foot)&&Number.isFinite(p.foot))steps.foot_height_error_m.push([Math.abs(p.foot-r.foot),[r.frame],r.time_s]);
    steps.absolute_heading_error_deg.push([Math.abs(wrap(p.heading-r.yaw)),[r.frame],r.time_s]);
    steps.absolute_extent_error_m.push([Math.max(...p.dimensions.map((x,i)=>Math.abs(x-r.dimensions[i]))),[r.frame],r.time_s]);
    const reference=footprint(r.world,r.yaw,r.dimensions),predicted=footprint(p.world,p.heading,p.dimensions);
    const error=Math.sqrt(Math.min(...[0,2].map(shift=>predicted.reduce((s,[x,y],i)=>{
      const [tx,ty]=reference[(i+shift)%4];return s+(x-tx)**2+(y-ty)**2;
    },0)/4)));
    steps.footprint_corner_rmse_m.push([error,[r.frame],r.time_s]);
    if(prev&&prev.r.frame===r.frame-1) {
      const old=prev.p,orig=prev.r,span=[orig.frame,r.frame];
      if(old.id!==p.id) switches.push({frame:r.frame,time_s:r.time_s,from:old.id,to:p.id});
      const ax=r.world[0]-orig.world[0],ay=r.world[1]-orig.world[1];
      steps.actor_step_m.push([Math.hypot(ax,ay),span,r.time_s]);
      if([p.foot,old.foot,r.foot,orig.foot].every(Number.isFinite))
        steps.vertical_step_error_m.push([Math.abs((p.foot-old.foot)-(r.foot-orig.foot)),span,r.time_s]);
      const same=old.id===p.id;
      if(same){physicalTravel+=Math.hypot(ax,ay);symbolTravel+=Math.hypot(p.world[0]-old.world[0],p.world[1]-old.world[1]);}
      steps.translation_m.push([Math.hypot(p.world[0]-old.world[0]-ax,p.world[1]-old.world[1]-ay),span,r.time_s,same]);
      steps.heading_degrees.push([Math.abs(wrap((p.heading-old.heading)-(r.yaw-orig.yaw))),span,r.time_s,same]);
      const size=p.dimensions.map((v,i)=>Math.abs(v-old.dimensions[i]-(r.dimensions[i]-orig.dimensions[i])));
      steps.extent_m.push([Math.max(...size),span,r.time_s,same]);
      for(const [i,field] of ['length_m','height_m','width_m'].entries())steps[field].push([size[i],span,r.time_s]);
    }
    prev={r,p};
  }
  return {matched:hits,distinct_ids:ids.size,id_changes:switches,runs,duplicate_frames:duplicateFrames.length,
    footprint_duplicate_frames:footprintDuplicates.length,footprint_duplicate_examples:footprintDuplicates.filter((_,i)=>i===0||i===Math.floor(footprintDuplicates.length/2)||i===footprintDuplicates.length-1),
    same_id:{actual_travel_m:round(physicalTravel),predicted_travel_m:round(symbolTravel),
      ...Object.fromEntries(['translation_m','heading_degrees','extent_m'].map(k=>[k,metric(steps[k].filter(v=>v[3]))]))},
    duplicate_examples:duplicateFrames.filter((_,i)=>i===0||i===Math.floor(duplicateFrames.length/2)||i===duplicateFrames.length-1),
    ...Object.fromEntries(Object.entries(steps).map(([k,v])=>[k,metric(v)]))};
}
function describe(rows) {
  function first(condition,details) {
    const r=rows.find(condition);return r?where(r,details?.(r)??{}):null;
  }
  const nearby=rows.filter(r=>r.range_m<=35);
  return {id:rows[0].id,role:rows[0].role,type:rows[0].type,frames:rows.length,within_35m_frames:nearby.length,
    start:where(rows[0]),end:where(rows.at(-1)),
    first_rgb_candidate:first(r=>r.rgb.length,r=>({box:r.rgb[0]})),
    first_sparse_range:first(r=>r.sparse_xy>=2,r=>({points:r.sparse_xy})),
    first_position_only_track:first(r=>r.position_only,r=>({id:r.position_only.id,label:r.position_only.label})),
    first_spatial_track:first(r=>r.tracker,r=>({id:r.tracker.id,status:r.tracker.status})),
    first_camera_range_track:first(r=>r.tracker?.cameras?.length,r=>({id:r.tracker.id,cameras:r.tracker.cameras})),
    first_draw:first(r=>r.display,r=>({id:r.display.id,opacity:r.display.opacity})),
    first_half_opacity:first(r=>r.display?.opacity>=.5,r=>({id:r.display.id,opacity:r.display.opacity})),
    tracker:layerStats(nearby,'tracker'),display:layerStats(nearby,'display')};
}

export async function motion(directory,{memoryRevision,memoryFile,townFile,staticFile,rowFile}={}) {
  const groups=new Map(), latest=new Map();
  const previous={tracker:new Map(),display:new Map()};
  const screen=Object.fromEntries(['tracker','display'].map(layer=>[layer,{distance:[],turn:[],extent:[],large:[]}]));
  const fd=rowFile?fs.openSync(rowFile,'w'):null;
  try {
    const summary=await audit(directory,{memoryRevision,memoryFile,townFile,staticFile,onFrame({frame,actors,tracker,display}) {
      const [ex,ey,yaw]=frame.observer_map.ego,c=Math.cos(yaw*rad),s=Math.sin(yaw*rad);
      const poses={
        tracker:tracker.map(t=>({...t,world:[ex+c*t.x-s*t.y,ey+s*t.x+c*t.y],heading:wrap(yaw+t.heading),
          foot:(frame.observer_map.ego_z??0)+t.z-t.height/2,dimensions:[t.length,t.height,t.width],
          cameras:t.cameras??(!t.coasting&&t.sources?.includes('rgb')?['front']:[])})),
        display:display.map(p=>({...p,dimensions:plausibleSize(p.label,{x:p.dimensions[0],y:p.dimensions[1],z:p.dimensions[2]},
          {set:(...dims)=>dims,copy:v=>[v.x,v.y,v.z]})}))};
      const links={},extras={},footprintExtras={};
      const positionOnly=new Map(match(actors,poses.tracker,3,false).map(p=>[p.actor,{id:p.prediction.id,label:p.prediction.label}]));
      for(const [key,preds] of Object.entries(poses)) {
        const pairs=match(actors,preds),used=new Set(pairs.map(x=>x.prediction));
        links[key]=new Map(pairs.map(x=>[x.actor,x.prediction]));
        const byPrediction=new Map(pairs.map(p=>[p.prediction,p.actor.id])),current=new Map();
        for(const p of preds.filter(p=>family(p.label)==='wheels'&&Math.hypot(p.x,p.y)<=35)) {
          const old=previous[key].get(p.id),actual=byPrediction.get(p)??null;
          current.set(p.id,{p,frame:frame.frame,actual});
          if(!old||old.frame!==frame.frame-1)continue;
          const dist=Math.hypot(p.world[0]-old.p.world[0],p.world[1]-old.p.world[1]),
            turn=Math.abs(wrap(p.heading-old.p.heading)),size=Math.max(...p.dimensions.map((x,i)=>Math.abs(x-old.p.dimensions[i]))),
            span=[old.frame,frame.frame];
          for(const [field,val] of [['distance',dist],['turn',turn],['extent',size]])screen[key][field].push([val,span,frame.sim_time]);
          if(dist>1||turn>15||size>.4)screen[key].large.push({frame:frame.frame,time_s:frame.sim_time,id:p.id,actual,
            prior_actor:old.actual,world_step_m:round(dist),heading_step_deg:round(turn),max_extent_step_m:round(size),
            from:signature(old.p),to:signature(p)});
        }
        previous[key]=current;
        extras[key]=new Map();footprintExtras[key]=new Map();
        for(const p of preds.filter(p=>!used.has(p))) {
          const options=actors.filter(a=>family(a.label)===family(p.label)).map(a=>[Math.hypot(a.x-p.x,a.y-p.y),a]).sort((a,b)=>a[0]-b[0]);
          if(options[0]?.[0]<=3) {const a=options[0][1];if(!extras[key].has(a))extras[key].set(a,[]);extras[key].get(a).push(p.id);}
          const inside=options.find(([,a])=>{
            const h=(a.yaw-yaw)*rad,dx=p.x-a.x,dy=p.y-a.y,offset=shapes[a.blueprint]?.offset_xyz??[0,0];
            return Math.abs(dx*Math.cos(h)+dy*Math.sin(h)-offset[0])<=a.dimensions[0]/2+.5&&
              Math.abs(dy*Math.cos(h)-dx*Math.sin(h)-offset[1])<=a.dimensions[2]/2+.5;
          });
          if(inside) {const a=inside[1];if(!footprintExtras[key].has(a))footprintExtras[key].set(a,[]);footprintExtras[key].get(a).push(p.id);}
        }
      }
      for(const a of actors) {
        const range=Math.hypot(a.x,a.y);
        if(range>60) continue;
        const earlier=latest.get(a.id),segment=earlier&&earlier.last===frame.frame-1?earlier.segment:(earlier?.segment??0)+1;
        latest.set(a.id,{last:frame.frame,segment});
        const key=[a.role,a.id,segment].join('/');
        const offset=shapes[a.blueprint]?.offset_xyz??[0,0,0], angle=a.yaw*rad;
        const world=[a.world[0]+offset[0]*Math.cos(angle)-offset[1]*Math.sin(angle),a.world[1]+offset[0]*Math.sin(angle)+offset[1]*Math.cos(angle)];
        const r={frame:frame.frame,time_s:frame.sim_time,id:a.id,role:a.role,type:a.type,range_m:round(range),world,
          foot:shapes[a.blueprint]||a.role==='parked-map-vehicle'?a.z+offset[2]-a.dimensions[1]/2:null,
          pose_reference:shapes[a.blueprint]?'registered_body_center':a.role==='parked-map-vehicle'?'static_box_center':'actor_origin',
          yaw:a.yaw,dimensions:a.dimensions,sparse_xy:a.points,rgb:imageCandidates(a,frame),position_only:positionOnly.get(a)??null};
        for(const layer of ['tracker','display']) {
          const p=links[layer].get(a);
          r[layer]=p?signature(p):null;r[layer+'_duplicates']=extras[layer].get(a)??[];
          r[layer+'_footprint_extras']=footprintExtras[layer].get(a)??[];
        }
        if(!groups.has(key))groups.set(key,[]);groups.get(key).push(r);
        if(fd!==null)fs.writeSync(fd,JSON.stringify({group:key,...r})+'\n');
      }
    }});
    const all_steps=Object.fromEntries(Object.entries(screen).map(([k,s])=>[k,{world_translation_m:metric(s.distance),
      heading_deg:metric(s.turn),max_extent_m:metric(s.extent),over_1m:s.large.filter(v=>v.world_step_m>1).length,
      over_15deg:s.large.filter(v=>v.heading_step_deg>15).length,over_0_4m_extent:s.large.filter(v=>v.max_extent_step_m>.4).length,
      examples:s.large.sort((a,b)=>Math.max(b.world_step_m,b.heading_step_deg/15,b.max_extent_step_m/.4)-
        Math.max(a.world_step_m,a.heading_step_deg/15,a.max_extent_step_m/.4)).slice(0,10)}]));
    return {frames:summary.frames,input_sha256:summary.sha256,memory_sha256_lf:summary.memory_sha256_lf,
      all_nearby_wheeled_same_id_steps:all_steps,
      groups:Object.fromEntries([...groups].map(([k,rows])=>[k,describe(rows)]))};
  } finally {if(fd!==null)fs.closeSync(fd);}
}
if(path.resolve(process.argv[1]??'')===fileURLToPath(import.meta.url)) {
  const {values,positionals}=parseArgs({allowPositionals:true,options:{output:{type:'string'},rows:{type:'string'},
    town:{type:'string'},'static-map':{type:'string'},'memory-revision':{type:'string'},'memory-file':{type:'string'}}});
  if(positionals.length!==1||!values.output)throw new Error('Pass a recording directory and --output summary.json');
  fs.writeFileSync(values.output,JSON.stringify(await motion(positionals[0],{memoryRevision:values['memory-revision'],memoryFile:values['memory-file'],
    townFile:values.town,staticFile:values['static-map'],rowFile:values.rows}),null,2)+'\n');
}
