// Compare displayed work-zone and slow-traffic objects with the native bodies after rendering.
import fs from 'node:fs';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {createInterface} from 'node:readline';
import {advance, contextOf, createState, laneAxes} from '../web/object-memory.mjs';
import {barrierReviewer} from '../web/barrier-review.mjs';
import {coneReviewer} from '../web/cone-observations.mjs';
import {cameraBoxes} from '../web/lamp-observations.mjs';
import {maneuverActivity} from '../web/maneuver-status.mjs';

const [stage, output, ...watch] = process.argv.slice(2);
if (!stage || !output) throw Error('Provide a sealed native capture directory, output JSON and optional frame:camera witnesses.');
const read = file => JSON.parse(fs.readFileSync(file));
const manifest = read(path.join(stage, 'capture.json'));
function sealed(name) {
  const data = fs.readFileSync(path.join(stage, 'raw', name));
  if (createHash('sha256').update(data).digest('hex') !== manifest.raw_sha256[name]) {
    throw Error('Capture identity mismatch for ' + name);
  }
  return JSON.parse(data);
}
const town = sealed('town.json');
const scores = sealed('telemetry.json');
const cameras = ['chase', 'front', ...manifest.perimeter_views];
const features = read(new URL(`../web/media/map-features/${town.producer.map}.json`, import.meta.url));
const endpoints = (a,b) => [a,b].map(p => p.slice(0,2).map(n=>n.toFixed(2)).join(',')).join('|');
const lanes = (town.roads ?? []).map(l => (Array.isArray(l) ? l : l.points ?? [])
  .filter(p => Array.isArray(p) && p.length >= 3 && p.slice(0,3).every(Number.isFinite)))
  .filter(points => points.length > 1);
const directions = new Map((features.lanes ?? []).map(l => [endpoints(l.a,l.b),l.forward]));
const axis = laneAxes(lanes,lanes.map(l => directions.get(endpoints(l[0],l.at(-1)))));
const state = createState(), deps = {laneAxis:axis, laneDistance:axis.distance, groundAt:axis.ground};
const qualifyBarrier = barrierReviewer(), qualifyCone = coneReviewer();
const roles = ['in-lane-obstruction', 'slow-traffic', 'adjacent-front', 'adjacent-rear'];
const classes = {cone:'traffic_cone', barrier:'barrier', car:'car', van:'car', vehicle:'car', bus:'bus', truck:'truck'};
const reports = Object.fromEntries(roles.map(role => [role, {
  nearby_actor_frames:0, matched_drawings:0, fresh_camera_and_range:0, camera_geometry_confirmed:0, actors:{}
}]));
const events = Object.fromEntries(manifest.summary.events.map(e=>[e.kind,e]));
const highlights = {};
for (const [role, mark, view] of [
  ['in-lane-obstruction','exposed_s','front'],
  ['in-lane-obstruction','alternate_alongside_s','chase'],
  ['in-lane-obstruction','returned_to_route_s','rear_right'],
  ['slow-traffic','occupied_gap_wait_s','rear_left'],
  ['slow-traffic','moving_alongside_s','chase'],
  ['slow-traffic','returned_to_route_s','rear_right'],
]) {
  const time = events[role]?.[mark];
  if (!Number.isFinite(time)) continue;
  const score = scores.find(row => row.events?.find(e => e.kind === role)?.[mark] === time);
  if (!score) throw Error('Missing event exposure for ' + mark);
  highlights[`${role}/${mark}`] = {frame:score.frame, view};
}
const waitFrame = manifest.maneuver_sequence?.first_actual_wait_frame;
if (Number.isInteger(waitFrame)) highlights['slow-traffic/first_actual_wait_frame'] = {frame:waitFrame, view:'rear_left'};
for (const witness of watch) {
  const [value,view,...extra] = witness.split(':');
  const frame = Number(value);
  if (extra.length || !Number.isInteger(frame) || frame<manifest.first_frame || frame>manifest.last_frame || !cameras.includes(view)) {
    throw Error('Invalid frame:camera witness ' + witness);
  }
  highlights['witness/'+witness] = {frame,view};
}

// Same rigid mounts and normalized intrinsics as crosswalk/camera_geometry.py.
const mounts = {
  front:[1.30,0,1.65,0], left:[1.30,-.75,1.65,-70], right:[1.30,.75,1.65,70],
  rear_left:[-1.10,-.70,1.72,-145], rear_right:[-1.10,.70,1.72,145],
};
function rotation(pitch, yaw, roll) {
  [pitch,yaw,roll] = [pitch,yaw,roll].map(d=>d*Math.PI/180);
  const cp=Math.cos(pitch),sp=Math.sin(pitch),cy=Math.cos(yaw),sy=Math.sin(yaw),cr=Math.cos(roll),sr=Math.sin(roll);
  return [
    [cp*cy, cy*sp*sr-sy*cr, -cy*sp*cr-sy*sr],
    [cp*sy, sy*sp*sr+cy*cr, -sy*sp*cr+cy*sr],
    [sp,-cp*sr,cp*cr],
  ];
}
const dot = (a,b) => a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const apply = (basis, point) => basis.map(row => dot(row,point));
const inverse = (basis, point) => point.map((_,i) => basis[0][i]*point[0]+basis[1][i]*point[1]+basis[2][i]*point[2]);
const sum = (a,b) => a.map((x,i) => x+b[i]);
const diff = (a,b) => a.map((x,i) => x-b[i]);
function body(a) {
  const basis = rotation(a.pitch ?? 0, a.yaw, a.roll ?? 0);
  const origin = [a.x,a.y,a.z];
  const center = sum(origin,apply(basis,a.bbox_offset_m ?? [0,0,0]));
  const [length,height,width] = a.dimensions_m;
  const local = rotation(...(a.bbox_rotation_deg ?? [0,0,0]));
  const corners = [];
  for (const x of [-length/2,length/2]) for (const y of [-width/2,width/2]) for (const z of [-height/2,height/2]) {
    corners.push(sum(center,apply(basis,apply(local,[x,y,z]))));
  }
  return {actor:a,xy:classes[a.type] === 'traffic_cone' || a.type === 'barrier' ? origin : center,corners};
}
function project(corners, pose, view) {
  const [mx,my,mz,yaw] = mounts[view];
  const ground = rotation(pose.pitch,pose.yaw,pose.roll);
  const optical = rotation(0,yaw,0);
  const points = corners.map(point => inverse(optical,diff(inverse(ground,diff(point,[pose.x,pose.y,pose.z])),[mx,my,mz])));
  // Clip OBB edges against the optical near plane before projecting.
  const near = .05;
  const clipped = points.filter(([d]) => d >= near);
  for (let i=0; i<8; i++) for (const bit of [1,2,4]) {
    const a = points[i], b = points[i|bit];
    if (i === (i|bit) || (a[0]>=near) === (b[0]>=near)) continue;
    const t = (near-a[0])/(b[0]-a[0]);
    clipped.push([near,a[1]+t*(b[1]-a[1]),a[2]+t*(b[2]-a[2])]);
  }
  if (!clipped.length) return null;
  const u = clipped.map(([d,x]) => .5+.5*x/d);
  const v = clipped.map(([d,x,y]) => .5-(8/9)*y/d);
  const box = [Math.max(0,Math.min(...u)),Math.max(0,Math.min(...v)),
    Math.min(1,Math.max(...u)),Math.min(1,Math.max(...v))];
  return box[2] > box[0] && box[3] > box[1] ? box : null;
}
function overlap(a,b) {
  const area = c => (c[2]-c[0])*(c[3]-c[1]);
  const shared = Math.max(0,Math.min(a[2],b[2])-Math.max(a[0],b[0])) *
    Math.max(0,Math.min(a[3],b[3])-Math.max(a[1],b[1]));
  return shared/(area(a)+area(b)-shared);
}
const round = n => Number(n.toFixed(3));

const fingerprint=createHash('sha256');
const input=fs.createReadStream(path.join(stage,'raw/perception.jsonl'));
input.on('data',chunk=>fingerprint.update(chunk));
const publicationWitnesses={};
const alongsideFrame=highlights['in-lane-obstruction/alternate_alongside_s']?.frame;
let ctx=null, count=0, priorOccupied=0;
for await (const line of createInterface({input,crlfDelay:Infinity})) {
  const raw=JSON.parse(line);
  const score = scores[count];
  if (raw.frame !== manifest.first_frame+count || score?.frame !== raw.frame ||
      score.camera_frame !== raw.frame || score.video_index !== count++ || score.sim_time !== raw.sim_time) {
    throw Error('Native replay and camera timing disagree');
  }
  const replay = {...raw,...score}; // crosswalk/showcase.py::write_replay
  const {ego,ego_z,others}=replay.observer_map;
  // Reference bodies never enter the drawing pipeline.
  const inlet=qualifyBarrier({...replay,observer_map:{ego,ego_z}});
  const frame=qualifyCone(inlet,inlet.cone_observations);
  const occupied=score.events?.find(e=>e.kind==='slow-traffic')?.occupied_gap_frames ?? 0;
  const yielding=occupied>priorOccupied && frame.tactical_pass?.maneuver_phase==='yield' && !frame.tactical_pass.committed;
  priorOccupied=occupied;
  ctx=contextOf(frame,ctx,()=>0,score.video_time);
  const drawings=advance(state,ctx,deps).filter(o=>o.opacity>=.5);
  const atFrame = Object.values(highlights).filter(h=>h.frame===raw.frame);
  for (const h of atFrame) Object.assign(h,{seconds:score.video_time,planner:maneuverActivity(frame.tactical_pass),
    objects:[],unmatched:[]});
  if (!others.some(a=>roles.includes(a.role))) continue;
  const tracks=new Map([...(frame.tracks??[]),...(frame.cone_tracks??[])].map(t=>[t.id,t]));
  const bodies = others.filter(a=>classes[a.type]).map(body);
  const distance = r => Math.hypot(r.xy[0]-ego[0],r.xy[1]-ego[1]);
  const boxes=new Map(), projections=new Map();
  const cameraViews=cameras.filter(view=>Object.hasOwn(mounts,view));
  function image(view) {
    if (!boxes.has(view)) boxes.set(view,cameraBoxes(frame,view).filter(box=>box.range_supported===true));
    return boxes.get(view);
  }
  function projected(view) {
    if (!projections.has(view)) projections.set(view,bodies.map(({actor,corners}) =>
      [actor.actor_id,project(corners,score.ego_pose,view)]).filter(([,bounds])=>bounds));
    return projections.get(view);
  }
  const viewsOf = actor => cameraViews.filter(view=>projected(view).some(([id])=>id===actor.actor_id));
  function linked(actor,object) {
    const sources=object.sourceIds??[object.sourceId];
    return sources.flatMap(id => {
      const track=tracks.get(id);
      if (!track || track.coasting || !(track.lidar_hits>0)) return [];
      return (track.cameras??[]).filter(view=>cameraViews.includes(view) &&
        (track.label==='traffic_cone' || frame.camera_inference_age_s?.[view]===0))
        .flatMap(view=>image(view).filter(item=>item.track_id===id).map(item => {
          const overlay = item.box.map(n=>Math.min(1,Math.max(0,n)));
          const fits = projected(view);
          const nativeBox = fits.find(([actorId])=>actorId===actor.actor_id)?.[1] ?? null;
          const fit = nativeBox ? overlap(overlay,nativeBox) : 0;
          let runnerUp = 0, competitor = null;
          for (const [actorId,bounds] of fits) {
            if (actorId === actor.actor_id) continue;
            const iou = overlap(overlay,bounds);
            if (iou > runnerUp) {
              runnerUp = iou;
              competitor = actorId;
            }
          }
          return {view,track_id:id,image_id:item.image_id??null,class:item.label,box:item.box,
            lidar_hits:track.lidar_hits,native_box:nativeBox,body_iou:round(fit),
            other_actor_id:competitor,other_body_iou:round(runnerUp),
            corroborated:fit>=.15 && (runnerUp<.15 || fit-runnerUp>=.05)};
        }));
    });
  }
  const pairs=[];
  for (const reference of bodies) {
    const {actor,xy} = reference;
    const range=distance(reference);
    const result = reports[actor.role];
    if (result) {
      const a = result.actors[actor.actor_id] ??= {class:classes[actor.type],reference_frames:0,nearby_frames:0,
        matched_frames:0,confirmed_frames:0,first_by_camera:{}};
      a.reference_frames++;
      if (!a.closest || range<a.closest.range_m) a.closest={frame:raw.frame,seconds:score.video_time,range_m:range};
      if (range<=35) {
        result.nearby_actor_frames++;
        a.nearby_frames++;
      }
    }
    if (range>35) continue;
    for (const object of drawings) {
      if (object.label!==classes[actor.type]) continue;
      const error=Math.hypot(xy[0]-object.x,xy[1]-object.z);
      const gate=['barrier','cone'].includes(actor.type) ? .65 : 3;
      if (error<=gate) pairs.push({actor,object,error,range});
    }
  }
  const assigned=new Set(), used=new Set();
  for (const {actor,object,error,range} of pairs.sort((a,b)=>a.error-b.error)) {
    if (assigned.has(actor.actor_id)||used.has(object.key)) continue;
    assigned.add(actor.actor_id); used.add(object.key);
    const result=reports[actor.role];
    if (!result) continue;
    result.matched_drawings++;
    const attribution=linked(actor,object);
    const fresh=attribution.length>0;
    const confirmed=attribution.some(item=>item.corroborated);
    if (fresh) result.fresh_camera_and_range++;
    if (confirmed) result.camera_geometry_confirmed++;
    const evidence={frame:raw.frame,seconds:score.video_time,actor_id:actor.actor_id,role:actor.role,
      source_id:object.sourceId,symbol_id:object.key,class:object.label,center_error_m:round(error),
      range_m:round(range),opacity:round(object.opacity),held:object.held,fresh_camera_and_range:fresh,
      cameras:[...new Set(attribution.filter(item=>item.corroborated).map(item=>item.view))],attribution};
    const a=result.actors[actor.actor_id];
    a.matched_frames++;
    a.first ??= evidence;
    if (fresh && !a.first_fresh) a.first_fresh=evidence;
    if (confirmed) a.confirmed_frames++;
    for (const view of evidence.cameras) a.first_by_camera[view] ??= evidence;
    const drum=attribution.find(item=>item.view==='right' && item.corroborated);
    const priorDrum=publicationWitnesses.obstruction_right;
    if (actor.role==='in-lane-obstruction' && object.label==='traffic_cone' && drum && Number.isInteger(alongsideFrame) &&
        (!priorDrum || Math.abs(evidence.frame-alongsideFrame)<Math.abs(priorDrum.frame-alongsideFrame))) {
      publicationWitnesses.obstruction_right={...evidence,view:drum.view,track_id:drum.track_id};
    }
    if (actor.role==='adjacent-rear' && yielding && !publicationWitnesses.yielding_to_adjacent_rear) {
      const supported=attribution.find(item=>item.view==='rear_left' && item.corroborated) ??
        attribution.find(item=>item.corroborated);
      if (supported) publicationWitnesses.yielding_to_adjacent_rear={...evidence,view:supported.view,track_id:supported.track_id};
    }
    for (const h of atFrame) h.objects.push({...evidence,in_camera_geometry:viewsOf(actor),
      selected_camera_linked:evidence.cameras.includes(h.view)});
  }
  for (const h of atFrame) {
    h.unmatched = bodies.filter(({actor})=>roles.includes(actor.role) && !assigned.has(actor.actor_id))
      .map(reference=>({actor_id:reference.actor.actor_id,role:reference.actor.role,class:classes[reference.actor.type],
        range_m:round(distance(reference)),
        in_camera_geometry:viewsOf(reference.actor)}));
  }
}
const hash=fingerprint.digest('hex');
if (count!==scores.length || count!==manifest.summary.frames || manifest.first_frame+count-1!==manifest.last_frame ||
    hash!==manifest.raw_sha256['perception.jsonl'] || Object.values(highlights).some(h=>!h.objects)) {
  throw Error('Capture identity, timing or frame count mismatch');
}
fs.writeFileSync(output,JSON.stringify({trial:manifest.trial_id,frames:count,first_frame:manifest.first_frame,
  last_frame:manifest.last_frame,capture_passed:manifest.passed,journal_sha256:hash,
  telemetry_sha256:manifest.raw_sha256['telemetry.json'],camera_views:cameras,
  method:'Use the public replay join, camera-box helpers, inline reviewers and symbol memory. Match same-class symbols of at least 0.5 memory opacity to native bodies within 35 m, closest admissible pair first, one drawing per actor per frame. Gates are 0.65 m for props and 3 m for vehicles. A fresh contributing source has a current linked rectangle in a recorded view and at least one lidar hit. To corroborate attribution, require normalized rectangle IoU of at least 0.15 with the projected native 3D body and either no competing body reaching 0.15 or a margin of at least 0.05. Record the best competing vehicle or prop overlap. Chase has no detector overlay. Counts include occluded bodies and symbols outside the current Follow viewport.',
  roles:reports,highlights,publication_witnesses:publicationWitnesses},null,2)+'\n');
