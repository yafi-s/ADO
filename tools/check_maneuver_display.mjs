// Compare displayed work-zone and slow-traffic objects with the native bodies after rendering.
import fs from 'node:fs';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {createInterface} from 'node:readline';
import {advance, contextOf, createState, laneAxes} from '../web/object-memory.mjs';
import {barrierReviewer} from '../web/barrier-review.mjs';
import {coneReviewer} from '../web/cone-observations.mjs';

const [stage, output] = process.argv.slice(2);
if (!stage || !output) throw Error('Provide a sealed native capture directory and an output JSON file.');
const read = file => JSON.parse(fs.readFileSync(file));
const manifest = read(path.join(stage, 'capture.json'));
const town = read(path.join(stage, 'raw/town.json'));
const features = read(new URL(`../web/media/map-features/${town.producer.map}.json`, import.meta.url));
const endpoints = (a,b) => [a,b].map(p => p.slice(0,2).map(n=>n.toFixed(2)).join(',')).join('|');
const lanes = (town.roads ?? []).map(l => l.points ?? l).filter(points => points.length > 1);
const directions = new Map((features.lanes ?? []).map(l => [endpoints(l.a,l.b),l.forward]));
const axis = laneAxes(lanes,lanes.map(l => directions.get(endpoints(l[0],l.at(-1)))));
const state = createState(), deps = {laneAxis:axis, laneDistance:axis.distance};
const qualifyBarrier = barrierReviewer(), qualifyCone = coneReviewer();
const roles = ['in-lane-obstruction', 'slow-traffic', 'adjacent-front', 'adjacent-rear'];
const classes = {cone:'traffic_cone', barrier:'barrier', car:'car', van:'car', vehicle:'car', bus:'bus', truck:'truck'};
const reports = Object.fromEntries(roles.map(role => [role, {nearby_actor_frames:0, matched_drawings:0, fresh_camera_and_range:0, actors:{}}]));
const events = Object.fromEntries(manifest.summary.events.map(e=>[e.kind,e]));
const highlights = {};
for (const [role, marks] of [['in-lane-obstruction',['exposed_s','alternate_alongside_s','returned_to_route_s']],
                             ['slow-traffic',['occupied_gap_wait_s','moving_alongside_s','returned_to_route_s']]]) {
  for (const mark of marks) {
    const time = events[role]?.[mark];
    if (Number.isFinite(time)) highlights[`${role}/${mark}`] = {frame:manifest.first_frame+Math.round(time/.1), objects:[]};
  }
}

function bodyCenter(a) {
  if (a.type === 'cone' || a.type === 'barrier') return [a.x,a.y];
  const [x,y,z] = a.bbox_offset_m ?? [0,0,0];
  const [pitch,yaw,roll] = [a.pitch ?? 0,a.yaw ?? 0,a.roll ?? 0].map(d=>d*Math.PI/180);
  const cp=Math.cos(pitch),sp=Math.sin(pitch),cy=Math.cos(yaw),sy=Math.sin(yaw),cr=Math.cos(roll),sr=Math.sin(roll);
  return [a.x+cp*cy*x+(cy*sp*sr-sy*cr)*y+(-cy*sp*cr-sy*sr)*z,
          a.y+cp*sy*x+(sy*sp*sr+cy*cr)*y+(-sy*sp*cr+cy*sr)*z];
}

const fingerprint=createHash('sha256');
const input=fs.createReadStream(path.join(stage,'raw/perception.jsonl'));
input.on('data',chunk=>fingerprint.update(chunk));
let ctx=null, count=0;
for await (const line of createInterface({input,crlfDelay:Infinity})) {
  const raw=JSON.parse(line);
  if (raw.frame !== manifest.first_frame+count++) throw Error('Nonconsecutive native journal');
  const {ego,ego_z,others}=raw.observer_map;
  // Reference bodies never enter the drawing pipeline.
  const inlet=qualifyBarrier({...raw,observer_map:{ego,ego_z}});
  const frame=qualifyCone(inlet,inlet.cone_observations);
  ctx=contextOf(frame,ctx,()=>0,raw.sim_time);
  const drawings=advance(state,ctx,deps).filter(o=>o.opacity>=.5);
  const tracks=new Map([...(frame.tracks??[]),...(frame.cone_tracks??[])].map(t=>[t.id,t]));
  const references=others.filter(a=>classes[a.type]).map(a=>({actor:a,xy:bodyCenter(a)}))
    .filter(r=>Math.hypot(r.xy[0]-ego[0],r.xy[1]-ego[1])<=35);
  const pairs=[];
  for (const {actor,xy} of references) {
    if (reports[actor.role]) reports[actor.role].nearby_actor_frames++;
    for (const object of drawings) {
      if (object.label!==classes[actor.type]) continue;
      const error=Math.hypot(xy[0]-object.x,xy[1]-object.z);
      const gate=['barrier','cone'].includes(actor.type) ? .65 : 3;
      if (error<=gate) pairs.push({actor,object,error});
    }
  }
  const assigned=new Set(), used=new Set();
  for (const {actor,object,error} of pairs.sort((a,b)=>a.error-b.error)) {
    if (assigned.has(actor.actor_id)||used.has(object.key)) continue;
    assigned.add(actor.actor_id); used.add(object.key);
    const result=reports[actor.role];
    if (!result) continue;
    result.matched_drawings++;
    const track=tracks.get(object.sourceId);
    const fresh=Boolean(track && !track.coasting && track.cameras?.length && track.lidar_hits>0);
    if (fresh) result.fresh_camera_and_range++;
    const evidence={frame:raw.frame,seconds:raw.sim_time,actor_id:actor.actor_id,source_id:object.sourceId,
      class:object.label,center_error_m:Number(error.toFixed(3)),fresh_camera_and_range:fresh,cameras:track?.cameras??[]};
    const a=result.actors[actor.actor_id] ??= {matched_frames:0,first:evidence};
    a.matched_frames++;
    if (fresh && !a.first_fresh) a.first_fresh=evidence;
    for (const h of Object.values(highlights)) if (h.frame===raw.frame) h.objects.push(evidence);
  }
}
const hash=fingerprint.digest('hex');
if (count!==manifest.summary.frames || hash!==manifest.raw_sha256['perception.jsonl']) throw Error('Capture identity or frame count mismatch');
fs.writeFileSync(output,JSON.stringify({trial:manifest.trial_id,frames:count,journal_sha256:hash,
  method:'Native exposures within 35m. Match rider symbols of at least 0.5 opacity to registered bodies of the same class, closest admissible pair first, one drawing per actor per frame. Gates are 0.65m for props and 3m for vehicles. Counts include occluded actors and drawings outside the current Follow viewport. Fresh means a current ranged source with camera attribution and at least one lidar hit. Coverage applies to the selected roles.',
  roles:reports,highlights},null,2)+'\n');
