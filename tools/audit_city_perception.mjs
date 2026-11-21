// Offline scorer for a city replay. Observer actors enter only after the display has produced its output.
import {createHash} from 'node:crypto';
import fs from 'node:fs';
import {execFileSync} from 'node:child_process';
import path from 'node:path';
import {pathToFileURL, fileURLToPath} from 'node:url';
import {parseArgs} from 'node:util';
import {createGunzip,gunzipSync} from 'node:zlib';
import {Transform} from 'node:stream';

const root = fileURLToPath(new URL('../', import.meta.url));
const rad = Math.PI / 180;
const limit = 35, gate = 3;
const truthLabels = {pedestrian:'person', car:'car', cyclist:'bicycle', van:'car', bus:'bus',
  bicycle:'bicycle', motorcycle:'motorcycle', truck:'truck'};
const rounded = value => Number(value.toFixed(3));
const family = label => label === 'person' ? 'person' : ['car','vehicle','bus','van','truck','bicycle','motorcycle'].includes(label) ? 'wheeled' : label;
const sum = values => values.reduce((a,b) => a+b, 0);
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const readJSON = name => JSON.parse(fs.readFileSync(name));
export function sector(x,y) {
  const angle = Math.atan2(y,x) / rad;
  if (Math.abs(angle) <= 45) return 'front';
  if (Math.abs(angle) >= 135) return 'rear';
  return angle < 0 ? 'left' : 'right';
}

// Rectangular Hungarian assignment; each prediction can also take an unmatched column.
// The unmatched penalty exceeds the sum of every feasible distance, so cardinality wins first.
export function match(actors, predictions, distanceGate=gate, requireFamily=true) {
  const rows = predictions.length, cols = actors.length + rows;
  const penalty = (Math.min(actors.length, rows) + 1) * (distanceGate + 1);
  const cost = predictions.map(p => actors.map(a => {
    const d = Math.hypot(a.x-p.x,a.y-p.y);
    return (!requireFamily || family(a.label) === family(p.label)) && d <= distanceGate ? d : 2*penalty;
  }));
  const u = Array(rows+1).fill(0), v = Array(cols+1).fill(0);
  const owner = Array(cols+1).fill(0), from = Array(cols+1).fill(0);
  for (let row=1; row<=rows; row++) {
    let col=0;
    owner[0]=row;
    const used = Array(cols+1).fill(false), slack = Array(cols+1).fill(Infinity);
    do {
      used[col]=true;
      const i=owner[col];
      let delta=Infinity, next=0;
      for (let j=1;j<=cols;j++) {
        if (used[j]) continue;
        const candidate=(cost[i-1][j-1] ?? penalty) - u[i] - v[j];
        if (candidate < slack[j]) { slack[j]=candidate; from[j]=col; }
        if (slack[j] < delta) { delta=slack[j]; next=j; }
      }
      for (let j=0;j<=cols;j++) {
        if (used[j]) { u[owner[j]]+=delta; v[j]-=delta; }
        else slack[j]-=delta;
      }
      col=next;
    } while (owner[col]);
    do { const prev=from[col]; owner[col]=owner[prev]; col=prev; } while (col);
  }
  return actors.flatMap((actor,j) => {
    const i=owner[j+1]-1;
    return i>=0 && cost[i][j]<=distanceGate ? [{actor, prediction:predictions[i], distance:cost[i][j]}] : [];
  });
}

function localPose(x,y,ego) {
  const yaw=ego[2]*rad, dx=x-ego[0], dy=y-ego[1];
  return {x:dx*Math.cos(yaw)+dy*Math.sin(yaw),y:-dx*Math.sin(yaw)+dy*Math.cos(yaw)};
}
function inRange(p) { return Math.hypot(p.x,p.y)<=limit; }
function observerActors(frame,others=frame.observer_map.others,sensorRadius=limit) {
  const {ego}=frame.observer_map;
  return others.filter(a => Object.hasOwn(truthLabels,a.type)).map(a => {
    const {x,y}=localPose(a.x,a.y,ego), h=(a.yaw-ego[2])*rad;
    const [length,,width]=a.dimensions_m;
    let points=0;
    if (Math.hypot(x,y)<=sensorRadius) {
      for (const [px,py] of frame.lidar ?? []) {
        if (Math.abs(px)<2.5 && Math.abs(py)<1.1) continue;
        const dx=px-x, dy=py-y;
        if (Math.abs(dx*Math.cos(h)+dy*Math.sin(h))<=length/2+.25 &&
            Math.abs(dy*Math.cos(h)-dx*Math.sin(h))<=width/2+.25) points++;
      }
    }
    return {id:a.actor_id, label:truthLabels[a.type], type:a.type, role:a.role,blueprint:a.blueprint, x,y, world:[a.x,a.y], z:a.z,
      yaw:a.yaw,dimensions:a.dimensions_m, points, sector:sector(x,y)};
  });
}
const bucket = () => ({actors:0,hits:0,predictions:0,unmatched_predictions:0,exact_label_hits:0});
const stats = () => ({overall:bucket(), sector:Object.fromEntries(['front','left','right','rear'].map(s=>[s,bucket()])),
  by_class:{}, by_status:{}, measured_views:{}, by_source:{}, spatial_only:bucket(), support:bucket(), residuals:[], states:new Map()});
function bins(s,a) {
  return [s.overall,s.sector[a.sector],s.by_class[a.type]??=bucket(),...(a.points>=2 ? [s.support] : [])];
}
function evaluate(s, actors, predictions, frameNumber) {
  // Match all observations before restricting the two denominators. A track straddling the
  // 35 m evaluation boundary can still have a real counterpart just outside the boundary.
  const pairs=match(actors,predictions);
  const spatial=match(actors,predictions,gate,false);
  s.spatial_only.hits+=spatial.filter(p=>inRange(p.actor)).length;
  s.spatial_only.unmatched_predictions+=predictions.filter(inRange).length-spatial.filter(p=>inRange(p.prediction)).length;
  const byActor=new Map(pairs.map(p=>[p.actor,p]));
  const byPrediction=new Map(pairs.map(p=>[p.prediction,p]));
  for (const a of actors.filter(inRange)) {
    const m=byActor.get(a);
    for (const b of bins(s,a)) {
      b.actors++;
      if (m) { b.hits++; b.exact_label_hits+=Number(a.label===m.prediction.label); }
    }
    const t=s.states.get(a.id)??{id:a.id, type:a.type, role:a.role, frames:0, hits:0,
      runs:0, reacquisitions:0, direct_id_switches:0, ids:new Set(), gap:0, longest_gap_frames:0};
    const continuous=t.frame===frameNumber-1;
    if (!continuous) { t.previous=null; t.gap=0; t.ever=false; }
    t.frames++;
    if (m) {
      t.hits++;
      t.ids.add(m.prediction.id);
      if (t.previous==null) { t.runs++; if (t.ever) t.reacquisitions++; }
      else if (t.previous!==m.prediction.id) t.direct_id_switches++;
      t.ever=true;
      t.gap=0;
      s.residuals.push(m.distance);
    } else {
      t.gap++;
      t.longest_gap_frames=Math.max(t.gap,t.longest_gap_frames);
    }
    t.previous=m?.prediction.id??null;
    t.frame=frameNumber;
    s.states.set(a.id,t);
  }
  for (const p of predictions.filter(inRange)) {
    const views=(p.cameras??[]).map(view=>s.measured_views[view]??=bucket());
    const source=p.sources?.length?[s.by_source[[...p.sources].sort().join('+')]??=bucket()]:[];
    for (const b of [s.overall,s.sector[sector(p.x,p.y)],s.by_status[p.status]??=bucket(),...views,...source]) {
      b.predictions++;
      if (!byPrediction.has(p)) b.unmatched_predictions++;
    }
  }
  return {actors:actors.filter(inRange).map(a=>{
    const m=byActor.get(a);
    return {id:a.id,type:a.type,role:a.role,sector:a.sector,x:rounded(a.x),y:rounded(a.y),range_m:rounded(Math.hypot(a.x,a.y)),
      sampled_xy_returns:a.points,match:m?{id:m.prediction.id,label:m.prediction.label,
        status:m.prediction.status,error_m:rounded(m.distance),cameras:m.prediction.cameras}:null};
  }),unmatched:predictions.filter(p=>inRange(p)&&!byPrediction.has(p)).map(p=>{
    const closest=actors.reduce((best,a)=>{
      const distance=Math.hypot(p.x-a.x,p.y-a.y);
      return !best||distance<best.distance?{id:a.id,role:a.role,distance}:best;
    },null);
    return {id:p.id,label:p.label,x:rounded(p.x),y:rounded(p.y),sector:sector(p.x,p.y),
      range_m:rounded(Math.hypot(p.x,p.y)),status:p.status,nearest:closest?{
        id:closest.id,role:closest.role,error_m:rounded(closest.distance)}:null};
  })};
}
function summarize(s) {
  const percentile = q => s.residuals.length ? rounded(s.residuals[Math.ceil(s.residuals.length*q)-1]) : null;
  s.residuals.sort((a,b)=>a-b);
  const residual={p50_m:percentile(.5),p95_m:percentile(.95)};
  const recalls=b=>({actor_frames:b.actors,matched:b.hits,exact_label_hits:b.exact_label_hits});
  const both=b=>({...recalls(b),prediction_frames:b.predictions,unmatched_predictions:b.unmatched_predictions});
  const predictionCounts=groups=>Object.fromEntries(Object.entries(groups).map(([k,b])=>[k,{
    prediction_frames:b.predictions,unmatched_predictions:b.unmatched_predictions}]));
  const actors=[...s.states.values()].map(({frame,previous,gap,ever,ids,...t})=>({...t,distinct_ids:ids.size}));
  return {overall:both(s.overall),sector:Object.fromEntries(Object.entries(s.sector).map(([k,v])=>[k,both(v)])),
    by_class:Object.fromEntries(Object.entries(s.by_class).map(([k,v])=>[k,recalls(v)])),
    by_status:predictionCounts(s.by_status),by_measuring_camera:predictionCounts(s.measured_views),by_sensor:predictionCounts(s.by_source),
    position_only:{...both(s.overall),exact_label_hits:undefined,matched:s.spatial_only.hits,unmatched_predictions:s.spatial_only.unmatched_predictions},
    sampled_lidar_support:recalls(s.support), matched_position_residual:residual,
    reacquisitions:sum(actors.map(a=>a.reacquisitions)),direct_id_switches:sum(actors.map(a=>a.direct_id_switches)),actors};
}

function recording(directory) {
  const input=['replay-pack.json.gz','perception.jsonl','perception.jsonl.gz','replay.json'].find(name=>fs.existsSync(path.join(directory,name)));
  if (!input) throw new Error('Recording has no replay or perception journal');
  const file=path.join(directory,input), digest=createHash('sha256');
  const size=fs.statSync(file).size;
  if (!size) throw new Error('Empty replay');
  const info={input,source_bytes:size,partial_tail_ignored:false};
  async function* readFrames() {
    if (input.startsWith('perception.jsonl')) {
      // Freeze the file length, so a running recorder can keep writing without changing this audit.
      let pending='';
      const raw=fs.createReadStream(file,{end:size-1}).pipe(new Transform({
        transform(chunk,encoding,callback) {digest.update(chunk);callback(null,chunk);}
      }));
      const stream=(input.endsWith('.gz')?raw.pipe(createGunzip()):raw).setEncoding('utf8');
      for await (const chunk of stream) {
        const lines=(pending+chunk).split('\n');
        pending=lines.pop();
        for (const line of lines) if (line.trim()) yield JSON.parse(line);
      }
      info.partial_tail_ignored=Boolean(pending.trim());
    } else {
      const bytes=fs.readFileSync(file);
      digest.update(bytes);
      const data=input.endsWith('.gz') ? JSON.parse(gunzipSync(bytes)).replay : JSON.parse(bytes);
      yield* Array.isArray(data)?data:data.frames;
    }
    info.sha256=digest.digest('hex');
  }
  return {frames:readFrames(),info};
}

export async function audit(directory,{memoryRevision,memoryFile:sourceFile,frameFile,townFile,staticFile,probes=[],onFrame}={}) {
  const {frames,info}=recording(directory);
  if(memoryRevision&&sourceFile)throw new Error('Choose one display source');
  const memoryFile=sourceFile?path.resolve(sourceFile):path.join(root,'web/object-memory.mjs');
  const source=memoryRevision
    ? execFileSync('git',['show',memoryRevision+':web/object-memory.mjs'],{cwd:root})
    : fs.readFileSync(memoryFile);
  const memoryURL=memoryRevision?'data:text/javascript;base64,'+source.toString('base64'):pathToFileURL(memoryFile);
  const {advance,contextOf,createState,laneAxes}=await import(memoryURL);
  const townBytes=fs.readFileSync(townFile??path.join(directory,'town.json'));
  const town=JSON.parse(townBytes);
  const staticBytes=staticFile?fs.readFileSync(staticFile):null;
  const parked=staticBytes?JSON.parse(staticBytes):null;
  if (parked && (parked.map!==town.producer.map||!Array.isArray(parked.actors))) throw new Error('Static reference does not match the recorded town');
  const featuresPath=path.join(root,'web/media/map-features',path.basename(town.producer.map)+'.json');
  const features=fs.existsSync(featuresPath)?readJSON(featuresPath):{};
  const ends=(a,b)=>[a,b].map(p=>p.slice(0,2).map(n=>n.toFixed(2)).join(',')).join('|');
  const described=new Map((features.lanes??[]).map(l=>[ends(l.a,l.b),l.forward]));
  const lanes=town.roads.map(l=>l.points??l).filter(l=>l.length>1);
  const laneAxis=laneAxes(lanes,lanes.map(l=>described.get(ends(l[0],l.at(-1)))));
  const state=createState(), deps={laneAxis,laneDistance:laneAxis.distance};
  const layers={tracker:stats(),display:stats()}, samples=[];
  const supplemented=parked?{tracker:stats(),display:stats()}:null;
  const target=new Set(probes);
  let context=null, prior=-Infinity, first=null, count=0;
  const fd=frameFile?fs.openSync(frameFile,'w'):null;
  try {
    for await (const frame of frames) {
      if (!Array.isArray(frame.observer_map?.others) || !Array.isArray(frame.observer_map?.ego) ||
          !Array.isArray(frame.tracks) || !Number.isInteger(frame.frame) || frame.frame<=prior) {
        throw new Error('Missing observer reference, tracker, ego or monotonic CARLA frame at '+frame.frame);
      }
      prior=frame.frame;
      if (frame.map && frame.map!==town.producer.map) throw new Error('Recording and supplied lane geometry identify different maps');
      first??=prior;
      count++;
      const {ego,ego_z}=frame.observer_map;
      const {sim_time,video_time,seed,run_id,scenario,tracks,detections,camera_detections,lidar,radar,corner_radar,perception_version}=frame;
      const input={frame:frame.frame,sim_time,video_time,seed,run_id,scenario,tracks,detections,
        camera_detections,lidar,radar,corner_radar,perception_version,observer_map:{ego,ego_z}};
      context=contextOf(input,context);
      const display=advance(state,context,deps).map(p=>({id:p.key,sourceId:p.sourceId,label:p.label,...localPose(p.x,p.z,ego),
        world:[p.x,p.z],foot:p.y,heading:p.heading,dimensions:[p.length,p.height,p.width],opacity:p.opacity,
        status:p.held?'remembered':'current'}));
      const actors=observerActors(frame,frame.observer_map.others,onFrame?60:limit);
      const tracker=frame.tracks.map(p=>({...p,status:p.coasting?'coasting':'measured'}));
      const record={frame:frame.frame,sim_time:frame.sim_time,ego,progress_m:frame.progress_m,active_event:frame.active_event,
        event_stations:Object.fromEntries((frame.events??[]).map(e=>[e.kind,e.station_m])),
        tracker:evaluate(layers.tracker,actors,tracker,frame.frame),
        display:evaluate(layers.display,actors,display,frame.frame)};
      const all=parked?[...actors,...observerActors(frame,parked.actors,onFrame?60:limit).filter(a=>Math.hypot(a.x,a.y)<=100)]:actors;
      if (parked) {
        const trackRows=evaluate(supplemented.tracker,all,tracker,frame.frame);
        const displayRows=evaluate(supplemented.display,all,display,frame.frame);
        record.with_parked={tracker:trackRows.unmatched,display:displayRows.unmatched};
        record.parked={tracker:trackRows.actors.filter(a=>a.role==='parked-map-vehicle'),
          display:displayRows.actors.filter(a=>a.role==='parked-map-vehicle')};
      }
      if (onFrame) onFrame({frame,actors:all,tracker,display});
      if (fd!==null) fs.writeSync(fd,JSON.stringify(record)+'\n');
      if (target.has(frame.frame)) samples.push({...record,detection_labels:(frame.detections??[]).map(d=>d.label),
        detection_age_s:frame.detection_age_s});
    }
  } finally {if (fd!==null) fs.closeSync(fd);}
  if (!count) throw new Error('No complete frames to audit');
  return {...info,town_sha256:hash(townBytes),memory_sha256_lf:hash(source.toString().replaceAll('\r\n','\n')),frames:count,
    first_frame:first,last_frame:prior,radius_m:limit,match_gate_m:gate,
    tracker:summarize(layers.tracker),display:summarize(layers.display),
    with_static_road_users:parked?{source_sha256:hash(staticBytes),tracker:summarize(supplemented.tracker),
      display:summarize(supplemented.display)}:undefined,probes:samples};
}

if (path.resolve(process.argv[1]??'')===fileURLToPath(import.meta.url)) {
  const {values,positionals}=parseArgs({allowPositionals:true,options:{
    output:{type:'string'},frames:{type:'string'},town:{type:'string'},'static-map':{type:'string'},
    'memory-revision':{type:'string'},'memory-file':{type:'string'},probe:{type:'string',multiple:true}}});
  if (positionals.length!==1) throw new Error('Pass one recording directory and optional --output summary.json --frames frame-matches.jsonl --probe FRAME --memory-revision COMMIT');
  const result=await audit(positionals[0],{memoryRevision:values['memory-revision'],memoryFile:values['memory-file'],frameFile:values.frames,
    townFile:values.town,staticFile:values['static-map'],probes:values.probe?.map(Number)});
  const json=JSON.stringify(result,null,2)+'\n';
  if (values.output) fs.writeFileSync(values.output,json); else process.stdout.write(json);
}
