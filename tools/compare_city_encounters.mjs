// Pair genuine encounters on the same route when vehicle pose and actor-relative pose agree.
import fs from 'node:fs';
import {createInterface} from 'node:readline';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
const eventOf={stopped_bus:'bus',occluder:'occluded',oncoming:'left',signal_queue:'signal'};
const fixed=n=>Number(n.toFixed(3));

async function encounters(file) {
  const groups=new Map();
  for await (const line of createInterface({input:fs.createReadStream(file),crlfDelay:Infinity})) {
    const f=JSON.parse(line);
    for(const a of f.tracker.actors) {
      // Background actors can have identical roles. Named event actors are unique.
      const station=f.event_stations[eventOf[a.role]??a.role];
      if (!Number.isFinite(station)||Math.abs(f.progress_m-station)>35) continue;
      const entry={frame:f.frame,t:f.sim_time,progress:f.progress_m,ego:f.ego,actor:a,
        tracker:a.match,display:f.display.actors.find(d=>d.id===a.id)?.match};
      if (!groups.has(a.role)) groups.set(a.role,[]);
      groups.get(a.role).push(entry);
    }
  }
  return groups;
}

function separation(a,b) {
  if (a.actor.sector!==b.actor.sector || Math.abs(a.progress-b.progress)>1.5) return Infinity;
  const ego=Math.hypot(a.ego[0]-b.ego[0],a.ego[1]-b.ego[1]);
  const actor=Math.hypot(a.actor.x-b.actor.x,a.actor.y-b.actor.y);
  const degrees=Math.abs(((b.ego[2]-a.ego[2]+540)%360)-180);
  return ego<=1 && actor<=1.5 && degrees<=4 ? ego+actor+degrees/10 : Infinity;
}

export function align(base,next) {
  // Monotonic one-to-one matching. Removing one pair costs more than every legal
  // pose error combined, so optimize pair count and then total pose error.
  const w=next.length+1, n=(base.length+1)*w;
  const scores=new Float64Array(n), steps=new Uint8Array(n);
  const bonus=4*(Math.min(base.length,next.length)+1);
  for (let i=1;i<=base.length;i++) for (let j=1;j<=next.length;j++) {
    const at=i*w+j;
    const skipBase=scores[at-w],skipNext=scores[at-1];
    scores[at]=Math.max(skipBase,skipNext);
    steps[at]=skipBase>=skipNext?1:2;
    const error=separation(base[i-1],next[j-1]);
    const paired=scores[at-w-1]+bonus-error;
    if (paired>scores[at]) { scores[at]=paired;steps[at]=3; }
  }
  const pairs=[];
  let i=base.length,j=next.length;
  while (i&&j) {
    const action=steps[i*w+j];
    if (action===3) pairs.push([base[--i],next[--j]]);
    else if (action===1) i--; else j--;
  }
  return pairs.reverse();
}

function continuity(samples,layer) {
  const ids=[],gaps=[];
  let current=null,gap=0,switches=0;
  for(const s of samples) {
    const id=s[layer]?.id??null;
    if (id===null) gap++;
    else {
      if (gap) {gaps.push(gap);gap=0;}
      if (current!==null&&id!==current) switches++;
      if (!ids.includes(id)) ids.push(id);
    }
    current=id;
  }
  if(gap) gaps.push(gap);
  return {matched:samples.filter(s=>s[layer]).length,frames:samples.length,ids,
    consecutive_id_changes:switches,longest_missed_frames:Math.max(0,...gaps)};
}
function range(samples) {
  return {frames:[samples[0].frame,samples.at(-1).frame],seconds:[samples[0].t,samples.at(-1).t],
    stations_m:[samples[0].progress,samples.at(-1).progress],
    tracker:continuity(samples,'tracker'),display:continuity(samples,'display')};
}
function summarize(pairs) {
  const result={pairs:pairs.length};
  for (const [side,index] of [['base',0],['candidate',1]]) {
    const samples=pairs.map(p=>p[index]);
    result[side]={tracker:samples.filter(s=>s.tracker).length,display:samples.filter(s=>s.display).length};
  }
  return result;
}
export async function compare(baseFile,nextFile) {
  const [base,next]=await Promise.all([encounters(baseFile),encounters(nextFile)]);
  const result={};
  for (const [role,first] of base) {
    const second=next.get(role);
    if (!second?.length) continue;
    const pairs=align(first,second);
    const bySector={};
    for (const sector of ['front','left','right','rear']) {
      bySector[sector]=summarize(pairs.filter(p=>p[0].actor.sector===sector));
    }
    result[role]={full_base:range(first),full_candidate:range(second),aligned:summarize(pairs),by_sector:bySector,
      paired_frames:pairs.map(([a,b])=>[a.frame,b.frame]),
      max_paired_ego_distance_m:fixed(Math.max(0,...pairs.map(([a,b])=>Math.hypot(a.ego[0]-b.ego[0],a.ego[1]-b.ego[1]))))};
  }
  return result;
}
if(path.resolve(process.argv[1]??'')===fileURLToPath(import.meta.url)) {
  if(process.argv.length!==5) throw new Error('Pass baseline JSONL, candidate JSONL and output JSON');
  fs.writeFileSync(process.argv[4],JSON.stringify(await compare(process.argv[2],process.argv[3]),null,2)+'\n');
}
