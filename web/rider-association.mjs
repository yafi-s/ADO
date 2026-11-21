// A camera can resolve the wheels and the rider separately. Only the rider has
// measured depth here. Require their boxes to move together before combining the display.
const center = b => [(b[0]+b[2])/2, (b[1]+b[3])/2];
const valid = b => Array.isArray(b) && b.length===4 && b.every(Number.isFinite) && b[2]>b[0] && b[3]>b[1];

function aligned(person, bicycle) {
  if (!valid(person) || !valid(bicycle)) return false;
  const [px, py] = center(person), [bx, by] = center(bicycle);
  const w = person[2]-person[0], h = person[3]-person[1];
  const overlapX = Math.min(person[2],bicycle[2])-Math.max(person[0],bicycle[0]);
  return Math.abs(px-bx) < .32*w && overlapX > .65*Math.min(w, bicycle[2]-bicycle[0]) &&
    by > py+.12*h && bicycle[1] > person[1]+.15*h && bicycle[1] < person[3] &&
    bicycle[3] >= person[3]-.04*h && bicycle[3] < person[3]+.6*h;
}

function paired(frame) {
  const found = [];
  for (const [camera, boxes] of Object.entries(frame.camera_detections ?? {})) {
    const people = boxes.filter(b => b.label==='person');
    const bicycles = boxes.filter(b => b.label==='bicycle' && valid(b.box));
    for (const bike of bicycles) {
      if (!Number.isFinite(bike.image_id) || bike.range_supported || bike.track_id != null) continue;
      const candidates = people.filter(p => aligned(p.box, bike.box));
      if (candidates.length!==1) continue;
      const person = candidates[0];
      if (!person.range_supported || bicycles.filter(b => aligned(person.box,b.box)).length!==1) continue;
      const track = frame.tracks?.find(t => t.id===person.track_id && t.label==='person' && !t.coasting &&
        t.cameras?.includes(camera) && t.sources?.includes('lidar') && t.lidar_hits>=3);
      if (!track) continue;
      found.push({id:track.id, camera, image_id:bike.image_id, time:frame.sim_time,
        person:center(person.box), bike:center(bike.box), height:person.box[3]-person.box[1], score:bike.score});
    }
  }
  return found;
}

export function bicycleAssociations(frames, index=frames.length-1) {
  const current=frames[index], result=new Map();
  if (current?.perception_version!=='surround-v1' || !Number.isFinite(current.sim_time)) return result;
  const history=[];
  for (let i=index; i>=0 && index-i<=12; --i) {
    const frame=frames[i];
    if (!frame || current.frame-frame.frame!==index-i || frame.run_id!==current.run_id ||
        current.sim_time-frame.sim_time>1.20001) break;
    history.push(frame);
  }
  const groups=new Map();
  for (const frame of history.reverse()) {
    for (const row of paired(frame)) {
      const key=JSON.stringify([row.id,row.camera,row.image_id]);
      let path=groups.get(key);
      if (!path || row.time-path.at(-1).time>.45) groups.set(key,path=[]);
      path.push(row);
    }
  }
  for (const path of groups.values()) {
    const first=path[0], last=path.at(-1);
    if (path.length<3 || last.time-first.time<.39999 || current.sim_time-last.time>.45 ||
        !path.some(p => p.score>=.55)) continue;
    const motion=(a,b) => [b[0]-a[0], b[1]-a[1]];
    const rider=motion(first.person,last.person), bike=motion(first.bike,last.bike);
    const bikeNet=Math.hypot(...bike), riderNet=Math.hypot(...rider);
    if (bikeNet<.02 || riderNet<.02 || bike[0]*rider[0]+bike[1]*rider[1]<.9*bikeNet*riderNet ||
        Math.abs(bike[0]-rider[0])>.014 || Math.abs(bike[1]-rider[1])>.24*last.height) continue;
    const target=current.tracks?.find(t => t.id===last.id && t.label==='person' &&
      (t.coasting ? t.age_s<=.2 : t.sources?.includes('lidar') && t.lidar_hits>=3));
    if (!target || current.tracks.some(t => t.label==='bicycle' && Math.hypot(t.x-target.x,t.y-target.y)<2.5)) continue;
    const fresh=current.camera_detections?.[last.camera] ?? [];
    const nowBike=fresh.find(b => b.label==='bicycle' && b.image_id===last.image_id);
    if (nowBike) {
      const people=fresh.filter(p => p.label==='person');
      const together=people.filter(p => aligned(p.box,nowBike.box));
      if (nowBike.range_supported || together.length>1 || (people.some(p=>p.track_id===last.id) && !together.length) ||
          together.some(p => p.track_id!=null && p.track_id!==last.id)) continue;
    }
    result.set(last.id,{label:'bicycle', classification:'camera bicycle with measured person',
      depth_source:target.coasting?'recent person lidar':'person lidar', range_age_s:target.age_s,
      camera:last.camera, image_id:last.image_id, first_s:first.time,
      last_s:last.time, paired_observations:path.length});
  }
  return result;
}
