// The producer supplies detections from individual native exposures. Identity is
// associated in world XY using only ego localization and earlier measurements.
const CAMERAS = ['front', 'left', 'right', 'rear_left', 'rear_right'];
const finite = Number.isFinite;

function validImage(observation) {
  const box = observation?.box;
  return observation?.label === 'traffic_cone' && CAMERAS.includes(observation.camera) &&
    Array.isArray(box) && box.length === 4 && box.every(v => finite(v) && v >= 0 && v <= 1) &&
    box[2] > box[0] && box[3] > box[1];
}

export function coneReviewer() {
  let previous, result, serial = 0;
  const anchors = new Map(), competing = new Map(), strongCars = new Map();
  return (frame, observations = frame?.cone_observations) => {
    if (!frame) return frame;
    const sameRun = previous && frame.run_id === previous.run_id && frame.scenario === previous.scenario &&
      frame.seed === previous.seed;
    if (sameRun && frame.frame === previous.frame && frame.sim_time === previous.sim_time) return result;
    const t = frame.sim_time ?? frame.video_time;
    if (!sameRun || !(t > (previous.sim_time ?? previous.video_time)) || frame.frame <= previous.frame) {
      anchors.clear();
      competing.clear();
      strongCars.clear();
      serial = 0;
    }
    previous = frame;
    for (const [id, anchor] of anchors) if (t - anchor.time > 8) anchors.delete(id);
    for (const [id, conflict] of competing) if (t-conflict.time > 1) competing.delete(id);
    for (const [id, time] of strongCars) if(t-time>4)strongCars.delete(id);
    for(const track of frame.tracks??[])if(track.label==='car'&&track.cameras?.length&&track.confidence>=.60)
      strongCars.set(track.id,t);
    if (!observations?.length && !anchors.size) return result = frame;
    const ego = frame.observer_map?.ego;
    const located = Array.isArray(ego) && ego.length >= 3 && ego.slice(0, 3).every(finite);
    const [ex, ey, yaw] = located ? ego : [0, 0, 0];
    const c = Math.cos(yaw * Math.PI / 180), s = Math.sin(yaw * Math.PI / 180);
    const tracks = [], boxes = [], candidates = [];
    for (const o of (Array.isArray(observations)?observations:[]).filter(validImage)) {
      const box = {label:o.label, view:o.camera, box:o.box, score:o.confidence, range_supported:false};
      boxes.push(box);
      if (!finite(o.x) || !finite(o.y) || !(o.lidar_hits > 0) || !o.sources?.includes('lidar')) continue;
      const wx = ex + c * o.x - s * o.y, wy = ey + s * o.x + c * o.y;
      candidates.push({o, box, wx, wy});
    }
    // Two cameras can see the same drum from different bearings. Preserve both
    // source rectangles and give the scene a single physical footprint.
    const groups = [];
    for (const candidate of candidates) {
      const group = groups.find(g => g.every(a => a.o.camera !== candidate.o.camera &&
        Math.hypot(a.wx - candidate.wx, a.wy - candidate.wy) <= .5));
      if (group) group.push(candidate);
      else groups.push([candidate]);
    }
    const positions = groups.map(group => group.reduce((best, row) => row.o.lidar_hits > best.o.lidar_hits ? row : best));
    const pairs = [];
    if (located) for (const [index, row] of positions.entries()) for (const [id, anchor] of anchors) {
      const distance = Math.hypot(row.wx - anchor.wx, row.wy - anchor.wy);
      if (distance <= .85) pairs.push({index, id, distance});
    }
    pairs.sort((a, b) => a.distance - b.distance);
    const matches = new Map(), used = new Set();
    for (const {index, id} of pairs) if (!matches.has(index) && !used.has(id)) {
      matches.set(index, id); used.add(id);
    }
    for (const [i, row] of positions.entries()) {
      const id = matches.get(i) ?? 'C' + (++serial);
      const prior = anchors.get(id);
      // A fixed channelizer does not move with the car as the observed face
      // changes. A bounded causal mean suppresses return quantization.
      const n = Math.min(5, (prior?.n ?? 0) + 1);
      const wx = prior ? prior.wx + (row.wx - prior.wx) / n : row.wx;
      const wy = prior ? prior.wy + (row.wy - prior.wy) / n : row.wy;
      const {o} = row;
      for (const entry of groups[i]) Object.assign(entry.box, {track_id:id, range_supported:true});
      const dimensions = {};
      for (const [dimension, nominal] of [['height',1],['length',.70],['width',.70]]) {
        const value = finite(o[dimension]) && o[dimension] > 0 ? o[dimension] : nominal;
        dimensions[dimension] = prior ? prior.track[dimension] + (value-prior.track[dimension])/n : value;
      }
      const track = {id, label:o.label, x:o.x, y:o.y, world_x:wx, world_y:wy, ...dimensions,
        z:o.z, sources:o.sources, lidar_hits:o.lidar_hits, range_source:o.range_source,
        cameras:groups[i].map(r => r.o.camera), confidence:o.confidence};
      tracks.push(track);
      if (located) anchors.set(id, {wx, wy, n, time:t, track});
    }
    if (located) for (const [id, anchor] of anchors) {
      const age = t - anchor.time;
      if (age <= 0 || age > 1.25001) continue;
      const dx = anchor.wx - ex, dy = anchor.wy - ey;
      const x = c * dx + s * dy, y = -s * dx + c * dy;
      const hits = (frame.lidar_xyz??[]).filter(p => p[2] > .18 && p[2] < 1.65 && Math.hypot(p[0]-x,p[1]-y) < .40).length;
      if (hits < 2 && age > .30001) continue;
      tracks.push({...anchor.track, x, y, cameras:[], sources:hits >= 2?['lidar']:[], range_source:'footprint',
        lidar_hits:hits >= 2?hits:0, age_s:age, coasting:hits < 2});
    }
    // A weak generic car class can fit a full car behind the near face of a
    // drum. Reconcile it only when its entire 3D patch belongs to a repeatedly
    // measured cone. A new independent surface or stronger RGB reading wins.
    const conflicts = [];
    if (located) for (const track of frame.tracks??[]) {
      if (track.label!=='car' || strongCars.has(track.id)) { competing.delete(track.id); continue; }
      const prior=competing.get(track.id);
      if (prior && track.coasting) {
        conflicts.push({track,cone_id:prior.id});
        continue;
      }
      const weakImage = track.cameras?.length && track.confidence < .60;
      const points = track.surface_xyz??[];
      let owned;
      if ((weakImage || (prior && !track.cameras?.length)) && points.length>=4) {
        for (const [id,anchor] of anchors) {
          if (anchor.n<2) continue;
          if (points.every(([x,y,z])=>z>.10 && z<anchor.track.height+.2 &&
            Math.hypot(ex+c*x-s*y-anchor.wx,ey+s*x+c*y-anchor.wy)<.48)) {
            owned=id;break;
          }
        }
      }
      if (owned) {
        competing.set(track.id,{id:owned,time:t});
        conflicts.push({track,cone_id:owned});
      } else competing.delete(track.id);
    }
    if (!boxes.length && !tracks.length && !conflicts.length) return result = frame;
    const enriched = {...frame,cone_observations:observations,cone_tracks:tracks,cone_boxes:boxes};
    if (conflicts.length) {
      const ids=new Set(conflicts.map(r=>r.track.id));
      enriched.cone_conflicts=conflicts;
      enriched.tracks=frame.tracks.filter(track=>!ids.has(track.id));
      if(frame.camera_detections) enriched.camera_detections=Object.fromEntries(
        Object.entries(frame.camera_detections).map(([view,detections])=>[view,Array.isArray(detections)?detections.filter(b=>!ids.has(b.track_id)):detections]));
      if(frame.detections) enriched.detections=frame.detections.filter(b=>!ids.has(b.track_id));
    }
    return result = enriched;
  };
}

export function applyConeReview(replay, review, entry) {
  const frames = Array.isArray(replay) ? replay : replay?.frames;
  if (!review && !frames?.some(f => f?.cone_observations?.length)) return replay;
  let rows;
  if (review) {
    if (!frames?.length || review.schema_version !== 1 || review.trial_id !== (entry?.trial_id ?? frames[0].run_id) ||
        review.first_frame !== frames[0].frame || review.last_frame !== frames.at(-1).frame ||
        review.capture_frames !== frames.length || !Array.isArray(review.frames)) throw Error('Cone observations do not match this recording.');
    const available = new Set(frames.map(f => f.frame));
    rows = new Map();
    for (const row of review.frames) {
      if (!available.has(row?.frame) || rows.has(row.frame) || !Array.isArray(row.cone_observations)) throw Error('Invalid cone exposure.');
      rows.set(row.frame, row.cone_observations);
    }
  }
  const qualify = coneReviewer();
  const qualified = frames.map(f => qualify(f, rows ? rows.get(f.frame) ?? [] : f.cone_observations));
  return Array.isArray(replay) ? qualified : {...replay, frames:qualified};
}
