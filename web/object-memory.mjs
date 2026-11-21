// ADO object memory: the steady set of road users the display draws.
//
// The tracker reports what it measured in each frame. Its detector runs on the front camera
// every other frame, drops a track half a second after the last measurement and gives a
// re-detected object a new id. Drawn as reported, objects blink, vanish beside the vehicle
// and reappear. This module keeps one drawn object per physical object instead:
//
//   confirm   an object is drawn once it has been measured two or three times (or once, close and sure)
//   stitch    a measurement beside a remembered object of the same kind continues that object
//   hold      a stopped object stays where it was last measured, drawn as remembered, until
//             the vehicle passes it, it is plainly gone, or the vehicle drives through the spot
//   carry     a travelling object continues along its course for a second, then fades
//
// Every position comes from a recorded measurement. Nothing here reads the simulator's actors.
// It is plain data in, plain data out, so it runs in the page and under Node.

const finite = Number.isFinite;
const clamp = (value, low, high) => Math.min(high, Math.max(low, value));
const rad = Math.PI / 180;
const ease = (seconds, over) => 1 - Math.exp(-Math.max(0, seconds) / over);
const turn = (from, to) => (((to - from) % 360) + 540) % 360 - 180;   // shortest signed difference, degrees
const toward = (value, target, step) => value < target ? Math.min(target, value + step) : Math.max(target, value - step);

const WHEELED = new Set(['car', 'truck', 'bus', 'motorcycle', 'bicycle']);
const LABELS = new Set([...WHEELED, 'person', 'barrier']);
// Classes the detector confuses with one another are one family for matching.
export const familyOf = label => label === 'person' || label === 'barrier' ? label : WHEELED.has(label) ? 'wheeled' : 'unknown';

// The front camera sits this far ahead of the vehicle origin and sees 90 degrees (crosswalk/robotaxi.py).
const CAMERA_X = 1.30;
// Versioned optical registration from crosswalk/camera_geometry.py. Positions are vehicle-local.
const SURROUND_VIEWS = [
  ['front', CAMERA_X, 0, 0], ['left', 1.30, -.75, -70], ['right', 1.30, .75, 70],
  ['rear_left', -1.10, -.70, -145], ['rear_right', -1.10, .70, 145]
].map(([name,x,y,yaw])=>({name,x,y,c:Math.cos(yaw*rad),s:Math.sin(yaw*rad)}));
const VIEW_SLOPE = .85;         // inside this share of the half field of view counts as clear view
const VIEW_RANGE_M = 40;

const GRACE_S = .25;            // the detector idles on alternate frames
const CANDIDATE_S = .7;         // an unconfirmed measurement waits this long for a second one
const SURE = {confidence: .6, range: 15}, FAIR = .42;   // the detector reports boxes from .35 up
const GATE_M = {person: 1.2, wheeled: 2.4, barrier: .8, unknown: 1.5};
const DRIFT_MPS = {person: 2, wheeled: 1.5, barrier: 0, unknown: 1.5};   // how far an unseen object may have moved, per second
const TRAVEL_DRIFT_MPS = 8;
const GATE_LIMIT_M = 6;
const CARRY_S = 1;              // a travelling object continues this long without a measurement
const CARRY_AFTER_S = .4;       // once it has been seen travelling for this long
const CARRY_LIMIT_S = 3;        // and no longer than this, even with lidar returns keeping pace
const UNSEEN_IN_VIEW_S = 2;     // held in clear view with no detection over it
const SEEN_SHARE = .6;          // a box narrower than this share of the object's expected width is not it
const SEEN_PAST_S = .4;         // something farther off seen where it would stand, for this long: it has gone
const LIDAR_EXTENDS_S = {wheeled: 8, person: 2, barrier: 0, unknown: 2};   // returns on its footprint extend that
const BESIDE_S = {near: 2, far: 6};   // out of the camera's view, with no lidar returns on it
const NEAR_SILENT_S = 1;        // in view and close by, with neither a box nor lidar returns
const NEAR_M = 12;
const PERSON_BESIDE_S = 3;
const BEHIND_M = 6, RANGE_M = 60;
const EGO_HALF = {length: 2.7, width: 1.3};
// A vehicle-class detection this far from every mapped driving-lane centre is not drawn. A van
// in a parking lane sits up to 3.7 m out; in the recordings, "vehicles" farther out than this
// were planters, a kiosk, a shed and street furniture.
const OFF_ROAD_M = 4.5;
const FADE_IN_S = .3, FADE_OUT_S = .6, DIM_S = .4;
const REMEMBERED = .6;          // how solid a remembered object is drawn

// How quickly the drawn shape follows the tracker's estimates.
const SHAPE_SECONDS = .55, HEADING_SECONDS = .4, TRACK_ALPHA = .5, TRACK_BETA = .14, STILL_SECONDS = 2.4, REST_SECONDS = .35;
// An object counts as travelling once its measured positions show net, straight movement over this window.
const TRAVEL_SECONDS = 1.2, TRAVEL_METRES = {person: .5}, TRAVEL_STRAIGHTNESS = .7, TRAVEL_STRIDE = .7;
// An object that has stood still for a while needs more than that to count as setting off: the
// measured centre of a parked car shifts by a metre or two as the vehicle passes it.
const SETTLED_S = 2, SETTLED_METRES = {person: 1}, STOPPED_SHARE = .4;

// Which way the lane runs at a point, in degrees. A town export stores lane centre lines without
// saying which way they are driven, so by default this is an axis, returned turned to agree with
// `hint`. Where lanes overlap inside a junction, the one that best agrees with `hint` wins.
// `forward[i]` says lane i is driven in its stored order (true) or against it (false); with
// `driven` set, the chosen lane's driving direction is returned where that is known.
const LANE_REACH = 10;
export function laneAxes(lanes, forward = []) {
  const cells = new Map(), size = 10;
  lanes.forEach((lane, index) => {
    const way = forward[index] === true ? 0 : forward[index] === false ? 180 : null;
    for (let i = 1; i < lane.length; i++) {
      const a = lane[i - 1], b = lane[i];
      const key = Math.floor((a[0] + b[0]) / 2 / size) + ',' + Math.floor((a[1] + b[1]) / 2 / size);
      if (!cells.has(key)) cells.set(key, []);
      const degrees = Math.atan2(b[1] - a[1], b[0] - a[0]) / rad;
      cells.get(key).push([a[0], a[1], b[0], b[1], degrees, way === null ? null : degrees + way, a[2], b[2]]);
    }
  });
  // Every lane segment within reach of a point, as [distance, direction, driven direction].
  const around = (x, z) => {
    const cx = Math.floor(x / size), cz = Math.floor(z / size), found = [];
    for (let i = -1; i <= 1; i++) for (let j = -1; j <= 1; j++) for (const lane of cells.get((cx + i) + ',' + (cz + j)) || []) {
      const dx = lane[2] - lane[0], dz = lane[3] - lane[1];
      const along = clamp(((x - lane[0]) * dx + (z - lane[1]) * dz) / (dx * dx + dz * dz || 1), 0, 1);
      const distance = Math.hypot(lane[0] + along * dx - x, lane[1] + along * dz - z);
      if (distance <= LANE_REACH) found.push([distance, lane[4], lane[5],lane[6]+along*(lane[7]-lane[6])]);
    }
    return found;
  };
  const lookup = (x, z, hint, driven = false) => {
    const found = around(x, z);
    const nearest = Math.min(Infinity, ...found.map(entry => entry[0]));
    let best = null, score = Infinity;
    for (const [distance, degrees, way] of found) {
      if (distance > nearest + 1) continue;
      const axis = Math.abs(turn(hint, degrees)) > 90 ? degrees + 180 : degrees;
      const disagreement = Math.abs(turn(hint, axis)) + distance * .01;
      if (disagreement < score) {
        score = disagreement;
        best = driven && way !== null ? way : axis;
      }
    }
    return best;
  };
  // Metres from a point to the nearest lane centre, or Infinity beyond reach of any lane.
  lookup.distance = (x, z) => Math.min(Infinity, ...around(x, z).map(entry => entry[0]));
  lookup.ground = (x,z,estimate) => {
    let best=Infinity, altitude=null;
    for(const [distance,,,height] of around(x,z)){
      // Restrict surveyed pavement to the carriageway, reject another bridge deck.
      if(!finite(height)||distance>1.9||Math.abs(height-estimate)>1.5)continue;
      const score=distance+.2*Math.abs(height-estimate);
      if(score<best){best=score;altitude=height;}
    }
    return altitude;
  };
  return lookup;
}

// Where and when a frame is, and whether it continues the previous one.
export function contextOf(frame, previous = null, heightAt = () => 0, clock = 0) {
  const t = finite(frame.sim_time) ? frame.sim_time : finite(frame.video_time) ? frame.video_time : clock;
  const map = frame.observer_map;
  const located = Array.isArray(map?.ego) && map.ego.length >= 3 && map.ego.slice(0, 3).every(finite);
  const [x, z, degrees] = located ? map.ego : [0, 0, 0];
  const h = located ? (finite(map.ego_z) ? map.ego_z : heightAt(x, z)) : 0;
  const dt = previous ? t - previous.t : 0;
  const before = previous?.frame;
  const reset = !before || frame.run_id !== before.run_id || frame.scenario !== before.scenario ||
    frame.seed !== before.seed || located !== previous.located || dt <= 0 || dt > .8 ||
    (finite(before.frame) && finite(frame.frame) && frame.frame <= before.frame) ||
    (located && Math.hypot(previous.x - x, previous.z - z) > 9);
  const yaw=degrees*rad, vx=reset?0:(x-previous.x)/dt, vz=reset?0:(z-previous.z)/dt;
  return {frame,t,dt,x,z,degrees,h,located,reset,
    forwardV:vx*Math.cos(yaw)+vz*Math.sin(yaw),rightV:vz*Math.cos(yaw)-vx*Math.sin(yaw),
    yawRate:reset?0:turn(previous.degrees,degrees)*rad/dt};
}

export function createState() {
  return {objects: new Map(), ids: new Map(), images: new Map(), uncertainCars:new Map(), next: 1, warm: 0};
}

function validSample(sample) {
  return sample?.id != null && [sample.x, sample.y, sample.z, sample.length, sample.width, sample.height].every(finite) &&
    sample.length > 0 && sample.width > 0 && sample.height > 0;
}

function surfaceBounds(s){
  if(!Array.isArray(s.surface_xyz)||s.surface_xyz.length<3)return null;
  const angle=(s.heading??0)*rad,c=Math.cos(angle),n=Math.sin(angle);
  const lo=[Infinity,Infinity,Infinity],hi=[-Infinity,-Infinity,-Infinity];
  for(const point of s.surface_xyz){
    if(!Array.isArray(point)||point.length!==3||!point.every(finite))continue;
    const [x,y,z]=point;
    const p=[x*c+y*n,y*c-x*n,z];
    for(let i=0;i<3;i++){lo[i]=Math.min(lo[i],p[i]);hi[i]=Math.max(hi[i],p[i]);}
  }
  if(!finite(lo[0]))return null;
  const [a,b,z]=lo.map((lower,i)=>(lower+hi[i])/2);
  const [length,width,height]=lo.map((lower,i)=>Math.max(.15,hi[i]-lower));
  return {x:a*c-b*n,y:a*n+b*c,z,length,width,height};
}

// A low-confidence car box can range against a long hedge or wall. Mark its
// depth cluster as unclassified while the same image or range track continues.
// Preserve the measured surface, including when two fits share that image.
function rangedReading(s,frame,state,t){
  if(frame.perception_version!=='surround-v1'||!motorVehicle(s.label))return {s};
  const bounds=surfaceBounds(s);
  if(s.label!=='car')return {s,bounds};
  const linked=(s.cameras??[]).flatMap(view=>(frame.camera_detections?.[view]??[])
    .filter(b=>b.track_id===s.id&&b.range_supported&&b.label==='car')
    .map(b=>state.images.get(view+':'+b.image_id+':car')).filter(Boolean));
  const impossible=bounds && Math.max(bounds.length,bounds.width)>7.5;
  if(impossible&&linked.length&&s.confidence<.6)for(const image of linked)image.diffuse=true;
  // A few points on the edge of a previously measured tall, unclassified
  // structure cannot establish a whole passenger car. Wait for a resolved
  // image/range observation while continuing to show the measured footprint.
  const previous=state.objects.get(state.ids.get(s.id));
  const weakUpgrade=previous?.family==='unknown' && previous.hits>=2 && previous.height>2.5 &&
    linked.length && s.lidar_hits<=4 && s.confidence<.5;
  if(weakUpgrade || (linked.some(image=>image.diffuse)&&s.confidence<.6))state.uncertainCars.set(s.id,t);
  else if(linked.length && s.confidence>=.6 && !impossible)state.uncertainCars.delete(s.id);
  if(!state.uncertainCars.has(s.id))return {s,bounds};
  state.uncertainCars.set(s.id,t);
  return {s:{...s,...bounds,label:'unknown'},bounds};
}

// Does anything recorded in this frame vouch for a remembered object at ego-frame (fx, fy)?
// `local` is its heading in the vehicle frame, in radians.
function evidence(object, fx, fy, local, frame) {
  const surround = frame.perception_version === 'surround-v1' && frame.camera_detections;
  let inView = false, box = false, past = false;
  const range = Math.hypot(fx,fy);
  for (const view of SURROUND_VIEWS) {
    if (!surround && view.name !== 'front') break;
    const dx=fx-view.x, dy=fy-view.y;
    const depth=dx*view.c+dy*view.s, side=dy*view.c-dx*view.s;
    if (depth<=2.5 || Math.abs(side)>=VIEW_SLOPE*depth || range>=VIEW_RANGE_M) continue;
    inView=true;
    // Where the object would sit across the camera image, 0 at the left edge and 1 at the right,
    // and the least share of the image's width it would fill at this range.
    const u = .5 + .5 * side / depth;
    const fills = .5 * (object.family === 'person' ? .5 : Math.min(object.width, 1.8)) / depth;
    const boxes = surround ? frame.camera_detections[view.name] : frame.detections;
    for (const item of Array.isArray(boxes) ? boxes : []) {
      const span = item?.box;
      if (!Array.isArray(span) || familyOf(item.label) !== object.family || u < span[0] - .02 || u > span[2] + .02) continue;
      // A box far too small to be this object is something farther off, seen where it would stand.
      if (span[2] - span[0] >= SEEN_SHARE * fills) box = true;
      else past = true;
    }
  }
  // Lidar returns are recorded in the vehicle frame. Count those on the object's footprint.
  const c = Math.cos(local), s = Math.sin(local);
  const half = object.family === 'person' ? [.5, .5] : [object.length / 2 + .35, object.width / 2 + .35];
  let returns = 0;
  for (const p of Array.isArray(frame.lidar) ? frame.lidar : []) {
    // Returns from the vehicle's own body are not evidence of anything else.
    if (!Array.isArray(p) || (Math.abs(p[0]) < EGO_HALF.length && Math.abs(p[1]) < EGO_HALF.width)) continue;
    const dx = p[0] - fx, dy = p[1] - fy;
    if (Math.abs(dx * c + dy * s) <= half[0] && Math.abs(dy * c - dx * s) <= half[1]) returns++;
  }
  return {inView, box, past: past && !box, lidar: returns >= (object.family === 'person' ? 1 : 2)};
}

// A travelling object was carried along its course and nothing returns from where that put it.
// Look back along the course, toward where it was last measured, for returns on its footprint.
function stoppedAt(object, ctx, cos, sin, requireDoppler=false) {
  const local = (object.heading - ctx.degrees) * rad;
  for (const share of [.75, .5, .25, 0]) {
    const x = object.raw.x + (object.x - object.raw.x) * share, z = object.raw.z + (object.z - object.raw.z) * share;
    const dx = x - ctx.x, dz = z - ctx.z;
    const fx=cos*dx+sin*dz, fy=-sin*dx+cos*dz;
    if(evidence(object,fx,fy,local,ctx.frame).lidar &&
       (!requireDoppler || stationaryRays(object,ctx,fx,fy,local,cos,sin)))return {x,z};
  }
  return null;
}

function leave(object, why) {
  if (!object.leaving) object.why = why;
  object.leaving = true;
}

const MASTS={radar:[2,0,.9],radar_front_left:[1.85,-.85,.82],radar_front_right:[1.85,.85,.82],
  radar_rear_left:[-2.05,-.85,.82],radar_rear_right:[-2.05,.85,.82]};
const motorVehicle=label=>label==='car'||label==='truck'||label==='bus';

function correctedRay(ctx,sensor,p){
  const mount=MASTS[sensor],front=sensor==='radar';
  const dx=p[0]-mount[0],dy=p[1]-mount[1],dz=front?0:p[2]-mount[2];
  const range=Math.hypot(dx,dy,dz);
  if(range<2)return null;
  const nx=dx/range,ny=dy/range;
  const speed=p[front?2:3]+(ctx.forwardV-ctx.yawRate*mount[1])*nx+(ctx.rightV+ctx.yawRate*mount[0])*ny;
  return {nx,ny,speed};
}

function stationaryRays(object,ctx,fx,fy,local,cos,sin){
  const vx=cos*object.vx+sin*object.vz,vy=cos*object.vz-sin*object.vx;
  const c=Math.cos(local),s=Math.sin(local);
  let still=0,moving=0;
  for(const sensor of Object.keys(MASTS)){
    for(const p of (sensor==='radar'?ctx.frame.radar:ctx.frame.corner_radar?.[sensor])??[]){
      const dx=p[0]-fx,dy=p[1]-fy;
      if(Math.abs(dx*c+dy*s)>object.length/2+.45||Math.abs(dy*c-dx*s)>object.width/2+.45)continue;
      const ray=correctedRay(ctx,sensor,p);
      if(!ray||Math.abs(vx*ray.nx+vy*ray.ny)<1)continue;
      if(Math.abs(ray.speed)<.5)still++;else moving++;
    }
  }
  return still>=2&&still>moving;
}

// A changing visible surface can move a fitted box several metres while the car is
// stationary. Measure motion independently on the Doppler rays assigned to the box.
// Poor side-on viewing geometry cannot resolve speed along a lane.
function laneSpeed(sample,ctx,axis){
  const {s}=sample;
  const sources=s.doppler??(s.sources?.includes('radar')&&finite(s.radial_mps)?{radar:{radial_mps:s.radial_mps}}:null);
  if(ctx.reset||axis==null||!sources||!motorVehicle(s.label))return null;
  const angle=(axis-ctx.degrees)*rad,c=Math.cos(angle),sn=Math.sin(angle);
  const speeds=[];
  for(const [sensor,hit] of Object.entries(sources)){
    const mount=MASTS[sensor];
    if(!mount||!finite(hit.radial_mps))continue;
    const front=sensor==='radar', rays=front?ctx.frame.radar:ctx.frame.corner_radar?.[sensor];
    for(const p of rays??[]){
      const v=p[front?2:3];
      if(!finite(v)||Math.abs(v-hit.radial_mps)>.65)continue;
      const x=p[0]-s.x,y=p[1]-s.y;
      if(Math.abs(x*c+y*sn)>s.length/2+.55||Math.abs(y*c-x*sn)>s.width/2+.45)continue;
      const ray=correctedRay(ctx,sensor,p);
      if(!ray)continue;
      const projection=ray.nx*c+ray.ny*sn;
      if(Math.abs(projection)>=.55)speeds.push(ray.speed/projection);
    }
  }
  if(speeds.length<3)return null;
  speeds.sort((a,b)=>a-b);
  if(speeds[Math.floor(speeds.length*.75)]-speeds[Math.floor(speeds.length*.25)]>1.3)return null;
  return speeds[Math.floor(speeds.length/2)];
}

// The visible side of a moving car can wander across its fitted centre,
// especially in a turn. Fit both world-space velocity components over several
// measurements, rejecting one-off jumps and poorly resolved direction.
function fittedCourse(seen){
  if(seen.length<4)return null;
  const origin=seen[0],end=seen.at(-1),span=end.t-origin.t;
  if(span<.75||seen.some(p=>p.sourceId!==end.sourceId))return null;
  let time=0,x=0,z=0;
  for(const p of seen){time+=p.t-origin.t;x+=p.x;z+=p.z;}
  const n=seen.length;
  time=origin.t+time/n;x/=n;z/=n;
  let tt=0,tx=0,tz=0;
  for(const p of seen){
    const dt=p.t-time;
    tt+=dt*dt;tx+=dt*(p.x-x);tz+=dt*(p.z-z);
  }
  const vx=tx/tt,vz=tz/tt,speed=Math.hypot(vx,vz);
  if(speed*span<1.6)return null;
  let along=0,across=0;
  for(const p of seen){
    const dx=p.x-x-vx*(p.t-time),dz=p.z-z-vz*(p.t-time);
    along+=((dx*vx+dz*vz)/speed)**2;
    across+=((dz*vx-dx*vz)/speed)**2;
  }
  if(Math.sqrt(along/n)>.17*speed*span||Math.sqrt(across/n)>.85)return null;
  return {vx,vz,x:x+vx*(end.t-time),z:z+vz*(end.t-time)};
}

// Fold one measurement into an object: where it is, which way it faces, how big it is.
function measure(object, sample, ctx, deps, state) {
  const {s, label, mx, mz} = sample;
  const {t, degrees, h, located} = ctx;
  const fresh = object.hits === 0;
  const surround=ctx.frame.perception_version==='surround-v1';
  const motor=surround&&motorVehicle(label);
  const sinceHit = fresh ? 0 : clamp(t - object.lastHit, 0, 1.5);
  const sinceDrawn = fresh ? 0 : clamp(t - object.t, 0, .8);

  // A depth-only update to a classified track carries its existing class.
  // Count the class vote only when a camera supplied new evidence.
  if (fresh || !Array.isArray(s.cameras) || s.cameras.length) {
    object.votes[label] = (object.votes[label] || 0) + (finite(s.confidence) ? clamp(s.confidence, .05, 1) : .05);
  }
  // The class with the most accumulated confidence names the object. It must lead clearly to take over.
  let leader = object.label;
  for (const [name, weight] of Object.entries(object.votes)) if (weight > (object.votes[leader] || 0)) leader = name;
  if (fresh || (object.votes[leader] || 0) > 1.5 * (object.votes[object.label] || 0) + .3) object.label = leader;
  // A box of another class on an established object shows it is still there. Its geometry is a
  // different reading of the object, so it does not move, turn or resize what is drawn.
  const vouchOnly = !fresh && label !== object.label;
  if (vouchOnly) return witnessed(object, sample, ctx, state);

  // Is it really going somewhere? The reported position of a stopped object wanders by a few tenths
  // of a metre as the view of it changes. Only net, straight movement of measured positions is travel.
  if (fresh) object.seen.length = 0;
  object.seen.push({t, x: mx, z: mz, sourceId:s.id});
  while (t - object.seen[0].t > TRAVEL_SECONDS) object.seen.shift();
  const first = object.seen[0], latest = object.seen.at(-1);
  let travelling = object.travelling;
  if (latest.t - first.t >= .4 && object.seen.length >= 3) {
    let path = 0, stride = 0;
    for (let i = 1; i < object.seen.length; i++) {
      const step = Math.hypot(object.seen[i].x - object.seen[i - 1].x, object.seen[i].z - object.seen[i - 1].z);
      path += step; stride = Math.max(stride, step);
    }
    const net = Math.hypot(latest.x - first.x, latest.z - first.z);
    const settled = object.stillFor >= SETTLED_S;
    const need = settled ? (SETTLED_METRES[object.label] ?? 2.5) : (TRAVEL_METRES[object.label] ?? .8);
    if (object.travelling) {
      // Once travelling it stays so through noisy readings, until the window shows it has stopped.
      travelling = !(latest.t - first.t >= .6 && net < STOPPED_SHARE * need);
    } else {
      // Travel is spread over the window. One long stride is the measurement moving, not the object.
      travelling = net >= need && net >= TRAVEL_STRAIGHTNESS * path && stride <= TRAVEL_STRIDE * net;
      if (travelling) {
        // Setting off: start from the speed the window already shows.
        object.vx = (latest.x - first.x) / (latest.t - first.t);
        object.vz = (latest.z - first.z) / (latest.t - first.t);
      }
    }
  }
  // Use a locally observable lane-speed component only for longitudinal motion.
  // Sustained travel across the lane keeps its unconstrained two-dimensional course.
  const axis=located&&motorVehicle(label)?deps.laneAxis?.(mx,mz,fresh?degrees:object.heading):null;
  const along=axis==null?null:laneSpeed(sample,ctx,axis);
  const ac=Math.cos((axis??0)*rad),as=Math.sin((axis??0)*rad);
  const lateralTravel=Math.abs((latest.z-first.z)*ac-(latest.x-first.x)*as);
  const stationaryRay=!surround&&along!=null&&Math.abs(along)<.42;
  if(stationaryRay)object.legacyStaticAt=t;
  else if(along!=null&&Math.abs(along)>1.1)object.legacyStaticAt=null;
  const radialStill=!surround&&object.legacyStaticAt!=null&&t-object.legacyStaticAt<=1.0&&lateralTravel<1.2;
  if(radialStill)travelling=false;
  const stabilized=motor||radialStill;
  const corroborated=(surround||radialStill)&&along!=null&&lateralTravel<Math.max(.8,.65*Math.hypot(latest.z-first.z,latest.x-first.x));
  if(corroborated){
    const prior=object.dopplerAt==null?Infinity:t-object.dopplerAt;
    object.dopplerSpeed=prior>.6?along:object.dopplerSpeed+(along-object.dopplerSpeed)*ease(prior,.26);
    object.dopplerAt=t;
    if(Math.abs(object.dopplerSpeed)<.42)travelling=false;
    else if(Math.abs(object.dopplerSpeed)>.75)travelling=true;
  }
  if(label==='barrier')travelling=false;
  object.travelFor = travelling ? object.travelFor + sinceHit : 0;
  object.stillFor = travelling ? 0 : object.stillFor + sinceHit;
  object.travelling = travelling;

  if (fresh || sinceHit <= 0 || (!surround&&!radialStill&&Math.hypot(mx-object.x,mz-object.z)>=3.5)) {
    object.x = mx; object.z = mz;
    if (!travelling || fresh) object.vx = object.vz = 0;
  } else if (travelling) {
    // Travelling: an alpha-beta filter keeps up with it and sheds frame-to-frame jitter.
    if(corroborated){
      const velocity=object.dopplerSpeed;
      object.vx=ac*velocity;object.vz=as*velocity;
    }
    // A turning vehicle can move across the mapped lane. Consistent world-space
    // motion from the same ranging track resolves velocity on both axes.
    const fitted=motor && object.classPeak>=.6 && object.sourceId===s.id ? fittedCourse(object.seen) : null;
    if(fitted && !corroborated){
      const gain=ease(sinceHit,.18);
      object.vx+=gain*(fitted.vx-object.vx);
      object.vz+=gain*(fitted.vz-object.vz);
    }
    if(fitted)object.lastCourse={t,speed:Math.hypot(fitted.vx,fitted.vz)};
    const px = object.x + object.vx * sinceDrawn, pz = object.z + object.vz * sinceDrawn;
    const rx = (fitted?.x??mx) - px, rz = (fitted?.z??mz) - pz;
    // A depth/identity handoff is an uncertain correction, never an instantaneous jump.
    const correction=motor?Math.min(1,Math.max(.10,(fitted?4:1.8)*sinceDrawn)/(Math.hypot(rx,rz)||1)):1;
    object.x = px + TRACK_ALPHA * correction*rx; object.z = pz + TRACK_ALPHA * correction*rz;
    if(!corroborated&&!fitted){
      object.vx += TRACK_BETA * correction*rx / sinceHit; object.vz += TRACK_BETA * correction*rz / sinceHit;
    }
  } else {
    // Staying put: drift only slowly toward where it is measured.
    const residual=Math.hypot(mx-object.x,mz-object.z);
    const follow = stabilized ? (residual<.15?0:Math.min(ease(sinceHit,STILL_SECONDS),.45*sinceDrawn/(residual||1))) : ease(sinceHit, STILL_SECONDS);
    object.x += (mx - object.x) * follow; object.z += (mz - object.z) * follow;
    object.vx = object.vz = 0;
  }

  // Heading is an estimate with a front/back ambiguity, and in recordings it can sit far off the
  // road's direction. A vehicle on or beside a mapped lane is drawn along that lane; its own travel
  // decides which way it faces and overrides the lane when it is crossing or turning.
  let facing = degrees + (finite(s.heading) ? s.heading : 0);
  const speed = Math.hypot(object.vx, object.vz);
  const underway = !fresh && speed > 1.5 && s.speed_mps > 1.5;
  const course = Math.atan2(object.vz, object.vx) / rad;
  if (located && deps.laneAxis && WHEELED.has(object.label)) {
    // A vehicle first seen in a lane faces the way that lane is driven, where the map says.
    const lane = deps.laneAxis(object.x, object.z, underway ? course : fresh ? facing : object.heading, fresh);
    if (lane !== null) {
      // One noisy velocity sample is not a turn: travel must stay off the lane for half a second.
      object.across = underway && Math.abs(turn(course, lane)) > 35 ? object.across + sinceHit : 0;
      facing = object.across >= .5 ? course : lane;
    }
  } else if (object.label === 'person' && !fresh) {
    // A pedestrian turns to face the way they have been walking, and otherwise stays as drawn.
    object.across = speed > .6 && s.speed_mps > .5 ? object.across + sinceHit : 0;
    facing = object.across >= .4 ? course : object.heading;
  }
  // Keep the facing the shape already has. Swap ends only after plain travel the other way.
  if (!fresh && Math.abs(turn(object.heading, facing)) > 90) facing += 180;
  object.against = underway && Math.abs(turn(course, object.heading)) > 110 ? object.against + sinceHit : 0;
  object.turned = fresh;
  if(object.against>=.5){
    if(!surround){object.heading+=180;facing+=180;object.against=0;object.turned=true;}
    else facing=course;
  }
  const rotation=turn(object.heading,facing)*ease(sinceHit,HEADING_SECONDS);
  const maxTurn=Math.min(70,18+7*speed)*sinceDrawn;
  object.heading = fresh ? facing : object.heading + (motor?(travelling?clamp(rotation,-maxTurn,maxTurn):0):rotation);

  // Length, width and height are estimates too; ease them so the shape does not pump.
  const settle = fresh ? 1 : ease(motor?sinceDrawn:sinceHit,motor?1.1:SHAPE_SECONDS);
  object.length += (s.length - object.length) * settle;
  object.width += (s.width - object.width) * settle;
  object.height += (s.height - object.height) * settle;
  // Track Z is the estimated centre. The shape stands on its measured base.
  const measuredBase=h+s.z-s.height/2;
  const surface=(deps.groundAt??deps.laneAxis?.ground)?.(object.x,object.z,measuredBase);
  object.onPavement=surface!=null&&finite(surface);
  if(object.onPavement)object.y=surface+.015;
  else if(fresh)object.y=measuredBase;
  else object.y+=clamp(measuredBase-object.y,-.4*sinceDrawn,.4*sinceDrawn)*ease(sinceHit,.28);

  object.raw = {x: mx, z: mz};
  object.rest = null;
  object.speed = finite(s.speed_mps) ? s.speed_mps : null;
  witnessed(object, sample, ctx, state);
}

// The object was measured in this frame.
function witnessed(object, sample, ctx, state) {
  const {s, range} = sample;
  if(object.sourceId!==s.id)object.mounted=null;
  if(sample.mounted){
    object.mounted={t:ctx.t-s.cycle_association.age_s,length:s.length,width:s.width,height:s.height};
  }
  object.sourceId = s.id;
  object.support=s.lidar_hits??0;
  if (s.label === 'barrier' && ['striped_board', 'chain'].includes(s.barrier_subtype)) {
    object.barrierSubtype = s.barrier_subtype;
    object.barrierSubtypeConfidence = s.subtype_confidence;
  }
  object.boxes=s.rgb_boxes;
  object.classReadings=(s.cameras??[]).flatMap(view=>{
    const box=ctx.frame.camera_detections?.[view]?.find(d=>d.track_id===s.id&&d.range_supported);
    if(box?.label===object.label && motorVehicle(box.label)){
      const history=state.images.get(view+':'+box.image_id+':'+box.label);
      object.classPeak=Math.max(object.classPeak??0,history?.peak??box.score);
      (object.classViews??=new Set()).add(view);
    }
    return box&&motorVehicle(box.label)?[[box.label,box.score]]:[];
  });
  for(const [label,score] of object.classReadings){
    if(label!==object.label)continue;
    object.classCount=(object.classCount??0)+1;
    object.classPeak=Math.max(object.classPeak??0,score);
  }
  if(motorVehicle(object.label)&&Array.isArray(s.surface_xyz)&&s.surface_xyz.length>=3){
    // A stationary object's exposed faces accumulate in world space. Moving
    // objects retain only the current scan so their past positions cannot elongate them.
    const moving=ctx.t-object.dopplerAt<.65&&Math.abs(object.dopplerSpeed)>.75;
    const history=object.surfaces??=[];
    if(moving||object.surfaceMoving)history.length=0;
    object.surfaceMoving=moving;
    while(history.length&&(ctx.t-history[0].t>8||history.length>=40))history.shift();
    if(!history.length||ctx.t-history.at(-1).t>=.18){
      const yaw=ctx.degrees*rad,c=Math.cos(yaw),n=Math.sin(yaw);
      const points=s.surface_xyz.filter(p=>Array.isArray(p)&&p.length===3&&p.every(finite)).map(([x,y,h])=>
        [ctx.x+x*c-y*n,ctx.z+x*n+y*c,ctx.h+h]);
      history.push({t:ctx.t,x:ctx.x,z:ctx.z,base:ctx.h+s.z-s.height/2,points});
      object.surfaceCells=new Set(history.flatMap(scan=>scan.points.map(p=>Math.floor(p[0]/.55)+','+Math.floor(p[1]/.55))));
    }
    // Repeated depth patches with substantial horizontal and vertical extent
    // can sustain a weak camera class. Thin poles and flat strips cannot.
    if(s.label===object.label && s.lidar_hits>=6 && sample.bounds){
      const {length,width,height}=sample.bounds;
      if(length>=.65&&width>=.65&&height>=.6)object.bodyScans=(object.bodyScans??0)+1;
    }
  }
  object.hits++;
  object.lastHit = object.lastBox = object.lastLidar = ctx.t;
  object.leaving = false; object.why = null; object.pastFor = 0;
  object.peak = Math.max(object.peak, finite(s.confidence) ? s.confidence : 0);
  if (sample.joining) object.lastHit = ctx.t - (finite(s.age_s) ? clamp(s.age_s, 0, 1) : GRACE_S + .05);
  // Two measurements confirm an object the detector was fairly sure of, three one it was not:
  // boxes near the detector's threshold, twice in a row, were bollards. Joining a drive part-way,
  // the tracker's own second measurement stands in; it reports a speed only from then on.
  // A near pedestrian with a camera class, lidar depth and independent Doppler
  // deserves an immediate symbol, even if this is its first camera box.
  const fusedPerson = s.label === 'person' && ['rgb','lidar','radar'].every(source => s.sources?.includes(source));
  // A resolved camera class supported by a lidar surface and actual Doppler can
  // localize a pedestrian beyond the range at which a class score alone is trusted.
  const repeatedPerson = fusedPerson && s.cameras?.some(camera=>{
    const linked=ctx.frame.camera_detections?.[camera]?.find(b=>b.track_id===s.id&&b.range_supported&&b.label==='person');
    const history=linked&&state.images.get(camera+':'+linked.image_id+':person');
    return history?.count>=3 && history.peak>=SURE.confidence;
  });
  const resolvedPerson = fusedPerson && (s.confidence >= SURE.confidence || repeatedPerson) && s.lidar_hits >= 4 &&
    s.cameras?.length > 0 && finite(s.radial_mps);
  const immediatePerson = resolvedPerson || (fusedPerson && range <= 12 && s.confidence >= .35);
  const sure = (s.confidence >= SURE.confidence && range <= SURE.range) || immediatePerson;
  const joined = state.warm < GRACE_S && finite(s.speed_mps);
  if (sure || joined || object.hits >= 3 || (object.hits >= 2 && object.peak >= FAIR)) {
    if (!object.confirmed && immediatePerson) object.alpha = .72;
    if (!object.confirmed && state.warm < GRACE_S) object.alpha = 1 - 1e-9;   // joining mid-drive: show what is tracked at once
    object.confirmed = true;
  }
  state.ids.set(s.id, object.key);
}

// Cameras can fit different faces of one parked vehicle under separate tracker
// IDs. Suppress only overlapping, aligned body centres, and keep independent boxes
// from a common camera distinct. The sidebar can inspect either contributing ID.
function sameFootprint(a,b,state,t,wasTogether) {
  if(!motorVehicle(a.label)||!motorVehicle(b.label)||Math.abs(a.y-b.y)>.65)return false;
  const angle=a.heading*rad,dx=b.x-a.x,dz=b.z-a.z;
  const lengthwise=Math.abs(dx*Math.cos(angle)+dz*Math.sin(angle)),across=Math.abs(dz*Math.cos(angle)-dx*Math.sin(angle));
  if(Math.abs(Math.cos((b.heading-a.heading)*rad))<.86||lengthwise>Math.max(a.length,b.length)+1||across>3.2)return false;
  const x=state.objects.get(a.key),y=state.objects.get(b.key);
  if(t-x.lastHit<.12&&t-y.lastHit<.12){
    for(const [view,p] of Object.entries(x.boxes??{})){
      const q=y.boxes?.[view];
      if(q && (p[2]<=q[0]||q[2]<=p[0]||p[3]<=q[1]||q[3]<=p[1]))return false;
    }
  }
  if(lengthwise<=Math.min(3.8,.84*Math.min(a.length,b.length))&&across<=Math.min(1,.55*Math.min(a.width,b.width)))return true;
  if(wasTogether&&(a.held||b.held)&&lengthwise<Math.min(a.length,b.length)&&across<1.7)return true;
  if((t-x.lastHit<=.12&&b.held&&Math.hypot(x.raw.x-b.x,x.raw.z-b.z)<1.25)||
     (t-y.lastHit<=.12&&a.held&&Math.hypot(y.raw.x-a.x,y.raw.z-a.z)<1.25))return true;
  if(x.surfaceMoving||y.surfaceMoving)return sameMovingSurface(x,y,t);
  // Partially visible front and rear fits can have very different inferred
  // centres while repeatedly hitting the very same patches of bodywork.
  if(!x.surfaceCells||!y.surfaceCells)return false;
  let common=0,low=Infinity,high=-Infinity;
  const [small,large]=x.surfaceCells.size<y.surfaceCells.size?[x.surfaceCells,y.surfaceCells]:[y.surfaceCells,x.surfaceCells];
  for(const key of small)if(large.has(key)){
    const [x,z]=key.split(',').map(Number),station=.55*(x*Math.cos(angle)+z*Math.sin(angle));
    common++;low=Math.min(low,station);high=Math.max(high,station);
    if(common>=4&&high-low>.8)return true;
  }
  return false;
}

function sameMovingSurface(a,b,t){
  const x=a.surfaces?.at(-1),y=b.surfaces?.at(-1);
  if(!x||!y||Math.min(x.points.length,y.points.length)<4||
     t-Math.min(x.t,y.t)>.85||Math.abs(x.t-y.t)>.55||Math.hypot(a.vx-b.vx,a.vz-b.vz)>1.6)return false;
  // Compare the actual surfaces at one exposure time. A different class prior
  // can displace the fitted centre without moving any of the lidar returns.
  const dt=y.t-x.t;
  const dx=(a.vx+b.vx)*.5*dt,dz=(a.vz+b.vz)*.5*dt;
  const hits=new Set(),otherHits=new Set();
  for(const p of x.points)for(const q of y.points){
    if((p[0]+dx-q[0])**2+(p[1]+dz-q[1])**2+(p[2]-q[2])**2<.55**2){hits.add(p);otherHits.add(q);}
  }
  if(hits.size<Math.max(4,.60*x.points.length)||otherHits.size<Math.max(4,.60*y.points.length))return false;
  // Require overlap along a patch of bodywork, rather than one corner or bumper tip.
  for(const p of hits)for(const q of hits)if(Math.hypot(p[0]-q[0],p[1]-q[1])>1)return true;
  return false;
}

function stationaryEnvelope(group,state,heading){
  const scans=[];
  for(const entry of group){
    const o=state.objects.get(entry.key);
    if(o.surfaceMoving)return null;
    if(o.surfaces)scans.push(...o.surfaces);
  }
  if(scans.length<4)return null;
  const c=Math.cos(heading*rad),s=Math.sin(heading*rad),along=[],across=[],heights=[];
  let viewerMin=Infinity,viewerMax=-Infinity;
  for(const scan of scans){
    const viewer=c*scan.x+s*scan.z;
    viewerMin=Math.min(viewerMin,viewer);viewerMax=Math.max(viewerMax,viewer);
    for(const [x,z,y] of scan.points){along.push(c*x+s*z);across.push(c*z-s*x);heights.push(y);}
  }
  if(along.length<24)return null;
  for(const values of [along,across,heights])values.sort((a,b)=>a-b);
  const last=along.length-1,lo=Math.floor(last*.01),hi=Math.ceil(last*.99);
  const l=along[hi]-along[lo],w=across[hi]-across[lo];
  if(viewerMin>=along[lo]||viewerMax<=along[hi]||l<2.5||l>15||w<1||w>3.6)return null;
  const mid=(along[lo]+along[hi])/2, side=(across[lo]+across[hi])/2;
  const bases=scans.map(scan=>scan.base).sort((a,b)=>a-b);
  return {x:c*mid-s*side,z:s*mid+c*side,length:l,width:w,roof:heights[hi],base:bases[Math.floor(bases.length*.10)]};
}

function stillOnBody(object,envelope,heading,t){
  if(!envelope||object.surfaceMoving||t-object.lastHit>.8||t-envelope.t>4)return false;
  const points=object.surfaces?.at(-1)?.points??[],c=Math.cos(heading*rad),s=Math.sin(heading*rad);
  if(points.length<3)return false;
  let supported=0;
  for(const [x,z] of points){
    const dx=x-envelope.x,dz=z-envelope.z;
    if(Math.abs(dx*c+dz*s)<=envelope.length/2+.4&&Math.abs(dz*c-dx*s)<=envelope.width/2+.4)supported++;
  }
  return supported>=.75*points.length;
}

function distinctVehicles(drawn,state,ctx){
  const {t,dt}=ctx;
  const prior=ctx.reset?new Map():state.presented??new Map();
  state.presented=new Map();
  const priorKeys=new Map();
  for(const [key,old] of prior)for(const member of old.members)priorKeys.set(member,key);
  const remaining=new Set(drawn),result=[];
  for(const a of drawn){
    if(!remaining.delete(a))continue;
    const group=[a];
    for(const b of remaining){
      const previous=priorKeys.get(a.key);
      if(sameFootprint(a,b,state,t,previous!=null&&previous===priorKeys.get(b.key))){group.push(b);remaining.delete(b);}
    }
    if(!motorVehicle(a.label)){result.push(a);continue;}
    const weight=p=>{
      const o=state.objects.get(p.key);
      return (1+Math.log1p(o.support))*Math.exp(-3*(t-o.lastHit));
    };
    const best=group.reduce((winner,p)=>weight(p)>weight(winner)?p:winner);
    const members=group.map(p=>p.key);
    let key=priorKeys.get(a.key)??a.key;
    if(state.presented.has(key))key=members.find(k=>!state.presented.has(k))??a.key;
    const out={...best,key,sourceIds:group.length>1?group.map(p=>p.sourceId):undefined};
    const old=prior.get(out.key);
    // The selected surface may switch from the front fit to a new rear-camera ID.
    // Its first class vote cannot replace the established body's dimensions. Only
    // fresh, linked camera classifications vote, with older votes decaying over time.
    const votes={};
    for(const [label,weight] of Object.entries(old?.votes??{}))votes[label]=weight*Math.exp(-dt/8);
    for(const member of group){
      const o=state.objects.get(member.key);
      if(o.lastHit===t)for(const [label,score] of o.classReadings??[])votes[label]=(votes[label]??0)+score;
    }
    if(old&&best.label!==old.pose.label&&(old.owner!==best.key||old.retainingLabel||group.length>1)){
      const priorLabel=old.pose.label;
      // Stabilize the owner's own settled class at a handoff. Votes are only a
      // check on a requested reclassification, never a source of a third label.
      out.label=(votes[best.label]??0)>1.35*(votes[priorLabel]??0)+1?best.label:priorLabel;
    }
    out.opacity=group.reduce((opacity,p)=>Math.max(opacity,p.opacity),out.opacity);
    let envelope=stationaryEnvelope(group,state,out.heading);
    if(envelope)envelope.t=t;
    else if(stillOnBody(state.objects.get(best.key),old?.envelope,out.heading,t))envelope=old.envelope;
    if(envelope){
      out.x=envelope.x;out.z=envelope.z;out.length=envelope.length;out.width=envelope.width;
      if(!state.objects.get(best.key).onPavement)out.y=envelope.base;
      out.height=Math.max(out.height,envelope.roof-out.y);
    }
    // Radius/extent differences between overlapping fits must not instantly teleport
    // or resize the same on-screen identity when a different camera wins the vote.
    if(old&&(envelope||group.length>1||old.blending||old.pose.sourceId!==out.sourceId)){
      const o=state.objects.get(best.key),p=old.pose;
      const px=p.x+(envelope?0:o.vx)*dt,pz=p.z+(envelope?0:o.vz)*dt;
      const dx=out.x-px,dz=out.z-pz,gain=Math.min(1,.9*dt/(Math.hypot(dx,dz)||1));
      out.x=px+gain*dx;out.z=pz+gain*dz;
      for(const field of ['length','width','height','y'])out[field]=toward(p[field],out[field],.65*dt);
      out.heading=p.heading+clamp(turn(p.heading,best.heading),-40*dt,40*dt);
    }
    const blending=group.length>1||Math.hypot(out.x-best.x,out.z-best.z)>.03||
      ['length','width','height','y'].some(field=>Math.abs(out[field]-best[field])>.03);
    state.presented.set(out.key,{pose:out,blending,members,envelope,votes,owner:best.key,retainingLabel:out.label!==best.label});
    result.push(out);
  }
  return result;
}

// Advance the memory by one frame and return the objects to draw.
// `deps.laneAxis(x, z, hintDegrees, driven)` gives the lane direction at a world point, or null.
// `deps.laneDistance(x, z)` gives the metres from a world point to the nearest lane centre.
export function advance(state, ctx, deps = {}) {
  const {frame, t, x: ex, z: ez, degrees, located} = ctx;
  if (ctx.reset) {
    state.objects.clear(); state.ids.clear(); state.images.clear(); state.uncertainCars.clear(); state.warm = 0;
  }
  const dt = ctx.reset ? 0 : clamp(ctx.dt, 0, .8);
  state.warm = Math.min(30, state.warm + dt);
  // A fresh negative barrier review withdraws a previously confirmed compound.
  for(const id of frame.barrier_review?.revoked??[]){
    const key=state.ids.get(id),previous=state.objects.get(key);
    if(previous?.sourceId===id&&previous.family==='barrier')state.objects.delete(key);
  }
  const cos = Math.cos(degrees * rad), sin = Math.sin(degrees * rad);

  for(const [key,obs] of state.images)if(t-obs.time>.5)state.images.delete(key);
  for(const [id,last] of state.uncertainCars)if(t-last>1.0)state.uncertainCars.delete(id);
  for(const [camera,boxes] of Object.entries(frame.camera_detections??{})){
    for(const box of boxes??[]){
      if((box.label!=='person'&&!motorVehicle(box.label))||!finite(box.image_id)||!finite(box.score))continue;
      const key=camera+':'+box.image_id+':'+box.label, prior=state.images.get(key);
      state.images.set(key,{time:t,count:(prior?.count??0)+1,peak:Math.max(prior?.peak??0,box.score),diffuse:prior?.diffuse??false});
    }
  }

  const samples = [], listed = new Map();
  for (const track of Array.isArray(frame.tracks) ? frame.tracks : []) {
    if (!validSample(track) || listed.has(track.id)) continue;
    listed.set(track.id,track);
    const {s,bounds}=rangedReading(track,frame,state,t);
    const sources = Array.isArray(s.sources) ? s.sources : [];
    // A 2D class score cannot establish depth, and a coast is the tracker's guess, not a measurement.
    // Joining a drive part-way is the exception: what the tracker is coasting was measured a moment
    // before, in frames this memory never saw, so it is taken up as already remembered.
    const joining = s.coasting === true && state.warm < GRACE_S && !sources.length && !state.ids.has(s.id);
    if (!joining && (s.coasting === true || !(sources.includes('lidar') || sources.includes('radar')))) continue;
    // The mounted-cycle model attaches a wheel classification to the ranged
    // person's native identity. A skipped wheel exposure yields "person" again
    // on that very same track. Keep one spatial body for both observations.
    const cycle=s.cycle_association;
    const mounted=s.label==='bicycle' && cycle?.paired_observations>=3 && finite(cycle.age_s) &&
      cycle.age_s>=0 && cycle.age_s<=.21 && ['shared rider/bicycle lidar','bicycle-exclusive lidar'].includes(cycle.range_source);
    const label = mounted?'person':LABELS.has(s.label)?s.label:'unknown';
    samples.push({s,mounted,bounds,label, family: familyOf(label), range: Math.hypot(s.x, s.y), object: null, joining,
      mx: ex + cos * s.x - sin * s.y, mz: ez + sin * s.x + cos * s.y});
  }

  const expected = object => {
    const age = t - object.lastHit;
    return object.travelling ? [object.raw.x + object.vx * age, object.raw.z + object.vz * age] : [object.raw.x, object.raw.z];
  };
  const taken = new Set();
  // The tracker's own continuity comes first.
  for (const sample of samples) {
    const object = state.objects.get(state.ids.get(sample.s.id));
    if (!object || taken.has(object.key)) continue;
    const classified = object.family === 'unknown' && sample.family !== 'unknown' && sample.s.cameras?.length;
    if (object.family !== sample.family && !classified) continue;
    const [px, pz] = expected(object);
    if (Math.hypot(sample.mx - px, sample.mz - pz) <= 5) {
      if (classified) {
        object.family = sample.family;
        object.label = sample.label;
        object.votes = {};
      }
      sample.object = object; taken.add(object.key);
    }
  }
  // Then the nearest remembered object of the same family, however the tracker numbered it.
  const pairs = [];
  for (const sample of samples) {
    if (sample.object) continue;
    for (const object of state.objects.values()) {
      if (taken.has(object.key) || object.family !== sample.family) continue;
      const [px, pz] = expected(object);
      const drift = (object.travelling ? TRAVEL_DRIFT_MPS : DRIFT_MPS[object.family]) * (t - object.lastHit);
      const gate = Math.min(GATE_LIMIT_M, Math.max(GATE_M[object.family], object.length / 2 + .5) + drift);
      const distance = Math.hypot(sample.mx - px, sample.mz - pz);
      if(frame.perception_version==='surround-v1'&&motorVehicle(object.label)&&
        Math.hypot(sample.mx-object.x,sample.mz-object.z)>Math.max(3.2,.7*object.length)+
        Math.hypot(object.vx,object.vz)*Math.min(.5,t-object.lastHit))continue;
      if (distance <= gate) pairs.push([distance, sample, object]);
    }
  }
  pairs.sort((a, b) => a[0] - b[0]);
  for (const [, sample, object] of pairs) {
    if (sample.object || taken.has(object.key)) continue;
    sample.object = object; taken.add(object.key);
  }
  for (const sample of samples) {
    if (sample.object) {
      measure(sample.object, sample, ctx, deps, state);
      continue;
    }
    // A second box on an object already measured this frame is that object, not another one.
    let owner = null;
    for (const object of state.objects.values()) {
      if (taken.has(object.key) && object.family === sample.family &&
          frame.perception_version !== 'surround-v1' &&
          Math.hypot(sample.mx - object.raw.x, sample.mz - object.raw.z) <= Math.max(1, object.length / 2 + .5)) {
        owner = object;
        break;
      }
    }
    if (owner) {
      state.ids.set(sample.s.id, owner.key);
      continue;
    }
    const object = {key: state.next++, family: sample.family, label: sample.label, votes: {}, hits: 0, confirmed: false,
      leaving: false, why: null, peak: 0, pastFor: 0, alpha: 0, solid: 1, x: sample.mx, z: sample.mz, y: 0, vx: 0, vz: 0, heading: 0, against: 0, across: 0,
      length: sample.s.length, width: sample.s.width, height: sample.s.height, seen: [], travelling: false, travelFor: 0, stillFor: 0, rest: null, turned: true,
      raw: {x: sample.mx, z: sample.mz}, lastHit: t, lastBox: t, lastLidar: t, t, speed: null};
    state.objects.set(object.key, object);
    taken.add(object.key);
    measure(object, sample, ctx, deps, state);
  }

  const drawn = [];
  for (const object of state.objects.values()) {
    const unseen = t - object.lastHit;
    if (!object.confirmed) {
      if (unseen > CANDIDATE_S) state.objects.delete(object.key);
      continue;
    }
    if (unseen > 0) {
      object.turned = false;
      // A course is followed only once travel has been seen for a while.
      if (object.travelling && object.travelFor < CARRY_AFTER_S) {
        object.travelling = false; object.vx = object.vz = 0;
      }
      if (object.travelling) {
        object.x += object.vx * dt; object.z += object.vz * dt;
      } else if (object.rest) {
        // Glide back to where it came to rest.
        const gap=Math.hypot(object.rest.x-object.x,object.rest.z-object.z);
        const follow = Math.min(ease(dt, REST_SECONDS),object.lastCourse?4*dt/(gap||1):1);
        object.x += (object.rest.x - object.x) * follow; object.z += (object.rest.z - object.z) * follow;
        if (Math.hypot(object.rest.x - object.x, object.rest.z - object.z) < .05) object.rest = null;
      }
      if (unseen > GRACE_S) {
        const dx = object.x - ex, dz = object.z - ez;
        const fx = cos * dx + sin * dz, fy = -sin * dx + cos * dz;
        const seen = evidence(object, fx, fy, (object.heading - degrees) * rad, frame);
        if (seen.box) object.lastBox = t;
        if (seen.lidar) object.lastLidar = t;
        if(frame.perception_version==='surround-v1'&&object.family==='unknown'&&(!seen.lidar||unseen>.65)){
          leave(object,'surface support expired');
        }
        // Depth-only reobservations can confirm a surface on a hedge or shelter.
        // A weak, isolated camera classification cannot sustain a vehicle after
        // the assigned depth measurements also stop.
        const tentativeVehicle = motorVehicle(object.label) && object.classCount>0 &&
          (object.bodyScans??0)<3 && object.classViews?.size===1 &&
          object.classPeak<.50 && object.classCount<3;
        if(frame.perception_version==='surround-v1' && unseen>.55 &&
           (tentativeVehicle || object.family==='barrier'))leave(object,'surface support expired');
        // Something farther off, seen where the object would stand, means it is no longer there.
        object.pastFor = seen.inView && seen.past ? object.pastFor + dt : 0;
        if (object.pastFor > SEEN_PAST_S) leave(object, 'seen past');
        else if (object.travelling) {
          if (unseen > CARRY_S / 2 && !seen.lidar) {
            // Nothing returns from where its course has taken it. Returns from a point it has
            // already passed mean it stopped there; none at all, a second on, mean it has gone.
            // Scattered XY returns behind a recently resolved moving car do
            // not justify reversing it down its last measured trajectory.
            const resolvedTravel=object.lastCourse?.speed>3 && object.lastHit-object.lastCourse.t<.4;
            const stop = stoppedAt(object,ctx,cos,sin,resolvedTravel);
            if (stop) {
              object.rest = stop; object.travelling = false; object.vx = object.vz = 0; object.stillFor = 0;
              object.lastLidar = t;
            } else if (unseen > CARRY_S) leave(object, 'course ran out');
          } else if (unseen > CARRY_LIMIT_S) leave(object, 'course ran out');   // returns keep pace with it for a while
        } else if (seen.inView) {
          const near = Math.hypot(fx, fy) < NEAR_M && object.family !== 'person';
          const patience = UNSEEN_IN_VIEW_S + (t - object.lastLidar <= .5 ? LIDAR_EXTENDS_S[object.family] : 0);
          // Close by, a vehicle returns many lidar points. With none, and no box either, it has gone.
          if (near && t - object.lastBox > NEAR_SILENT_S && t - object.lastLidar > NEAR_SILENT_S) leave(object, 'no returns');
          else if (t - object.lastBox > patience) leave(object, 'no longer detected');
        } else if (object.family === 'person') {
          if (unseen > PERSON_BESIDE_S) leave(object, 'out of view');
        } else if (t - object.lastLidar > (Math.hypot(fx, fy) < NEAR_M ? BESIDE_S.near : BESIDE_S.far)) {
          leave(object, 'no returns');
        }
      }
    }
    const dx = object.x - ex, dz = object.z - ez;
    const fx = cos * dx + sin * dz, fy = -sin * dx + cos * dz;
    // The vehicle is where the object was remembered, or has left it behind.
    const overrun = unseen > GRACE_S && Math.abs(fx) < EGO_HALF.length && Math.abs(fy) < EGO_HALF.width;
    if (overrun) leave(object, 'driven through');
    else if (fx < -BEHIND_M && frame.perception_version !== 'surround-v1') leave(object, 'passed');
    else if (frame.perception_version === 'surround-v1' && unseen > CARRY_LIMIT_S) leave(object, 'depth support expired');
    else if (Math.hypot(fx, fy) > RANGE_M) leave(object, 'out of range');

    const held = unseen > GRACE_S;
    const offRoad = frame.perception_version !== 'surround-v1' && object.family === 'wheeled' && deps.laneDistance ? deps.laneDistance(object.x, object.z) > OFF_ROAD_M : false;
    const shown = !object.leaving && !offRoad;
    const fadeOut=overrun?.15:object.why==='surface support expired'?.25:FADE_OUT_S;
    object.alpha = toward(object.alpha, shown ? 1 : 0, dt / (shown ? FADE_IN_S : fadeOut));
    object.solid = toward(object.solid, held ? REMEMBERED : 1, dt / DIM_S);
    object.t = t;
    if (object.leaving && object.alpha <= 0) {
      state.objects.delete(object.key);
      continue;
    }
    if (object.alpha <= 0) continue;
    const speed = object.speed;
    if(object.mounted){
      // Use the same two-second semantic horizon and current/recent lidar
      // continuity required for an optical rider association. The source's
      // class is still available in the raw measurement inspector.
      const source=listed.get(object.sourceId);
      const validSource=source && (source.label==='person'||source.label==='bicycle') &&
        (source.coasting?source.age_s<=.20001:source.sources?.includes('lidar') && source.lidar_hits>=3);
      if(!validSource||t-object.mounted.t>2.00001)object.mounted=null;
    }
    drawn.push({key: object.key, sourceId: object.sourceId, label: object.label, family: object.family, x: object.x, y: object.y, z: object.z,
      heading: object.heading, length: object.length, width: object.width, height: object.height,
      ...(object.mounted?{label:'bicycle',family:'wheeled',mountedRider:true,
        length:object.mounted.length,width:object.mounted.width,height:object.mounted.height}:{}),
      ...(object.label === 'barrier' && object.barrierSubtype ? {barrierSubtype:object.barrierSubtype,
        barrierSubtypeConfidence:object.barrierSubtypeConfidence} : {}),
      opacity: object.alpha * object.solid, held, leaving: object.leaving, why: object.why, turned: object.turned,
      moving: located && !held && object.travelling && finite(speed) && speed > (object.label === 'person' ? .30 : .55),
      speed});
  }
  for (const [id, key] of state.ids) if (!state.objects.has(key)) state.ids.delete(id);
  return frame.perception_version==='surround-v1'?distinctVehicles(drawn,state,ctx):drawn;
}

// The page's memory. Frames may arrive in any order: a frame reached again, or reached with
// less history than before, shows what continuous playback showed there.
export class ObjectMemory {
  constructor(deps = {}) {
    this.deps = deps;
    this.reset();
  }
  reset() {
    this.state = createState();
    this.cache = new WeakMap();
    this.ctx = null;
  }
  step(frame, heightAt, clock) {
    const ctx = contextOf(frame, this.ctx, heightAt, clock);
    const cached = this.cache.get(frame);
    const history = ctx.reset ? 0 : Math.min(30, this.state.warm + clamp(ctx.dt, 0, .8));
    let objects;
    if (cached && cached.warm >= history) {
      this.state = structuredClone(cached.state);
      objects = cached.objects;
    } else {
      objects = advance(this.state, ctx, this.deps);
      this.cache.set(frame, {warm: this.state.warm, state: structuredClone(this.state), objects});
    }
    this.ctx = ctx;
    return {ctx, objects};
  }
}

// Replay the same sensor observations whether a rider scrubs or plays straight through.
// Checkpoints keep a long drive seekable without retaining a full tracker copy at every frame.
export class ReplayMemory {
  constructor(deps) {
    this.deps = deps;
  }
  at(frames, index, heightAt) {
    if (frames !== this.frames) {
      this.frames = frames;
      this.checkpoints = [];
      this.recent = new Map();
      this.index = -1;
      this.ctx = null;
      this.state = createState();
    }
    if (this.recent.has(index)) return this.recent.get(index);
    const stride = 100;
    const block = Math.min(Math.floor(index / stride), this.checkpoints.length - 1);
    const checkpoint = this.checkpoints[block];
    if (index <= this.index || (checkpoint && this.index < block * stride - 1)) {
      this.index = block * stride - 1;
      this.ctx = checkpoint.ctx;
      this.state = structuredClone(checkpoint.state);
      this.recent.clear();
    }
    while (this.index < index) {
      const i = ++this.index;
      if (i % stride === 0 && !this.checkpoints[i / stride]) {
        this.checkpoints[i / stride] = {ctx: this.ctx, state: structuredClone(this.state)};
      }
      this.ctx = contextOf(frames[i], this.ctx, heightAt, i / 10);
      const objects = advance(this.state, this.ctx, this.deps);
      this.recent.set(i, {ctx: this.ctx, objects});
      if (this.recent.size > 2) this.recent.delete(this.recent.keys().next().value);
    }
    return this.recent.get(index);
  }
}
