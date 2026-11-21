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
const LABELS = new Set([...WHEELED, 'person']);
// Classes the detector confuses with one another are one family for matching.
export const familyOf = label => label === 'person' ? 'person' : WHEELED.has(label) ? 'wheeled' : 'unknown';

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
const GATE_M = {person: 1.2, wheeled: 2.4, unknown: 1.5};
const DRIFT_MPS = {person: 2, wheeled: 1.5, unknown: 1.5};   // how far an unseen object may have moved, per second
const TRAVEL_DRIFT_MPS = 8;
const GATE_LIMIT_M = 6;
const CARRY_S = 1;              // a travelling object continues this long without a measurement
const CARRY_AFTER_S = .4;       // once it has been seen travelling for this long
const CARRY_LIMIT_S = 3;        // and no longer than this, even with lidar returns keeping pace
const UNSEEN_IN_VIEW_S = 2;     // held in clear view with no detection over it
const SEEN_SHARE = .6;          // a box narrower than this share of the object's expected width is not it
const SEEN_PAST_S = .4;         // something farther off seen where it would stand, for this long: it has gone
const LIDAR_EXTENDS_S = {wheeled: 8, person: 2, unknown: 2};   // returns on its footprint extend that
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
      cells.get(key).push([a[0], a[1], b[0], b[1], degrees, way === null ? null : degrees + way]);
    }
  });
  // Every lane segment within reach of a point, as [distance, direction, driven direction].
  const around = (x, z) => {
    const cx = Math.floor(x / size), cz = Math.floor(z / size), found = [];
    for (let i = -1; i <= 1; i++) for (let j = -1; j <= 1; j++) for (const lane of cells.get((cx + i) + ',' + (cz + j)) || []) {
      const dx = lane[2] - lane[0], dz = lane[3] - lane[1];
      const along = clamp(((x - lane[0]) * dx + (z - lane[1]) * dz) / (dx * dx + dz * dz || 1), 0, 1);
      const distance = Math.hypot(lane[0] + along * dx - x, lane[1] + along * dz - z);
      if (distance <= LANE_REACH) found.push([distance, lane[4], lane[5]]);
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
  return {frame, t, dt, x, z, degrees, h, located, reset};
}

export function createState() {
  return {objects: new Map(), ids: new Map(), next: 1, warm: 0};
}

function validSample(sample) {
  return sample?.id != null && [sample.x, sample.y, sample.z, sample.length, sample.width, sample.height].every(finite) &&
    sample.length > 0 && sample.width > 0 && sample.height > 0;
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
function stoppedAt(object, ctx, cos, sin) {
  const local = (object.heading - ctx.degrees) * rad;
  for (const share of [.75, .5, .25, 0]) {
    const x = object.raw.x + (object.x - object.raw.x) * share, z = object.raw.z + (object.z - object.raw.z) * share;
    const dx = x - ctx.x, dz = z - ctx.z;
    if (evidence(object, cos * dx + sin * dz, -sin * dx + cos * dz, local, ctx.frame).lidar) return {x, z};
  }
  return null;
}

function leave(object, why) {
  if (!object.leaving) object.why = why;
  object.leaving = true;
}

// Fold one measurement into an object: where it is, which way it faces, how big it is.
function measure(object, sample, ctx, deps, state) {
  const {s, label, mx, mz} = sample;
  const {t, degrees, h, located} = ctx;
  const fresh = object.hits === 0;
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
  object.seen.push({t, x: mx, z: mz});
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
  object.travelFor = travelling ? object.travelFor + sinceHit : 0;
  object.stillFor = travelling ? 0 : object.stillFor + sinceHit;
  object.travelling = travelling;

  if (fresh || sinceHit <= 0 || Math.hypot(mx - object.x, mz - object.z) >= 3.5) {
    object.x = mx; object.z = mz;
    if (!travelling || fresh) object.vx = object.vz = 0;
  } else if (travelling) {
    // Travelling: an alpha-beta filter keeps up with it and sheds frame-to-frame jitter.
    const px = object.x + object.vx * sinceDrawn, pz = object.z + object.vz * sinceDrawn;
    const rx = mx - px, rz = mz - pz;
    object.x = px + TRACK_ALPHA * rx; object.z = pz + TRACK_ALPHA * rz;
    object.vx += TRACK_BETA * rx / sinceHit; object.vz += TRACK_BETA * rz / sinceHit;
  } else {
    // Staying put: drift only slowly toward where it is measured.
    const follow = ease(sinceHit, STILL_SECONDS);
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
  if (object.against >= .5) {
    object.heading += 180; facing += 180; object.against = 0; object.turned = true;
  }
  object.heading = fresh ? facing : object.heading + turn(object.heading, facing) * ease(sinceHit, HEADING_SECONDS);

  // Length, width and height are estimates too; ease them so the shape does not pump.
  const settle = fresh ? 1 : ease(sinceHit, SHAPE_SECONDS);
  object.length += (s.length - object.length) * settle;
  object.width += (s.width - object.width) * settle;
  object.height += (s.height - object.height) * settle;
  // Track Z is the estimated centre. The shape stands on its measured base.
  object.y = h + s.z - s.height / 2;

  object.raw = {x: mx, z: mz};
  object.rest = null;
  object.speed = finite(s.speed_mps) ? s.speed_mps : null;
  witnessed(object, sample, ctx, state);
}

// The object was measured in this frame.
function witnessed(object, sample, ctx, state) {
  const {s, range} = sample;
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
  const closePerson = s.label === 'person' && range <= 12 && s.confidence >= .35 &&
    ['rgb','lidar','radar'].every(source => s.sources?.includes(source));
  const sure = (s.confidence >= SURE.confidence && range <= SURE.range) || closePerson;
  const joined = state.warm < GRACE_S && finite(s.speed_mps);
  if (sure || joined || object.hits >= 3 || (object.hits >= 2 && object.peak >= FAIR)) {
    if (!object.confirmed && closePerson) object.alpha = .72;
    if (!object.confirmed && state.warm < GRACE_S) object.alpha = 1 - 1e-9;   // joining mid-drive: show what is tracked at once
    object.confirmed = true;
  }
  state.ids.set(s.id, object.key);
}

// Advance the memory by one frame and return the objects to draw.
// `deps.laneAxis(x, z, hintDegrees, driven)` gives the lane direction at a world point, or null.
// `deps.laneDistance(x, z)` gives the metres from a world point to the nearest lane centre.
export function advance(state, ctx, deps = {}) {
  const {frame, t, x: ex, z: ez, degrees, located} = ctx;
  if (ctx.reset) {
    state.objects.clear(); state.ids.clear(); state.warm = 0;
  }
  const dt = ctx.reset ? 0 : clamp(ctx.dt, 0, .8);
  state.warm = Math.min(30, state.warm + dt);
  const cos = Math.cos(degrees * rad), sin = Math.sin(degrees * rad);

  const samples = [], listed = new Set();
  for (const s of Array.isArray(frame.tracks) ? frame.tracks : []) {
    if (!validSample(s) || listed.has(s.id)) continue;
    listed.add(s.id);
    const sources = Array.isArray(s.sources) ? s.sources : [];
    // A 2D class score cannot establish depth, and a coast is the tracker's guess, not a measurement.
    // Joining a drive part-way is the exception: what the tracker is coasting was measured a moment
    // before, in frames this memory never saw, so it is taken up as already remembered.
    const joining = s.coasting === true && state.warm < GRACE_S && !sources.length && !state.ids.has(s.id);
    if (!joining && (s.coasting === true || !(sources.includes('lidar') || sources.includes('radar')))) continue;
    const label = LABELS.has(s.label) ? s.label : 'unknown';
    samples.push({s, label, family: familyOf(label), range: Math.hypot(s.x, s.y), object: null, joining,
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
        const follow = ease(dt, REST_SECONDS);
        object.x += (object.rest.x - object.x) * follow; object.z += (object.rest.z - object.z) * follow;
        if (Math.hypot(object.rest.x - object.x, object.rest.z - object.z) < .05) object.rest = null;
      }
      if (unseen > GRACE_S) {
        const dx = object.x - ex, dz = object.z - ez;
        const fx = cos * dx + sin * dz, fy = -sin * dx + cos * dz;
        const seen = evidence(object, fx, fy, (object.heading - degrees) * rad, frame);
        if (seen.box) object.lastBox = t;
        if (seen.lidar) object.lastLidar = t;
        // Something farther off, seen where the object would stand, means it is no longer there.
        object.pastFor = seen.inView && seen.past ? object.pastFor + dt : 0;
        if (object.pastFor > SEEN_PAST_S) leave(object, 'seen past');
        else if (object.travelling) {
          if (unseen > CARRY_S / 2 && !seen.lidar) {
            // Nothing returns from where its course has taken it. Returns from a point it has
            // already passed mean it stopped there; none at all, a second on, mean it has gone.
            const stop = stoppedAt(object, ctx, cos, sin);
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
    object.alpha = toward(object.alpha, shown ? 1 : 0, dt / (shown ? FADE_IN_S : overrun ? .15 : FADE_OUT_S));
    object.solid = toward(object.solid, held ? REMEMBERED : 1, dt / DIM_S);
    object.t = t;
    if (object.leaving && object.alpha <= 0) {
      state.objects.delete(object.key);
      continue;
    }
    if (object.alpha <= 0) continue;
    const speed = object.speed;
    drawn.push({key: object.key, label: object.label, family: object.family, x: object.x, y: object.y, z: object.z,
      heading: object.heading, length: object.length, width: object.width, height: object.height,
      opacity: object.alpha * object.solid, held, leaving: object.leaving, why: object.why, turned: object.turned,
      moving: located && !held && object.travelling && finite(speed) && speed > (object.label === 'person' ? .30 : .55),
      speed});
  }
  for (const [id, key] of state.ids) if (!state.objects.has(key)) state.ids.delete(id);
  return drawn;
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
