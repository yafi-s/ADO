// Join independently scored front-camera matches to the actual replay's displayed objects.
import fs from 'node:fs';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {createInterface} from 'node:readline';
import {parseArgs} from 'node:util';
import {barrierReviewer} from '../web/barrier-review.mjs';
import {coneReviewer} from '../web/cone-observations.mjs';
import {advance, contextOf, createState, laneAxes} from '../web/object-memory.mjs';

const {positionals: [stage, candidatePath, output], values} = parseArgs({
  allowPositionals: true,
  options: {barriers: {type: 'string'}, cones: {type: 'string'}, 'physical-audit': {type: 'string'},
    probe: {type: 'string', multiple: true}}
});
if (!stage || !candidatePath || !output) throw Error('Pass the capture stage, candidate journal, and report destination.');

const read = file => JSON.parse(fs.readFileSync(file));
const capture = read(path.join(stage, 'capture.json'));
const town = read(path.join(stage, 'raw/town.json'));
const features = read(new URL(`../web/media/map-features/${town.producer.map}.json`, import.meta.url));
const offsets = read(new URL('../results/perception/town03-body-offsets.json', import.meta.url));
const physical = values['physical-audit'] ? read(values['physical-audit']) : null;
if (physical && physical.capture !== capture.capture_id) throw Error('Physical calibration is for a different recording.');
Object.assign(offsets, physical?.supplementary_blueprint_calibration?.stable_blueprints);
const ends = (a, b) => [a, b].map(p => p.slice(0, 2).map(n => n.toFixed(2)).join(',')).join('|');
const routes = (town.roads ?? []).map(lane => lane.points ?? lane).filter(lane => lane.length > 1);
const directions = new Map((features.lanes ?? []).map(lane => [ends(lane.a, lane.b), lane.forward]));
const axis = laneAxes(routes, routes.map(lane => directions.get(ends(lane[0], lane.at(-1)))));
const deps = {laneAxis: axis, laneDistance: axis.distance};
const candidates = new Map(fs.readFileSync(candidatePath, 'utf8').trim().split('\n').filter(Boolean).map(line => {
  const row = JSON.parse(line);
  return [row.frame, row];
}));
const probes = new Set([...candidates.keys()].flatMap(frame => Array.from({length: 11}, (_, i) => frame + i - 5)));
const history = new Map();
const barrierReview = values.barriers ? read(values.barriers) : null;
const qualifyBarriers = barrierReviewer(barrierReview);
const coneReview = values.cones ? read(values.cones) : null;
const sidecars = [barrierReview, coneReview].filter(Boolean);
if (sidecars.some(review => review.first_frame !== capture.first_frame || review.last_frame !== capture.last_frame) ||
    coneReview && (coneReview.trial_id !== capture.trial_id || coneReview.capture_frames !== capture.summary.frames)) {
  throw Error('Sensor reviews must use the raw recording’s capture interval.');
}
const coneRows = coneReview ? new Map(coneReview.frames.map(row => [row.frame, row.cone_observations])) : null;
const qualifyCones = coneReviewer();
const state = createState();
let ctx = null, index = 0;
const frames = [];
const incomplete = [];
const counts = {vehicle_and_pedestrian: 0, two_vehicles_and_pedestrian: 0, two_of_each: 0};

function witnessed(frame, objects, tracks, hit) {
  const record = {actor_id: hit.actor_id, track_id: hit.track_id, actor_type: hit.actor_type};
  const trackRank = tracks.findIndex(t => t.id === hit.track_id);
  if (trackRank < 0 || !frame.camera_detections?.front?.some(b => b.image_id === hit.image_id &&
      b.track_id === hit.track_id && b.range_supported === true)) return {...record, reason: 'measurement_removed_in_replay_review'};
  const obj = objects.find(p => p.sourceId === hit.track_id && p.label === hit.class_label);
  if (!obj) return {...record, reason: 'no_drawn_object_using_source_id_and_class'};
  if (obj.held || obj.opacity < .5) {
    return {...record, reason: 'held_or_below_opacity_threshold', opacity: +obj.opacity.toFixed(3)};
  }
  const [wx, wy, yaw] = frame.observer_map.ego, radians = yaw * Math.PI / 180;
  const dx = obj.x - wx, dy = obj.z - wy;
  const x = dx * Math.cos(radians) + dy * Math.sin(radians);
  const y = -dx * Math.sin(radians) + dy * Math.cos(radians);
  const error = Math.hypot(x - hit.local_xy_m[0], y - hit.local_xy_m[1]);
  if (error > hit.center_error_gate_m) return {...record, reason: 'drawn_center_outside_gate',
    error_m: +error.toFixed(4)};
  return {...hit, screen_id: obj.key, screen_center_error_m: +error.toFixed(4),
    screen_opacity: +obj.opacity.toFixed(3), raw_range_rank: trackRank + 1};
}

function centerOf(actor) {
  const shift = actor.bbox_offset_m ?? offsets[actor.blueprint]?.offset_xyz;
  if (!shift) return null;
  const [pitch, yaw, roll] = [actor.pitch ?? 0, actor.yaw, actor.roll ?? 0].map(deg => deg * Math.PI / 180);
  const [cp, cy, cr] = [pitch, yaw, roll].map(Math.cos);
  const [sp, sy, sr] = [pitch, yaw, roll].map(Math.sin);
  const [x, y, z] = shift;
  return [actor.x + cp*cy*x + (cy*sp*sr-sy*cr)*y + (-cy*sp*cr-sy*sr)*z,
    actor.y + cp*sy*x + (sy*sp*sr+cy*cr)*y + (-sy*sp*cr+cy*sr)*z];
}

function continuity(frame, hit) {
  let previous = null, raw = 0, rawMeasured = 0, drawn = 0, drawnCurrent = 0;
  const frames = [], steps = [], sourceIds = new Set();
  for (let n = frame - 5; n <= frame + 5; n++) {
    const exposure = history.get(n);
    if (!exposure) continue;
    frames.push(n);
    const original = exposure.rawTracks.get(hit.track_id);
    const object = exposure.drawn.get(hit.screen_id);
    raw += Number(Boolean(original));
    rawMeasured += Number(Boolean(original && !original.coasting));
    if (object) sourceIds.add(object.sourceId);
    drawn += Number(Boolean(object));
    const actual = exposure.actors.get(hit.actor_id);
    const close = actual && object && object.label === hit.class_label && !object.held && object.opacity >= .5 &&
      Math.hypot(object.x - actual[0], object.z - actual[1]) <= hit.center_error_gate_m;
    drawnCurrent += Number(Boolean(close));
    if (close && previous?.frame === n - 1) {
      steps.push({ending_frame: n, residual: Math.hypot(object.x - previous.object.x - (actual[0] - previous.actual[0]),
        object.z - previous.object.z - (actual[1] - previous.actual[1]))});
    }
    previous = close ? {frame: n, object, actual} : null;
  }
  const worst = steps.reduce((best, step) => (!best || step.residual > best.residual) ? step : best, null);
  return {frame_interval: [frames[0], frames.at(-1)], exposures: frames.length, exact_raw_track_id_frames: raw,
    exact_raw_track_id_measured_frames: rawMeasured, exact_display_id_frames: drawn,
    position_qualified_current_display_frames: drawnCurrent, observed_screen_source_ids: [...sourceIds],
    matched_same_display_id_steps: steps.length, max_0_1s_translation_residual_m: worst ? +worst.residual.toFixed(4) : null,
    worst_step_ending_frame: worst?.ending_frame ?? null};
}

for await (const line of createInterface({
  input: fs.createReadStream(path.join(stage, 'raw/perception.jsonl')),
  crlfDelay: Infinity
})) {
  const raw = JSON.parse(line);
  const {ego, ego_z} = raw.observer_map;
  // The object memory and measurement review receive localization only. Target actors
  // arrive later through the independent, read-only candidate list.
  const inlet = qualifyBarriers({...raw, observer_map: {ego, ego_z}});
  const frame = qualifyCones(inlet, coneRows ? coneRows.get(raw.frame) ?? [] : inlet.cone_observations);
  ctx = contextOf(frame, ctx, () => 0, index++ / 10);
  const objects = advance(state, ctx, deps);
  if (probes.has(frame.frame)) {
    history.set(frame.frame, {rawTracks: new Map(frame.tracks.map(t => [t.id, t])),
      drawn: new Map(objects.map(p => [p.key, p])),
      actors: new Map(raw.observer_map.others.map(a => [a.actor_id, centerOf(a)]))});
  }
  const claim = candidates.get(frame.frame);
  if (!claim) continue;
  candidates.delete(frame.frame);
  const tracks = frame.tracks.filter(t => Number.isFinite(t.x) && Number.isFinite(t.y))
    .sort((a, b) => Math.hypot(a.x, a.y) - Math.hypot(b.x, b.y));
  const checks = claim.front_camera_matches.map(hit => witnessed(frame, objects, tracks, hit));
  const matches = checks.filter(hit => !hit.reason);
  const pedestrians = matches.filter(hit => hit.actor_type === 'pedestrian').length;
  const vehicles = matches.length - pedestrians;
  const misses = checks.filter(hit => hit.reason);
  if (misses.length) incomplete.push({frame: claim.frame, time_s: claim.time_s,
    optical_matches: claim.front_camera_matches.length, matched_screen_objects: matches.length, misses});
  if (!pedestrians || !vehicles) continue;
  counts.vehicle_and_pedestrian++;
  counts.two_vehicles_and_pedestrian += Number(vehicles >= 2);
  counts.two_of_each += Number(vehicles >= 2 && pedestrians >= 2);
  const moving = matches.filter(hit => hit.preceding_half_second_mps != null &&
    hit.preceding_half_second_mps >= (hit.actor_type === 'pedestrian' ? .5 : 1.));
  const eligible_default_rows = matches.filter(hit => hit.raw_range_rank <= 3).length;
  frames.push({...claim, front_camera_matches: undefined, matched_screen_objects: matches,
    vehicles, pedestrians, moving: moving.length, eligible_default_rows,
    nearest_m: Math.min(...matches.map(hit => hit.center_range_m)),
    furthest_m: Math.max(...matches.map(hit => hit.center_range_m))});
}
if (candidates.size || index !== capture.summary.frames) throw Error('Screen replay and optical audit frame counts differ.');

frames.sort((a, b) => Math.min(b.vehicles, b.pedestrians) - Math.min(a.vehicles, a.pedestrians) ||
  b.matched_screen_objects.length - a.matched_screen_objects.length || b.moving - a.moving ||
  b.eligible_default_rows - a.eligible_default_rows || a.furthest_m - b.furthest_m);
const representatives = [];
const withContinuity = frame => ({...frame, matched_screen_objects: frame.matched_screen_objects.map(hit =>
  ({...hit, local_motion_and_id: continuity(frame.frame, hit)}))});
for (const frame of frames) {
  if (representatives.every(prior => Math.abs(prior.time_s - frame.time_s) >= 4)) {
    representatives.push(withContinuity(frame));
  }
  if (representatives.length >= 10) break;
}
const requested = new Set((values.probe ?? []).map(Number));
const requestedMatches = frames.filter(frame => requested.delete(frame.frame)).map(withContinuity);
if (requested.size) throw Error('Requested frames do not meet the mixed display gate: ' + [...requested].join(', '));
const hash = file => createHash('sha256').update(fs.readFileSync(file)).digest('hex');
fs.writeFileSync(output, JSON.stringify({
  schema: 'ado-city-interaction-display-v1', capture: capture.capture_id, trial: capture.trial_id,
  frames: index, counts, representatives, all_verified_exposures: frames,
  requested_exposures: requestedMatches,
  incomplete_optical_exposures: incomplete,
  source_hashes: {candidates: hash(candidatePath),
    object_memory: hash(new URL('../web/object-memory.mjs', import.meta.url)),
    physical_audit: values['physical-audit'] ? hash(values['physical-audit']) : null,
    barrier_review: values.barriers ? hash(values.barriers) : null,
    cone_review: values.cones ? hash(values.cones) : null}
}, null, 2) + '\n');
