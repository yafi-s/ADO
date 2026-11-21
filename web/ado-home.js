// ADO home. Source selection, playback clock, telemetry and footage around the
// PerceptionScene renderer. The renderer is loaded on its own so that footage and
// telemetry keep working if the 3D scene cannot start.

import {recordedChapters} from './encounter-chapters.mjs';
import {loadReplay} from './replay-pack.mjs';
import {cornerReturns, lidarCount} from './sensor-returns.mjs';
import {cameraBoxes, cameraLamp, measuredHeads, LAMP_COLOURS} from './lamp-observations.mjs';

const CAR_URL = './vehicle-assets/waymo-ipace.glb';
const TOWN_URL = './media/town.json';
const CAMERA_LABELS = {front: 'Front', chase: 'Chase', left: 'Left', right: 'Right', rear_left: 'Rear left', rear_right: 'Rear right', infrared: 'Thermal'};
const CAMERA_ORDER = Object.keys(CAMERA_LABELS);
const LOCAL_CONSOLE = ['127.0.0.1', 'localhost', '[::1]'].includes(location.hostname);
const POLICY_LABELS = {
  fusion: 'Learned multisensor', camera: 'Learned vision',
  'ado-city-cruise-v1': 'ADO urban cruise', 'ado-city-policy-v1': 'ADO city driver', 'ado-behavior-v2': 'ADO behavior v2',
  demonstrator: 'Training instructor', teacher: 'Training instructor'
};
const OUTCOME_LABELS = {complete: 'Route complete', red_light: 'Red-light violation'};
const SOURCE_LABELS = {rgb: 'Camera', lidar: 'LiDAR', radar: 'Radar'};
const VEHICLE_LABELS = new Set(['car', 'truck', 'bus']);
const ROAD_USER_LABELS = new Set(['person', 'pedestrian', 'bicycle', 'motorcycle']);
const LAYER_TITLES = {path: 'Planned path (P)', tracks: 'Tracked objects (T)', returns: 'LiDAR and radar returns (R)'};
const CORRIDOR_HALF_WIDTH_M = 1.6;  // a track this close to the vehicle axis is shown as "ahead"
const TRACK_ROWS = 3;

const $ = id => document.getElementById(id);
const finite = Number.isFinite;
const clamp = (value, low, high) => Math.min(high, Math.max(low, value));
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;

const whole = new Intl.NumberFormat(undefined, {maximumFractionDigits: 0});
const tenth = new Intl.NumberFormat(undefined, {minimumFractionDigits: 1, maximumFractionDigits: 1});
const signedTenth = new Intl.NumberFormat(undefined, {minimumFractionDigits: 1, maximumFractionDigits: 1, signDisplay: 'exceptZero'});
const hundredth = new Intl.NumberFormat(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
const percent = new Intl.NumberFormat(undefined, {style: 'percent', maximumFractionDigits: 0});

const film = $('film');
const still = $('still');
const sourceButtons = [...document.querySelectorAll('#source-switch [data-source]')];
const viewButtons = [...document.querySelectorAll('[data-view]')];
const layerButtons = [...document.querySelectorAll('[data-layer]')];
const cameraButtons = [...document.querySelectorAll('[data-camera]')];

const state = {
  local: false,          // a local simulator API answered
  controls: false,       // /api/options confirmed that drives can be started here
  options: null,
  liveActive: false,     // a drive is running right now
  wanted: 'auto',        // 'auto' | 'live' | 'recorded'
  source: 'none',        // 'live' | 'recorded' | 'training' | 'none'
  build: null,
  catalog: [],
  entry: null,
  lastEntry: null,
  run: null,
  training: null,        // {subject, policy, logUrl}
  replay: null,          // {kind, frames, times, end, uniform}
  dir: '',
  liveRun: undefined,
  frame: null,
  camera: null,
  preferredCamera: 'chase',   // the view that shows the vehicle in its street
  cameras: [],
  view: 'follow',
  layers: {path: true, tracks: true, returns: true},
  feed: 'inset',
  feedChosen: false,
  defaultShown: false
};

// One playback clock. The video is the master whenever it can play; a timer
// stands in only for a recorded drive whose footage is missing.
const clock = {video: false, t: 0, playing: false, pendingT: 0, pendingPlay: false};

const stage = {scene: null, failed: false, faulted: false, mapUrl: TOWN_URL, mapName: null};
let loadControl;
let shownActivity = '';
let candidateActivity = '';
let trackSignature = '';
let lastTick;

// ---------- Small helpers ----------

function text(id, value, blank = '—') {
  const element = $(id);
  const content = value == null || value === '' ? blank : String(value);
  if (element.textContent !== content) element.textContent = content;
}

function show(id, visible) {
  const element = $(id);
  if (element.hidden === visible) element.hidden = !visible;
}

function attr(element, name, value) {
  if (element.getAttribute(name) !== value) element.setAttribute(name, value);
}

function note(id, message = '') {
  text(id, message, '');
  show(id, Boolean(message));
}

function feedNote(message = '') {
  note('feed-note', message);
}

function sentence(value) {
  if (typeof value !== 'string' || !value) return '';
  const label = value.replace(/[_-]+/g, ' ').replace(/([a-z])([A-Z])/g, '$1 $2').trim().toLowerCase();
  return label ? label[0].toUpperCase() + label.slice(1) : '';
}

function clockLabel(value) {
  if (!finite(value)) return '–:––';
  const seconds = Math.floor(Math.max(0, value));
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`;
}

function artifactUrl(value) {
  if (typeof value !== 'string' || !value.trim()) return null;
  try {
    const url = new URL(value, document.baseURI);
    if (url.protocol === 'https:' || (url.protocol === 'http:' && url.origin === location.origin)) return url.href;
  } catch {
    return null;
  }
  return null;
}

function previewImage(build) {
  const image = build?.preview_image;
  return artifactUrl(typeof image === 'string' ? image : image?.url);
}

function validRecordingPath(path) {
  return typeof path === 'string' && (!path || /^[a-z0-9_-]+(?:\/[a-z0-9_-]+)*$/i.test(path));
}

function sleep(milliseconds) {
  return new Promise(resolve => setTimeout(resolve, milliseconds));
}

async function getJson(url, {signal, timeout = 8000, cache = 'no-cache'} = {}) {
  const control = new AbortController();
  const cancel = () => control.abort();
  const timer = setTimeout(cancel, timeout);
  if (signal?.aborted) cancel();
  else signal?.addEventListener('abort', cancel, {once: true});
  try {
    const response = await fetch(url, {cache, signal: control.signal});
    if (!response.ok) throw Object.assign(new Error(`${url} returned ${response.status}`), {status: response.status});
    try {
      return await response.json();
    } catch (error) {
      if (control.signal.aborted) throw error;
      throw Object.assign(new Error(`${url} is not JSON`), {notJson: true});
    }
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener('abort', cancel);
  }
}

// ---------- 3D scene ----------

function reportSceneFault(error) {
  if (stage.faulted) return;
  stage.faulted = true;
  console.error(error);
  note('note-scene', 'The 3D scene reported an error and may be out of date.');
}

function sceneCall(method, ...args) {
  const scene = stage.scene;
  if (!scene || typeof scene[method] !== 'function') return undefined;
  try {
    const result = scene[method](...args);
    if (typeof result?.catch === 'function') result.catch(reportSceneFault);
    return result;
  } catch (error) {
    reportSceneFault(error);
    return undefined;
  }
}

function sceneUnavailable(message) {
  stage.failed = true;
  $('view').dataset.scene = 'off';
  note('note-scene', message);
  // With no 3D scene, give the real footage the room.
  if (!state.feedChosen) setFeed('dock');
}

async function loadSceneMap(url = TOWN_URL, mapName = null) {
  stage.mapUrl = url;
  stage.mapName = mapName;
  const scene = stage.scene;
  if (!scene || (stage.loadedMap === url && stage.loadedMapName === mapName) ||
      (stage.mapRequest?.url === url && stage.mapRequest.mapName === mapName)) return;
  stage.mapRequest?.control.abort();
  stage.loadedMap = null;
  sceneCall('setMap', null);
  note('note-map', '');
  const request = {url, mapName, control: new AbortController()};
  stage.mapRequest = request;
  try {
    if (typeof url !== 'string' || (url !== './api/map' && !/^\.\/media\/(?:[\w-]+\/)*[\w-]+\.json$/i.test(url))) throw new Error('Invalid road map path');
    const data = await getJson(url, {timeout: 30000, signal: request.control.signal});
    if (stage.mapRequest !== request) return;
    if (!Array.isArray(data.roads) || !data.roads.length || (mapName && data.producer?.map !== mapName)) {
      throw new Error('Road geometry does not match the recording');
    }
    scene.setMap(data);
    stage.loadedMap = url;
    stage.loadedMapName = mapName;
  } catch (error) {
    if (request.control.signal.aborted) return;
    console.error(error);
    note('note-map', 'Road map unavailable. The scene is placed from telemetry alone.');
  } finally {
    if (stage.mapRequest === request) stage.mapRequest = null;
  }
}

async function bootScene() {
  note('note-scene', 'Loading the 3D scene…');
  let module;
  try {
    module = await import('./perception-scene.js');
    if (typeof module.PerceptionScene !== 'function') throw new Error('PerceptionScene is not exported');
  } catch (error) {
    console.error(error);
    sceneUnavailable('The 3D scene could not be loaded. Footage and telemetry are still available.');
    return;
  }
  try {
    stage.scene = new module.PerceptionScene($('scene'));
  } catch (error) {
    console.error(error);
    sceneUnavailable('3D rendering is unavailable in this browser. Turn on WebGL to see the vehicle and its surroundings.');
    return;
  }
  $('view').dataset.scene = 'on';
  sceneCall('setView', state.view);
  for (const [layer, visible] of Object.entries(state.layers)) sceneCall('setLayer', layer, visible);
  if (state.replay) presentAt(position());
  else if (state.frame) sceneCall('update', state.frame);

  note('note-scene', 'Loading the vehicle…');
  const scene = stage.scene;
  const car = Promise.resolve()
    .then(() => scene.loadCar(CAR_URL))
    .then(() => true, error => { console.error(error); return false; });
  const [carReady] = await Promise.all([car, loadSceneMap(stage.mapUrl, stage.mapName)]);
  if (!stage.faulted) note('note-scene', carReady ? '' : 'The vehicle model could not be loaded.');
  if (state.replay) presentAt(position());
  else if (state.frame) sceneCall('update', state.frame);
}

function setView(view) {
  state.view = view;
  sceneCall('setView', view);
  for (const button of viewButtons) attr(button, 'aria-pressed', String(button.dataset.view === view));
}

function setLayer(layer, visible) {
  state.layers[layer] = visible;
  if(layer==='tracks')renderCameraBoxes();
  sceneCall('setLayer', layer, visible);
  for (const button of layerButtons) {
    if (button.dataset.layer === layer) attr(button, 'aria-pressed', String(visible));
  }
}

function renderLayers(frame) {
  const available = {
    path: !frame || (Array.isArray(frame.observer_map?.path) && frame.observer_map.path.length > 1),
    tracks: !frame || Array.isArray(frame.tracks),
    returns: !frame || Array.isArray(frame.lidar_xyz) || Array.isArray(frame.lidar) || Array.isArray(frame.radar) || Boolean(frame.corner_radar)
  };
  for (const button of layerButtons) {
    const ready = available[button.dataset.layer];
    if (button.disabled === ready) button.disabled = !ready;
    attr(button, 'title', ready ? LAYER_TITLES[button.dataset.layer] : 'Not recorded in this drive');
  }
}

// ---------- Frame presentation ----------

function present(frame) {
  if (frame === state.frame) return;
  state.frame = frame;
  sceneCall('update', frame);
  renderTelemetry(frame);
}

function activityOf(frame) {
  const speed = frame?.speed_mps;
  if (!finite(speed)) {
    return {state: 'idle', label: state.source === 'training' ? 'Training demonstration' : 'Standing by'};
  }
  const {brake, throttle, steer} = frame;
  if (![brake, throttle, steer].some(finite)) {
    return speed < .15 ? {state: 'stopped', label: 'Stopped'} : {state: 'moving', label: 'Driving'};
  }
  if (speed < .15) {
    return throttle > .05 && !(brake > .05)
      ? {state: 'accelerating', label: 'Pulling away'}
      : {state: 'stopped', label: 'Stopped'};
  }
  if (brake > .05) return {state: 'braking', label: 'Braking'};
  if (Math.abs(steer) >= .15) {
    return steer < 0 ? {state: 'left', label: 'Turning left'} : {state: 'right', label: 'Turning right'};
  }
  if (throttle > .45) return {state: 'accelerating', label: 'Accelerating'};
  return {state: 'cruising', label: 'Cruising'};
}

function renderActivity(frame) {
  const next = activityOf(frame);
  const key = next.state + '|' + next.label;
  if (key === shownActivity) {
    candidateActivity = '';
    return;
  }
  // While time is running, a new activity must hold for two frames before it replaces the label.
  const running = state.source === 'live' || isPlaying();
  if (running && frame && shownActivity && candidateActivity !== key) {
    candidateActivity = key;
    return;
  }
  shownActivity = key;
  candidateActivity = '';
  $('cluster').dataset.state = next.state;
  text('activity', next.label);
}

function meter(id, name, value) {
  $(id).style.setProperty(name, finite(value) ? clamp(value, 0, 1).toFixed(3) : '0');
}

function trackKind(label) {
  if (ROAD_USER_LABELS.has(label)) return 'vru';
  return VEHICLE_LABELS.has(label) ? 'vehicle' : 'other';
}

function trackBearing(track) {
  if (track.x < -1) return 'behind';
  if (Math.abs(track.y) <= CORRIDOR_HALF_WIDTH_M) return 'ahead';
  return track.y < 0 ? 'left' : 'right';
}

function trackMeta(track) {
  const parts = [];
  if (track.coasting) {
    parts.push(finite(track.age_s) ? `Coasting ${tenth.format(track.age_s)} s` : 'Coasting');
  } else if (Array.isArray(track.sources) && track.sources.length) {
    parts.push(track.sources.map(source => SOURCE_LABELS[source] ?? sentence(String(source))).join(' + '));
  }
  if (Array.isArray(track.cameras) && track.cameras.length) {
    parts.push(track.cameras.map(view => String(view).replaceAll('_', ' ')).join(' / '));
  }
  if (finite(track.speed_mps)) parts.push(`${whole.format(track.speed_mps * 3.6)} km/h`);
  if (finite(track.radial_mps)) parts.push(`Doppler ${signedTenth.format(track.radial_mps)} m/s`);
  return parts.join(' · ');
}

function renderSensing(frame) {
  const visible = Boolean(frame) && state.source !== 'training';
  show('sensing', visible);
  const counts=new Map();
  for(const head of measuredHeads(frame)){
    const lamp=cameraLamp(head);
    if(lamp==='unknown')continue;
    const key=`${lamp}/${head.view??''}`;
    counts.set(key,(counts.get(key)??0)+1);
  }
  const lights=$('camera-lights');
  lights.hidden=!counts.size;
  lights.replaceChildren(...[...counts].map(([key,count])=>{
    const [lamp,view]=key.split('/');
    const row=document.createElement('li');
    row.style.setProperty('--lamp',LAMP_COLOURS[lamp]);
    const label=count>1?`${count} ${lamp} lights`:`${sentence(lamp)} light`;
    row.textContent=`${label} · ${CAMERA_LABELS[view]??'RGB'} camera`;
    return row;
  }));
  if (!visible) {
    show('ahead', false);
    trackSignature = '';
    return;
  }
  const recorded = Array.isArray(frame.tracks);
  const tracks = (recorded ? frame.tracks : [])
    .filter(track => track && finite(track.x) && finite(track.y))
    .map(track => ({track, range: Math.hypot(track.x, track.y)}))
    .sort((a, b) => a.range - b.range);

  text('track-count', recorded ? whole.format(tracks.length) : '', '');
  show('tracks-empty', !tracks.length);
  if (!tracks.length) {
    text('tracks-empty', recorded ? 'Nothing tracked right now.' : 'This drive has no sensor tracks.');
  }

  const rows = tracks.slice(0, TRACK_ROWS).map(({track, range}) => ({
    kind: trackKind(track.label),
    coasting: Boolean(track.coasting),
    name: sentence(String(track.label ?? '')) || 'Object',
    where: trackBearing(track),
    range: `${tenth.format(range)} m`,
    meta: trackMeta(track)
  }));
  const signature = JSON.stringify(rows);
  if (signature !== trackSignature) {
    trackSignature = signature;
    $('tracks').replaceChildren(...rows.map(row => {
      const item = document.createElement('li');
      item.dataset.kind = row.kind;
      if (row.coasting) item.dataset.coasting = 'true';
      const mark = document.createElement('i');
      mark.className = 'mark';
      mark.setAttribute('aria-hidden', 'true');
      const name = document.createElement('span');
      name.className = 'name';
      name.textContent = row.name;
      const where = document.createElement('small');
      where.textContent = row.where;
      name.append(where);
      const range = document.createElement('span');
      range.className = 'range';
      range.textContent = row.range;
      item.append(mark, name, range);
      if (row.meta) {
        const meta = document.createElement('span');
        meta.className = 'meta';
        meta.textContent = row.meta;
        item.append(meta);
      }
      return item;
    }));
  }

  // The nearest fresh track on the vehicle axis is called out beside the speed.
  const lead = tracks.find(({track}) => !track.coasting && track.x > 0 && Math.abs(track.y) <= CORRIDOR_HALF_WIDTH_M);
  show('ahead', Boolean(lead));
  if (lead) {
    const name = sentence(String(lead.track.label ?? '')) || 'Object';
    text('ahead', `${name} ahead · ${tenth.format(lead.track.x)} m`);
    $('ahead').dataset.kind = trackKind(lead.track.label);
  }

  const lidar = lidarCount(frame);
  const radar = frame.corner_radar || Array.isArray(frame.radar) ? (frame.radar?.length??0)+cornerReturns(frame).length : null;
  show('returns-line', lidar != null || radar != null);
  text('lidar-count', lidar == null ? null : whole.format(lidar));
  text('radar-count', radar == null ? null : whole.format(radar));
}

function fact(name, value) {
  const empty = value == null || value === '';
  show('row-' + name, !empty);
  text('fact-' + name, empty ? '' : value, '');
}

function hashFact(name, value, url) {
  const label = typeof value === 'string' ? value : '';
  const link = $(`fact-${name}-link`);
  const code = $(`fact-${name}`);
  const href = label ? artifactUrl(url) : null;
  show('row-' + name, Boolean(label));
  link.hidden = !href;
  code.hidden = Boolean(href) || !label;
  if (href) link.href = href;
  else link.removeAttribute('href');
  link.textContent = code.textContent = label.slice(0, 10);
  link.title = code.title = label;
}

function renderFrameFacts(frame) {
  const {run, training, source, build} = state;
  const policy = frame?.policy ?? run?.policy ?? (source === 'training' ? training?.policy : null);
  fact('model', typeof policy === 'string' && policy ? POLICY_LABELS[policy] ?? sentence(policy) : null);
  fact('scenario', sentence(frame?.scenario) || sentence(run?.case_id));
  const seed = frame?.seed ?? run?.seed;
  fact('seed', finite(seed) ? String(seed) : null);
  fact('weather', sentence(frame?.weather));

  const simulator = build?.simulator;
  const mapName = typeof frame?.map === 'string' && frame.map ? frame.map.split('/').pop() : source === 'live' ? '' : simulator?.map;
  const version = source !== 'live' && simulator?.server_version ? ` ${simulator.server_version}` : '';
  fact('simulator', version || mapName ? `CARLA${version}${mapName ? ' · ' + mapName : ''}` : null);

  const lateral = frame?.lat_error_m ?? frame?.lateral_error_m;
  const destination = state.options?.destinations?.find?.(item => item?.id === frame?.destination)?.label;
  const extras = {
    lateral: finite(lateral) ? `${hundredth.format(lateral)} m` : null,
    collisions: finite(frame?.collision_count) ? whole.format(frame.collision_count) : null,
    redlight: finite(frame?.red_light_violations) ? whole.format(frame.red_light_violations) : null,
    yield: finite(frame?.yield_violations) ? whole.format(frame.yield_violations) : null,
    grip: finite(frame?.grip) ? percent.format(frame.grip) : null,
    delay: finite(frame?.crossing_delay) && frame.crossing_delay !== 0 ? `${frame.crossing_delay} s` : null,
    destination: destination ?? sentence(frame?.destination),
    frame: finite(frame?.frame) ? String(frame.frame) : null,
    simtime: finite(frame?.sim_time) ? `${tenth.format(frame.sim_time)} s` : null
  };
  let any = false;
  for (const [name, value] of Object.entries(extras)) {
    fact(name, value);
    any ||= value != null && value !== '';
  }
  show('more', any);
}

function renderTelemetry(frame) {
  text('speed', finite(frame?.speed_mps) ? whole.format(Math.max(0, Math.round(frame.speed_mps * 3.6))) : '—');
  renderActivity(frame);

  const phase = typeof frame?.phase === 'string' ? sentence(frame.phase) : '';
  const context = state.source === 'training'
    ? (state.training?.subject || (frame ? 'Instructor at the controls' : ''))
    : phase ? `Scenario · ${phase}` : '';
  text('context', context, '');
  show('context', Boolean(context));

  const steer = frame?.steer;
  const throttle = frame?.throttle;
  const brake = frame?.brake;
  const driven = [steer, throttle, brake].some(finite);
  show('inputs', driven);
  if (driven) {
    meter('steer-meter', '--l', finite(steer) ? -steer : NaN);
    meter('steer-meter', '--r', steer);
    meter('throttle-meter', '--v', throttle);
    meter('brake-meter', '--v', brake);
    text('steer', !finite(steer) ? null
      : Math.abs(steer) < .005 ? 'Centre' : `${percent.format(Math.abs(steer))} ${steer < 0 ? 'left' : 'right'}`);
    text('throttle', finite(throttle) ? percent.format(throttle) : null);
    text('brake', finite(brake) ? percent.format(brake) : null);
  }

  const progress = frame?.progress_m;
  const length = frame?.route_length_m;
  const routed = finite(progress) && finite(length) && length > 0;
  show('route', routed);
  if (routed) {
    const ratio = clamp(progress / length, 0, 1);
    const rail = $('rail');
    rail.style.setProperty('--p', ratio.toFixed(4));
    attr(rail, 'aria-valuenow', String(Math.round(ratio * 100)));
    attr(rail, 'aria-valuetext', `${tenth.format(progress)} of ${whole.format(length)} metres`);
    text('progress', tenth.format(progress));
    text('route-total', whole.format(length));
    text('route-pct', percent.format(ratio));
    const destination = state.options?.destinations?.find?.(item => item?.id === frame.destination)?.label
      ?? sentence(frame.destination);
    text('route-dest', destination ? `to ${destination}` : '', '');
    show('route-dest', Boolean(destination));
  }

  text('live-clock', clockLabel(frame?.sim_time));
  text('live-frame', finite(frame?.frame) ? String(frame.frame) : null);
  renderSensing(frame);
  renderCameraBoxes();
  renderLayers(frame);
  renderFrameFacts(frame);
}

// ---------- Mode, source and run ----------

function setMode(mode, label) {
  const pill = $('mode');
  if (pill.dataset.mode !== mode) pill.dataset.mode = mode;
  if (pill.textContent !== label) pill.textContent = label;
}

function renderSource() {
  show('source-switch', state.local);
  for (const button of sourceButtons) {
    const live = button.dataset.source === 'live';
    attr(button, 'aria-pressed', String(live ? state.source === 'live' : state.source !== 'live'));
    if (live) button.dataset.active = String(state.liveActive && state.source !== 'live');
  }
  show('idle-hint', state.local && !state.liveActive && state.source !== 'recorded');
  show('idle-link', state.controls);
}

function listedRuns() {
  const runs = Array.isArray(state.build?.runs) ? state.build.runs : [];
  const showcases = Array.isArray(state.build?.showcases) ? state.build.showcases : [];
  const nativeReplays = Array.isArray(state.build?.native_replays) ? state.build.native_replays : [];
  const development = Array.isArray(state.build?.development_runs) ? state.build.development_runs : [];
  return [...runs, ...showcases, ...nativeReplays, ...development];
}

function findRun(entry) {
  return listedRuns().find(run => run?.recording_path === entry.path && (!entry.trial_id || run.trial_id === entry.trial_id)) ?? null;
}

function resolveRun(trialId) {
  const direct = state.catalog.find(entry => entry.trial_id === trialId);
  if (direct) return direct;
  const run = listedRuns().find(item => item?.trial_id === trialId);
  if (!run || !validRecordingPath(run.recording_path)) return null;
  return state.catalog.find(entry => !entry.trial_id && entry.path === run.recording_path) ?? null;
}

function fillRuns() {
  const select = $('run');
  const placeholder = new Option('Choose a run', '');
  placeholder.disabled = true;
  placeholder.hidden = true;
  const groups = new Map();
  state.catalog.forEach((entry, index) => {
    const policy = findRun(entry)?.policy;
    const group = typeof policy === 'string' && policy ? POLICY_LABELS[policy] ?? sentence(policy) : '';
    if (!groups.has(group)) groups.set(group, []);
    groups.get(group).push(new Option(entry.label, String(index)));
  });
  const nodes = groups.size > 1
    ? [...groups].map(([label, options]) => {
      const group = document.createElement('optgroup');
      group.label = label || 'Other';
      group.append(...options);
      return group;
    })
    : [...groups.values()].flat();
  select.replaceChildren(placeholder, ...nodes);
  select.value = '';
  show('run-pick', state.catalog.length > 0);
}

function syncRunPicker() {
  const index = state.entry ? state.catalog.indexOf(state.entry) : -1;
  $('run').value = index >= 0 ? String(index) : '';
}

function syncUrl(entry) {
  try {
    const url = new URL(location.href);
    if (entry?.trial_id) url.searchParams.set('run', entry.trial_id);
    else url.searchParams.delete('run');
    history.replaceState(null, '', url);
  } catch {
    // The address is cosmetic; playback does not depend on it.
  }
}

function link(id, value, label) {
  const anchor = $(id);
  const url = artifactUrl(value);
  anchor.hidden = !url;
  if (url) anchor.href = url;
  else anchor.removeAttribute('href');
  if (label) anchor.textContent = label;
}

function renderRunLinks() {
  const {source, run, training} = state;
  link('link-log', source === 'recorded' ? run?.log_url : source === 'training' ? training?.logUrl : null,
    source === 'training' ? 'Demonstration log' : 'Run log');
  link('link-replay', source === 'recorded' && state.replay ? `${state.dir}/replay.json` : null);
  const footage = (source === 'recorded' || source === 'training') && film.hasAttribute('src');
  link('link-mp4', footage ? film.src : null);
}

function renderChapters() {
  const replay = state.source === 'recorded' ? state.replay : null;
  const chapters = replay?.end > 0 ? replay.chapters : [];
  show('chapters', Boolean(chapters?.length));
  $('chapter-list').replaceChildren(...(chapters ?? []).map(chapter => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'chip';
    button.textContent = chapter.label;
    button.setAttribute('aria-label', `Jump to ${chapter.label}`);
    button.addEventListener('click', () => {
      if (state.replay === replay) seekTo(replay.times[chapter.index]);
    });
    return button;
  }));
}

function renderRun() {
  const {build, entry, run, source, training} = state;
  let title = '';
  let id = '';
  if (source === 'recorded' && entry) {
    title = entry.label;
    id = entry.trial_id || run?.trial_id || '';
  } else if (source === 'live') {
    title = 'Live drive';
    id = state.liveRun != null ? `Run ${state.liveRun}` : '';
  } else if (source === 'training') {
    title = (typeof build?.preview_label === 'string' && build.preview_label) || 'Training demonstration';
    id = training?.subject || '';
  }
  show('run-block', Boolean(title));
  text('run-title', title, '');
  text('run-id', id, '');
  show('run-id', Boolean(id));

  const recorded = source === 'recorded';
  fact('outcome', recorded && run?.outcome ? OUTCOME_LABELS[run.outcome] ?? sentence(String(run.outcome)) : null);
  const checkpoint = Object.values(build?.checkpoints ?? {}).find(item => item.sha256 === run?.model_sha256);
  hashFact('checkpoint', recorded ? run?.model_sha256 : null,
    checkpoint?.sha256 && checkpoint.sha256 === run?.model_sha256 ? checkpoint.url : null);
  const revision = run?.source ?? build?.legacy_run_source ?? build?.source;
  hashFact('revision', source === 'recorded' || source === 'training' ? (revision?.commit ?? revision?.fingerprint) : null, revision?.url);
  renderFrameFacts(state.frame);
  renderChapters();
  renderRunLinks();
}

function renderBuild() {
  const build = state.build;
  if (!build) return;
  const commit = typeof build.source?.commit === 'string' ? build.source.commit : '';
  const status = {commissioning: 'Commissioning', development: 'Development', evaluated: 'Evaluated'}[build.status] ?? '';
  const name = (typeof build.release?.label === 'string' && build.release.label) || status || commit.slice(0, 7);
  if (!name) return;
  text('build-name', name);
  show('build-kicker', true);
  const zip = artifactUrl(build.release?.zip_url);
  const download = $('build-zip');
  download.hidden = !zip;
  if (zip) download.href = zip;
  else download.removeAttribute('href');
  show('build-note', !zip && ['commissioning', 'development'].includes(build.status));
  attr($('build-link'), 'title', commit ? `Builds and logs · source revision ${commit.slice(0, 12)}` : 'Builds and logs');
}

// ---------- Footage and playback clock ----------

function videoPending() {
  return film.hasAttribute('src') && !clock.video;
}

function timerDriven() {
  return Boolean(state.replay) && state.replay.kind !== 'training';
}

function duration() {
  if (clock.video && finite(film.duration) && film.duration > 0) return film.duration;
  return timerDriven() ? state.replay.end : 0;
}

function position() {
  return clock.video ? film.currentTime : clock.t;
}

function isPlaying() {
  if (clock.video) return !film.paused && !film.ended;
  return videoPending() ? clock.pendingPlay : clock.playing;
}

function presentAt(time) {
  const {frames, times} = state.replay;
  let low = 0;
  let high = times.length - 1;
  while (low < high) {
    const middle = Math.ceil((low + high) / 2);
    if (times[middle] <= time + .001) low = middle;
    else high = middle - 1;
  }
  const frame = frames[low];
  const fraction = times[low + 1] > times[low] ? clamp((time - times[low]) / (times[low + 1] - times[low]), 0, 1) : 0;
  if (state.frame !== frame) {
    state.frame = frame;
    renderTelemetry(frame);
  }
  sceneCall('updateReplay', frames, low, fraction);
}

// Frames without their own video timestamps map one-to-one onto the video's frames.
function alignToVideo(replay, length) {
  if (!replay.uniform || !finite(length) || length <= 0) return;
  const count = replay.frames.length;
  const nominal = count > 1 ? replay.end * count / (count - 1) : 0;
  if (replay.kind !== 'training' && Math.abs(length - nominal) <= Math.max(.3, nominal * .02)) return;
  replay.times = replay.frames.map((_, index) => index * length / count);
  replay.end = replay.times[count - 1];
}

function syncNow() {
  if (state.source === 'live') return;
  // The training log only ever follows its own footage.
  if (state.replay && (clock.video || timerDriven())) presentAt(position());
  renderClock();
}

function renderClock() {
  if (state.source === 'live') return;
  const time = position();
  const total = duration();
  const seek = $('seek');
  text('clock', clockLabel(time));
  attr(seek, 'aria-valuetext', `${clockLabel(time)} of ${clockLabel(total > 0 ? total : NaN)}`);
  if (Math.abs(Number(seek.value) - time) > .004) seek.value = String(time);
  seek.style.setProperty('--p', total > 0 ? clamp(time / total, 0, 1).toFixed(4) : '0');
}

function renderTransport() {
  const live = state.source === 'live';
  const total = duration();
  const ready = !live && total > 0;
  const offered = !live && (state.source === 'recorded' || state.source === 'training') && (ready || film.hasAttribute('src'));
  show('live-strip', live);
  show('transport', offered);
  const playing = ready && isPlaying();
  const play = $('play');
  play.disabled = !ready;
  play.dataset.playing = String(playing);
  attr(play, 'aria-label', playing ? 'Pause' : 'Play');
  attr(play, 'title', playing ? 'Pause (K)' : 'Play (K)');
  const seek = $('seek');
  seek.disabled = !ready;
  seek.max = ready ? String(total) : '1';
  text('end', ready ? clockLabel(total) : '–:––');
  renderClock();
}

function play() {
  if (state.source === 'live' || duration() <= 0) return;
  if (position() >= duration() - .05) seekTo(0);
  if (clock.video) film.play().catch(() => renderTransport());
  else if (videoPending()) clock.pendingPlay = true;
  else clock.playing = true;
  renderTransport();
}

function pause() {
  if (clock.video) film.pause();
  else if (videoPending()) clock.pendingPlay = false;
  else clock.playing = false;
  renderTransport();
}

function togglePlay() {
  if (isPlaying()) pause();
  else play();
}

function seekTo(time) {
  const total = duration();
  if (state.source === 'live' || total <= 0) return;
  const target = clamp(time, 0, total);
  if (clock.video) film.currentTime = target;
  else {
    clock.t = target;
    clock.pendingT = target;
  }
  syncNow();
}

function resetPlayback() {
  clock.video = false;
  clock.playing = false;
  clock.t = 0;
  clock.pendingT = 0;
  clock.pendingPlay = false;
  film.pause();
  film.controls = false;
  film.removeAttribute('poster');
  if (film.hasAttribute('src')) {
    film.removeAttribute('src');
    film.load();
  }
  film.hidden = true;
  still.hidden = true;
  still.removeAttribute('src');
  still.alt = '';
}

function loadVideo(url, at, autoplay) {
  clock.video = false;
  clock.playing = false;
  clock.t = at;
  clock.pendingT = at;
  clock.pendingPlay = autoplay;
  film.pause();
  still.hidden = true;
  film.hidden = false;
  film.src = url;
  feedNote('Loading footage…');
  renderFeed();
  renderTransport();
  renderRunLinks();
}

function videoUrl() {
  return artifactUrl(state.entry?.camera_urls?.[state.camera]) ?? `${state.dir}/${state.camera}.mp4`;
}

function setCameras(cameras) {
  state.cameras = cameras;
  state.camera = cameras.includes(state.preferredCamera) ? state.preferredCamera : cameras[0] ?? null;
  $('camera-list').replaceChildren(...cameras.map(camera => new Option(CAMERA_LABELS[camera], camera)));
  renderFeed();
}

function setFeed(size, chosen = false) {
  state.feed = size;
  if (chosen) state.feedChosen = true;
  $('app').dataset.feed = size;
  renderFeed();
}

function renderCameraBoxes() {
  const overlay=$('feed-observations');
  const live=state.source==='live';
  const width=live?still.naturalWidth:film.videoWidth;
  const height=live?still.naturalHeight:film.videoHeight;
  const boxes=state.layers.tracks && (live||clock.video) && width && height ? cameraBoxes(state.frame,state.camera) : [];
  overlay.replaceChildren();
  if(!boxes.length)return;
  // SVG's default meet alignment matches the media element's object-fit: contain.
  overlay.setAttribute('viewBox',`0 0 ${width} ${height}`);
  for(const item of boxes){
    const [x0,y0,x1,y1]=item.box.map(n=>clamp(n,0,1));
    const rect=document.createElementNS('http://www.w3.org/2000/svg','rect');
    rect.setAttribute('x',x0*width);rect.setAttribute('y',y0*height);
    rect.setAttribute('width',(x1-x0)*width);rect.setAttribute('height',(y1-y0)*height);
    rect.setAttribute('stroke',item.lamp?LAMP_COLOURS[item.lamp]:trackKind(item.label)==='vru'?'#ffb76b':'#9edbd6');
    if(item.range_supported===false)rect.setAttribute('stroke-dasharray','3 2');
    overlay.append(rect);
  }
}

function renderFeed() {
  renderCameraBoxes();
  const source = state.source;
  const camera = state.camera ? CAMERA_LABELS[state.camera] : '';
  const live = state.liveActive ? 'Live' : state.frame ? 'Last frame' : '';
  const tag = source === 'live' ? `${live}${live && camera ? ' · ' + camera : ''}`
    : source === 'recorded' ? `Recorded${camera ? ' · ' + camera : ''}`
    : source === 'training' ? 'Training demonstration' : '';
  const tagElement = $('feed-tag');
  text('feed-tag', tag, '');
  tagElement.hidden = !tag;
  tagElement.dataset.mode = source === 'live' && !state.liveActive ? 'idle' : source;

  const list = state.cameras.some(camera => !['front','chase'].includes(camera));
  show('cam-switch', !list && state.cameras.length > 1);
  show('camera-list', list);
  if (state.camera) $('camera-list').value = state.camera;
  for (const button of cameraButtons) {
    attr(button, 'aria-pressed', String(button.dataset.camera === state.camera));
    button.hidden = !state.cameras.includes(button.dataset.camera);
  }
  text('feed-caption', source === 'training' ? 'CARLA training drive' : 'CARLA camera', '');

  const docked = state.feed === 'dock';
  // Native controls whenever the footage is the main thing on screen.
  const controls = film.hasAttribute('src') && (docked || source === 'training');
  if (film.controls !== controls) film.controls = controls;
  const title = source === 'training' ? 'Training demonstration footage'
    : `${state.entry?.label ?? 'Recorded drive'}${camera ? `, ${camera.toLowerCase()} camera` : ''}`;
  attr(film, 'aria-label', title);

  const toggle = $('feed-size');
  attr(toggle, 'aria-expanded', String(docked));
  attr(toggle, 'title', docked ? 'Collapse footage (C)' : 'Expand footage (C)');
  text('feed-size-label', docked ? 'Collapse footage' : 'Expand footage');
}

// The video takes over the clock as soon as its length is known. Some mobile
// browsers hold back `canplay` until playback is requested, so metadata is enough.
function videoReady() {
  if (!film.hasAttribute('src') || clock.video) return;
  if (state.replay) alignToVideo(state.replay, film.duration);
  if (clock.pendingT > 0 && finite(film.duration)) film.currentTime = Math.min(clock.pendingT, film.duration);
  clock.video = true;
  clock.playing = false;
  feedNote('');
  if (clock.pendingPlay) film.play().catch(() => renderTransport());
  syncNow();
  renderFeed();
  renderTransport();
}
film.addEventListener('loadedmetadata', videoReady);
film.addEventListener('canplay', videoReady);
still.addEventListener('load', renderCameraBoxes);

film.addEventListener('error', () => {
  if (!film.hasAttribute('src')) return;
  const at = clock.video ? film.currentTime : clock.pendingT;
  const resume = clock.video ? !film.paused && !film.ended : clock.pendingPlay;
  if (state.source === 'recorded') {
    const remaining = state.cameras.filter(camera => camera !== state.camera);
    setCameras(remaining);
    if (remaining.length) {
      loadVideo(videoUrl(), at, resume);
      return;
    }
  }
  film.removeAttribute('src');
  film.load();
  film.hidden = true;
  clock.video = false;
  clock.t = at;
  clock.pendingT = at;
  clock.pendingPlay = false;
  if (state.source === 'training') {
    // Without its footage the demonstration log is not replayed on its own.
    clock.playing = false;
    const image = previewImage(state.build);
    if (image) showStill(image, state.build.preview_image?.alt);
    else feedNote('The training footage could not be played.');
    setMode(image ? 'training' : 'unavailable', image ? 'Camera capture' : 'Footage unavailable');
  } else {
    clock.playing = resume && timerDriven();
    feedNote('Camera footage for this run could not be played. Telemetry still plays.');
  }
  renderFeed();
  renderTransport();
  renderRunLinks();
});

for (const event of ['play', 'pause', 'ended', 'durationchange', 'seeked']) {
  film.addEventListener(event, () => {
    renderTransport();
    syncNow();
  });
}

function showStill(url, alt) {
  still.alt = typeof alt === 'string' && alt ? alt : 'Camera frame captured in the simulator';
  still.src = url;
  still.hidden = false;
  feedNote('');
}

still.addEventListener('error', () => {
  if (still.hidden || !still.hasAttribute('src')) return;
  still.hidden = true;
  feedNote('The camera image could not be displayed.');
});

// ---------- Sources ----------

function beginSource(source) {
  loadControl?.abort();
  loadControl = new AbortController();
  resetPlayback();
  if (state.entry) state.lastEntry = state.entry;
  state.source = source;
  state.entry = null;
  state.run = null;
  state.replay = null;
  state.training = null;
  state.dir = '';
  state.cameras = [];
  state.camera = null;
  $('app').dataset.source = source;
  shownActivity = '';
  candidateActivity = '';
  note('note-data', '');
  present(null);
  if (source !== 'recorded') void loadSceneMap();
  renderTelemetry(null);
  return loadControl.signal;
}

function parseReplay(data) {
  const frames = Array.isArray(data) ? data : data?.frames;
  if (!Array.isArray(frames) || !frames.length || !frames.every(frame => finite(frame?.frame))) {
    throw new Error('Recorded telemetry is empty or invalid');
  }
  const rate = Array.isArray(data) ? NaN : data.fps;
  const stamped = frames.every(frame => finite(frame.video_time));
  const origin = frames[0].sim_time;
  const times = frames.map((frame, index) => stamped ? frame.video_time
    : finite(rate) && rate > 0 ? index / rate : frame.sim_time - origin);
  if (times.some((time, index) => !finite(time) || (index > 0 && time < times[index - 1]))) {
    throw new Error('Recorded telemetry timestamps are invalid');
  }
  return {
    kind: 'replay', uniform: !stamped, frames, times, end: times[times.length - 1],
    cameras: Array.isArray(data) ? undefined : data.cameras,
    chapters: recordedChapters(frames)
  };
}

// The training log holds the instructor's measured pose and speed for each video
// frame. Only those values are re-keyed into the frame shape; nothing is filled in.
function parseTrainingLog(log) {
  const rows = log?.frames;
  if (!Array.isArray(rows) || rows.length < 2 ||
      !rows.every(row => finite(row?.frame) && finite(row.x_m) && finite(row.y_m) && finite(row.yaw_deg))) {
    throw new Error('Training log has no usable poses');
  }
  const rate = finite(log.fps) && log.fps > 0 ? log.fps : 10;
  const map = typeof log.producer?.map === 'string' ? log.producer.map : undefined;
  const frames = rows.map(row => ({
    frame: row.frame,
    sim_time: row.sim_time_s,
    speed_mps: row.speed_mps,
    policy: log.policy,
    scenario: log.scenario,
    seed: log.seed,
    map,
    observer_map: {ego: [row.x_m, row.y_m, row.yaw_deg]}
  }));
  const times = rows.map((_, index) => index / rate);
  return {kind: 'training', uniform: true, frames, times, end: times[times.length - 1]};
}

const replays = new Map();

async function loadChapterManifest(entry, replay, signal) {
  if (!entry.trial_id || typeof entry.manifest !== 'string' ||
      !/^\.\/media\/(?:[\w-]+\/)*manifest\.json$/i.test(entry.manifest)) return;
  try {
    const manifest = await getJson(entry.manifest, {signal});
    if (signal.aborted || state.replay !== replay || manifest.trial_id !== entry.trial_id ||
        manifest.first_frame !== replay.frames[0].frame || manifest.last_frame !== replay.frames.at(-1).frame) return;
    replay.chapters = recordedChapters(replay.frames, manifest);
    renderChapters();
  } catch {
    // Playback and run details are available without optional chapter metadata.
  }
}

async function openRecorded(entry, startSeconds = 0) {
  const signal = beginSource('recorded');
  state.entry = entry;
  state.run = findRun(entry);
  state.dir = entry.path ? `./media/${entry.path}` : './media';
  const mapUrl = entry.map_url ?? state.run?.map_url;
  const mapName = entry.map ?? state.run?.map;
  void loadSceneMap(mapUrl ?? TOWN_URL, typeof mapName === 'string' ? mapName : null);
  if (!state.feedChosen && !stage.failed) setFeed('inset');
  setMode('loading', 'Loading');
  feedNote('Loading footage…');
  syncRunPicker();
  renderSource();
  renderRun();
  renderFeed();
  renderTransport();

  const dir = state.dir;
  let replay = replays.get(dir) ?? null;
  if (!replay) {
    try {
      replay = await loadReplay(entry.replay_pack_url ?? state.run?.replay_pack_url, signal, parseReplay,
        () => getJson(`${dir}/replay.json`, {signal, timeout: 30000}));
      if (replay && !signal.aborted) replays.set(dir, replay);
    } catch (error) {
      if (signal.aborted) return;
      console.error(error);
      note('note-data', 'Recorded telemetry for this run could not be loaded. Only its footage is shown.');
    }
  }
  if (signal.aborted) return;
  state.replay = replay;
  setMode('recorded', 'Recorded');

  const listed = Array.isArray(entry.cameras) ? entry.cameras : Array.isArray(replay?.cameras) ? replay.cameras : ['front','chase'];
  setCameras([...new Set(listed.filter(camera => CAMERA_ORDER.includes(camera)))]);
  const start = replay && finite(startSeconds) ? Math.max(0, Math.min(startSeconds, replay.end)) : 0;
  if (replay) presentAt(start);
  if (state.camera) {
    loadVideo(videoUrl(), start, !reducedMotion);
  } else {
    feedNote('No camera footage is listed for this run.');
    clock.playing = Boolean(replay) && !reducedMotion;
    if (!replay) setMode('unavailable', 'Unavailable');
  }
  renderRun();
  renderTransport();
  if (replay && !replay.chapters.length && !replay.frames.some(frame => Array.isArray(frame.events))) {
    void loadChapterManifest(entry, replay, signal);
  }
}

function trainingLogUrl(build) {
  if (typeof build?.preview_log === 'string') return build.preview_log;
  const evidence = Array.isArray(build?.evidence) ? build.evidence : [];
  return evidence.find(item => typeof item?.subject === 'string' && typeof item.url === 'string' && /\.json$/i.test(item.url))?.url ?? null;
}

// Commissioning: there are no learned drives yet. Show the real training footage,
// and move the vehicle only from the pose log recorded with that footage.
async function openTraining() {
  const build = state.build;
  const video = artifactUrl(build?.preview_video);
  const image = previewImage(build);
  const signal = beginSource(video || image ? 'training' : 'none');
  syncRunPicker();
  renderSource();
  if (!video && !image) {
    setMode('unavailable', 'No drives yet');
    note('note-data', build
      ? 'No recorded drives have been published for this build yet.'
      : 'No recorded drives were found here.');
    feedNote('No footage yet.');
    renderRun();
    renderFeed();
    renderTransport();
    return;
  }
  if (!state.feedChosen) setFeed('dock');
  const logUrl = video ? trainingLogUrl(build) : null;
  state.training = {subject: '', policy: null, logUrl};
  if (video) {
    setMode('training', 'Training footage');
    if (image) film.poster = image;
    loadVideo(video, 0, !reducedMotion);
  } else {
    setMode('training', 'Camera capture');
    showStill(image, build.preview_image?.alt);
  }
  renderRun();
  renderFeed();
  renderTransport();

  const url = artifactUrl(logUrl);
  if (!url) return;
  try {
    const log = await getJson(url, {signal, timeout: 30000});
    if (signal.aborted) return;
    const replay = parseTrainingLog(log);
    state.training = {
      subject: typeof log.subject === 'string' ? log.subject : '',
      policy: typeof log.policy === 'string' ? log.policy : null,
      logUrl
    };
    if (clock.video) alignToVideo(replay, film.duration);
    state.replay = replay;
    renderRun();
    renderTelemetry(state.frame);
    syncNow();
  } catch (error) {
    if (!signal.aborted) console.error(error);
  }
}

function showDefault() {
  state.defaultShown = true;
  const entry = state.entry ?? state.lastEntry ?? state.catalog[0];
  if (entry) return openRecorded(entry);
  return openTraining();
}

function showUnresolved(trialId) {
  beginSource('none');
  setMode('unavailable', 'Run unavailable');
  note('note-data', `Run “${trialId.slice(0, 80)}” has no published recording here. Choose another run, or find its log in Builds & logs.`);
  feedNote('No footage for this run.');
  syncRunPicker();
  renderSource();
  renderRun();
  renderFeed();
  renderTransport();
}

function enterLive() {
  beginSource('live');
  state.liveRun = undefined;
  if (!state.feedChosen && !stage.failed) setFeed('inset');
  setMode('connecting', 'Connecting');
  feedNote('Waiting for a camera frame…');
  syncRunPicker();
  renderSource();
  renderRun();
  renderFeed();
  renderTransport();
}

function showLiveImage(frame) {
  const images = frame?.cameras ?? frame?.images ?? {};
  const cameras = CAMERA_ORDER.filter(camera => typeof images[camera] === 'string' && images[camera]);
  if (cameras.join() !== state.cameras.join() || !cameras.includes(state.camera)) setCameras(cameras);
  const jpeg = state.camera ? images[state.camera] : null;
  if (!jpeg) {
    still.hidden = true;
    feedNote('Waiting for a camera frame…');
    return;
  }
  const source = jpeg.startsWith('data:') ? jpeg : 'data:image/jpeg;base64,' + jpeg;
  if (still.getAttribute('src') !== source) still.src = source;
  still.alt = `Live simulator ${CAMERA_LABELS[state.camera].toLowerCase()} camera`;
  still.hidden = false;
  feedNote('');
}

function showLive(frame) {
  setMode('live', 'Live');
  if (frame.run_id !== state.liveRun) {
    state.liveRun = frame.run_id;
    void loadSceneMap(frame.map_url ?? TOWN_URL, frame.map?.split('/').at(-1) ?? null);
    renderRun();
  }
  showLiveImage(frame);
  present(frame);
}

function onLivePoll(frame, status) {
  const active = Boolean(frame);
  if (active !== state.liveActive) {
    state.liveActive = active;
    renderSource();
    renderFeed();
  }
  if (active && state.wanted !== 'recorded') {
    if (state.source !== 'live') enterLive();
    showLive(frame);
    return;
  }
  if (state.source === 'live') {
    if (state.frame) {
      // Hold the final frame of the drive on screen.
      setMode('idle', status?.state === 'complete' ? 'Drive ended' : status?.state === 'error' ? 'Drive stopped' : 'Simulator idle');
    } else if (state.wanted === 'auto') {
      showDefault();
    } else {
      setMode('idle', status?.state === 'queued' ? 'Starting' : 'Simulator idle');
      feedNote(status?.state === 'queued' ? 'Waiting for the first simulator frame…' : 'No drive is running.');
    }
  } else if (state.source === 'none' && state.wanted === 'auto' && !state.defaultShown) {
    showDefault();
  }
}

// The local console exposes status without requiring a running drive. Read-only
// console servers expose just the frame endpoint, which returns 503 while idle.
async function watchLive() {
  let unreachable = 0;
  for (;;) {
    let status = null;
    let frame = null;
    let waiting = false;
    try {
      const value = await getJson('./api/status', {timeout: 1900, cache: 'no-store'});
      if (typeof value?.state === 'string') status = value;
    } catch {
      // A read-only console may still have camera frames.
    }
    if (!status || status.state === 'running') {
      try {
        frame = await getJson('./api/frame', {timeout: 1900, cache: 'no-store'});
      } catch (error) {
        waiting = error.status === 503;
      }
    }
    if (!state.local && (typeof status?.state === 'string' || finite(frame?.frame) || waiting)) {
      state.local = true;
      renderSource();
    }
    if (!state.local) {
      if (++unreachable >= 2) {
        if (state.source === 'none' && state.wanted === 'auto' && !state.defaultShown) showDefault();
        return;
      }
      await sleep(600);
      continue;
    }
    const active = finite(frame?.frame) && (!status || (status.state === 'running' &&
      (frame.run_id == null || status.run_id == null || frame.run_id === status.run_id)));
    onLivePoll(active ? frame : null, status);
    await sleep(active && state.source === 'live' ? 100 : 1200);
  }
}

async function loadBuild() {
  try {
    const build = await getJson('./media/build.json', {timeout: 8000});
    return build && typeof build === 'object' && !Array.isArray(build) ? build : null;
  } catch {
    return null;
  }
}

async function loadCatalog() {
  for (const file of ['./media/viewer-catalog.json', './media/catalog.json']) {
    try {
      const data = await getJson(file, {timeout: 8000});
      const entries = Array.isArray(data) ? data : data?.records;
      if (!Array.isArray(entries)) continue;
      const paths = new Set();
      const catalog = entries.filter(entry => {
        if (typeof entry?.label !== 'string' || !entry.label.trim() || !validRecordingPath(entry.path) || paths.has(entry.path)) return false;
        paths.add(entry.path);
        return true;
      });
      if (catalog.length) return catalog;
    } catch {
      // Earlier releases only include the primary catalog.
    }
  }
  return [];
}

function catalogFromBuild() {
  return listedRuns().filter(run => run?.recording_path && validRecordingPath(run.recording_path)).map(run => ({
    trial_id: run.trial_id,
    path: run.recording_path,
    label: run.label || state.build?.title || sentence(run.case_id) || run.trial_id || 'Recorded drive',
    cameras: run.cameras,
    map: run.map,
    map_url: run.map_url,
    replay_pack_url: run.replay_pack_url
  }));
}

async function confirmControls() {
  try {
    const options = await getJson('./api/options', {timeout: 1900, cache: 'no-store'});
    if (!options || typeof options !== 'object' || !Array.isArray(options.scenarios)) return;
    state.options = options;
    state.controls = true;
    state.local = true;
    renderSource();
  } catch {
    // Not a local simulator, or a read-only one.
  }
}

// ---------- Input ----------

function closeMenu() {
  $('links').dataset.open = 'false';
  attr($('menu-btn'), 'aria-expanded', 'false');
}

function bind() {
  for (const button of sourceButtons) {
    button.addEventListener('click', () => {
      if (button.dataset.source === 'live') {
        state.wanted = 'live';
        syncUrl(null);
        if (state.source !== 'live') enterLive();
      } else {
        state.wanted = 'recorded';
        if (state.source === 'live' || state.source === 'none') showDefault();
      }
      renderSource();
    });
  }

  $('run').addEventListener('change', () => {
    const entry = state.catalog[Number($('run').value)];
    if (!entry) return;
    state.wanted = 'recorded';
    syncUrl(entry);
    openRecorded(entry);
  });

  for (const button of viewButtons) button.addEventListener('click', () => setView(button.dataset.view));
  for (const button of layerButtons) {
    button.addEventListener('click', () => setLayer(button.dataset.layer, !state.layers[button.dataset.layer]));
  }

  function selectCamera(camera) {
    if (camera === state.camera || !state.cameras.includes(camera)) return;
    state.preferredCamera = camera;
    state.camera = camera;
    if (state.source === 'live') {
      showLiveImage(state.frame);
      renderFeed();
    } else if (state.source === 'recorded') {
      loadVideo(videoUrl(), position(), isPlaying());
    }
  }
  for (const button of cameraButtons) button.addEventListener('click', () => selectCamera(button.dataset.camera));
  $('camera-list').addEventListener('change', event => selectCamera(event.target.value));

  $('feed-size').addEventListener('click', () => setFeed(state.feed === 'dock' ? 'inset' : 'dock', true));
  $('play').addEventListener('click', togglePlay);
  $('seek').addEventListener('input', () => seekTo(Number($('seek').value)));

  $('menu-btn').addEventListener('click', () => {
    const open = $('links').dataset.open !== 'true';
    $('links').dataset.open = String(open);
    attr($('menu-btn'), 'aria-expanded', String(open));
  });
  document.addEventListener('click', event => {
    if (!event.target.closest('#links, #menu-btn')) closeMenu();
  });

  document.querySelector('.more-link').addEventListener('click', () => {
    requestAnimationFrame(() => $('brief').focus({preventScroll: true}));
  });

  document.addEventListener('keydown', event => {
    if (event.defaultPrevented || event.ctrlKey || event.metaKey || event.altKey) return;
    if (event.key === 'Escape') {
      closeMenu();
      return;
    }
    const target = event.target;
    const tag = target?.tagName;
    if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA' || tag === 'VIDEO' || target?.isContentEditable) return;
    const onControl = tag === 'BUTTON' || tag === 'A' || tag === 'SUMMARY';
    switch (event.key) {
      case ' ':
        if (onControl) return;
        event.preventDefault();
        togglePlay();
        break;
      case 'k': case 'K': togglePlay(); break;
      case 'ArrowLeft': seekTo(position() - 2); break;
      case 'ArrowRight': seekTo(position() + 2); break;
      case 'f': case 'F': setView('follow'); break;
      case 'o': case 'O': setView('overview'); break;
      case 'c': case 'C': setFeed(state.feed === 'dock' ? 'inset' : 'dock', true); break;
      case 'p': case 'P': if (!layerButton('path').disabled) setLayer('path', !state.layers.path); break;
      case 't': case 'T': if (!layerButton('tracks').disabled) setLayer('tracks', !state.layers.tracks); break;
      case 'r': case 'R': if (!layerButton('returns').disabled) setLayer('returns', !state.layers.returns); break;
      default: break;
    }
  });

  new ResizeObserver(() => {
    $('app').style.setProperty('--bar-h', Math.ceil($('bar').getBoundingClientRect().height) + 'px');
  }).observe($('bar'));

  addEventListener('pagehide', event => {
    if (!event.persisted) sceneCall('dispose');
  });
}

function layerButton(layer) {
  return layerButtons.find(button => button.dataset.layer === layer);
}

function tick(stamp) {
  requestAnimationFrame(tick);
  if (document.hidden) {
    lastTick = undefined;
    return;
  }
  const elapsed = lastTick == null ? 0 : Math.min(.25, (stamp - lastTick) / 1000);
  lastTick = stamp;
  if (state.source === 'live') return;
  if (!clock.video && clock.playing && timerDriven()) {
    clock.t = Math.min(state.replay.end, clock.t + elapsed);
    if (clock.t >= state.replay.end) {
      clock.playing = false;
      renderTransport();
    }
  }
  syncNow();
}

async function boot() {
  bind();
  renderTelemetry(null);
  renderFeed();
  renderTransport();
  renderSource();
  bootScene();
  if (LOCAL_CONSOLE) confirmControls();

  const search = new URLSearchParams(location.search);
  const requested = search.get('run');
  const [build, catalog] = await Promise.all([loadBuild(), loadCatalog()]);
  state.build = build;
  state.catalog = catalog.length ? catalog : catalogFromBuild();
  renderBuild();
  fillRuns();

  if (requested) {
    state.wanted = 'recorded';
    const entry = resolveRun(requested);
    if (entry) openRecorded(entry, Number(search.get('at')));
    else showUnresolved(requested);
  } else if (!LOCAL_CONSOLE) {
    showDefault();
  }
  requestAnimationFrame(tick);
  if (LOCAL_CONSOLE) watchLive();
}

boot();
