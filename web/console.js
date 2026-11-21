import {SensorView} from './sensor.js';
import {cornerReturns, lidarCount} from './sensor-returns.mjs';
import {loadReplay} from './replay-pack.mjs';
import {cameraBoxes, LAMP_COLOURS} from './lamp-observations.mjs';
import {artifactUrl, previewImage, setArtifactLink, validRecordingPath} from './build-data.js';

const ids = [
  'source', 'source-picker', 'source-state', 'encounter-picker', 'encounter', 'viewport', 'film', 'stream',
  'detections', 'camera-empty', 'view-label', 'frame-label',
  'playback', 'play', 'clock', 'seek', 'end-clock', 'loop', 'live-strip', 'live-clock', 'ir-note', 'front-note',
  'sensor-panel', 'data-panel', 'sensor', 'radar-reading', 'lidar-count', 'radar-count', 'map', 'frame', 'speed',
  'sensor-unit', 'sensor-controls', 'plot-scale', 'route-source', 'returns-legend', 'route-legend',
  'route-traffic', 'route-hazard', 'route-hazard-label',
  'steer', 'steer-direction', 'throttle', 'brake',
  'progress', 'route-total', 'lateral', 'collisions', 'scenario', 'policy', 'seed', 'phase', 'weather',
  'run-panel', 'run-form', 'run-scenario', 'run-destination', 'run-seed', 'run-delay', 'run-start',
  'run-repeat', 'run-stop', 'run-status', 'run-request-error', 'destination-field', 'delay-field',
  'weather-field', 'grip-field', 'run-weather', 'run-grip', 'telemetry-grip', 'grip',
  'telemetry-destination', 'destination', 'telemetry-delay', 'crossing-delay',
  'telemetry-red-light', 'red-light-count', 'telemetry-yield', 'yield-count',
  'preview-video', 'preview-still', 'camera-evidence', 'capture-caption', 'archive-trial-row', 'archive-trial',
  'archive-source-row', 'archive-source-link', 'archive-source-text', 'archive-model-row', 'archive-model-link',
  'archive-model-text', 'archive-mp4', 'archive-replay', 'archive-log'
];
const ui = Object.fromEntries(ids.map(id => [id, document.getElementById(id)]));
const cameraButtons = [...document.querySelectorAll('[data-camera]')];
const sensorButtons = [...document.querySelectorAll('[data-sensor-view]')];
const cameraLabels = {
  chase: 'Chase camera', front: 'Front camera', infrared: 'Thermal proxy',
  left: 'Left camera', right: 'Right camera', rear_left: 'Rear left camera', rear_right: 'Rear right camera'
};
const policyLabels = {fusion: 'Learned multisensor', camera: 'Learned vision'};
const delayScenarios = new Set(['trip', 'pedestrian', 'occluded']);
const terminalStates = new Set(['complete', 'error']);
const recordings = new Map();
const sensor = new SensorView(ui.sensor, ui['radar-reading']);
const integer = new Intl.NumberFormat(undefined, {maximumFractionDigits: 0});
const decimal = new Intl.NumberFormat(undefined, {minimumFractionDigits: 1, maximumFractionDigits: 1});
const precision = new Intl.NumberFormat(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
const delayFormat = new Intl.NumberFormat(undefined, {maximumFractionDigits: 3});
const percent = new Intl.NumberFormat(undefined, {style: 'percent'});
const signedPercent = new Intl.NumberFormat(undefined, {style: 'percent', signDisplay: 'exceptZero'});

let sourceControl;
let mode = '';
let selectedCamera = 'chase';
let availableCameras = [];
let latestFrame;
let renderedFrame;
let replay;
let mediaReady = false;
let pendingPosition = 0;
let pendingPlay = false;
let animation;
let catalog;
let buildRequest;
let options;
let runStatus;
let runError;
let submitting = false;
let submittal = 0;
let stopState = '';

function setText(id, value) {
  const text = String(value ?? '—');
  if (ui[id].textContent !== text) ui[id].textContent = text;
}

function setMode(nextMode, label) {
  mode = nextMode;
  const stopLabel = mode === 'live' && simulatorStopLabel();
  ui['source-state'].dataset.mode = stopLabel ? (stopState === 'stopped' ? 'stopped' : 'stopping') : mode;
  setText('source-state', stopLabel || label);
  ui['encounter-picker'].hidden = mode !== 'replay' || !ui.encounter.options.length;
  ui['run-panel'].hidden = mode !== 'live' || !options;
  ui.playback.hidden = mode === 'live';
  ui['live-strip'].hidden = mode !== 'live';
}

function simulatorStopLabel() {
  if (!stopState) return '';
  return stopState === 'stopped' ? 'Simulator stopped' : 'Stopping simulator';
}

function showMessage(message) {
  const empty = ui['camera-empty'];
  empty.hidden = !message;
  if (message && empty.lastElementChild.textContent !== message) {
    empty.lastElementChild.textContent = message;
  }
  ui['view-label'].hidden = Boolean(message);
  ui['frame-label'].hidden = Boolean(message);
  renderDetections();
}

function setCameras(cameras) {
  availableCameras = cameras;
  if (!cameras.includes(selectedCamera) && cameras.length) selectedCamera = cameras[0];
  for (const button of cameraButtons) {
    button.disabled = !cameras.includes(button.dataset.camera);
    if (button.dataset.camera.startsWith('rear_')) button.hidden = button.disabled;
    button.setAttribute('aria-pressed', button.dataset.camera === selectedCamera);
  }
  setText('view-label', cameraLabels[selectedCamera]);
  ui['ir-note'].hidden = selectedCamera !== 'infrared' || !cameras.includes(selectedCamera);
}

function numeric(value, format = integer) {
  return Number.isFinite(value) ? format.format(value) : '—';
}

function words(value) {
  if (typeof value !== 'string' || !value) return '—';
  const label = value.replace(/[_-]/g, ' ').replace(/([a-z])([A-Z])/g, '$1 $2');
  return label[0].toUpperCase() + label.slice(1);
}

function timeLabel(value) {
  if (!Number.isFinite(value)) return '–:––';
  const seconds = Math.floor(Math.max(0, value));
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`;
}

function steeringDirection(steer) {
  if (!Number.isFinite(steer)) return '';
  if (Math.abs(steer) < 0.005) return 'Center';
  return steer > 0 ? 'Right' : 'Left';
}

function renderDetections() {
  const overlay = ui.detections;
  const detections = cameraBoxes(renderedFrame, selectedCamera);
  const visible = ui['camera-empty'].hidden;
  const width = mode === 'live' ? ui.stream.naturalWidth : ui.film.videoWidth;
  const height = mode === 'live' ? ui.stream.naturalHeight : ui.film.videoHeight;
  const show = visible && detections.length && width && height;
  overlay.hidden = !show;
  ui['front-note'].hidden = !show;
  if (!show) return;
  setText('front-note', `Object detections from the ${selectedCamera.replaceAll('_',' ')} RGB camera.`);
  const viewWidth = ui.viewport.clientWidth;
  const viewHeight = ui.viewport.clientHeight;
  const scale = Math.min(viewWidth / width, viewHeight / height);
  overlay.style.width = `${width * scale}px`;
  overlay.style.height = `${height * scale}px`;
  overlay.style.left = `${(viewWidth - width * scale) / 2}px`;
  overlay.style.top = `${(viewHeight - height * scale) / 2}px`;
  const boxes = [];
  for (const detection of detections) {
    const {box, label, score} = detection;
    if (!Array.isArray(box) || box.length !== 4 || !box.every(Number.isFinite)) continue;
    const [x1, y1, x2, y2] = box.map(value => Math.max(0, Math.min(1, value)));
    if (x1 >= x2 || y1 >= y2) continue;
    const element = document.createElement('div');
    element.className = 'detection';
    if (detection.lamp) element.style.borderColor = LAMP_COLOURS[detection.lamp];
    element.style.left = `${x1 * 100}%`;
    element.style.top = `${y1 * 100}%`;
    element.style.width = `${(x2 - x1) * 100}%`;
    element.style.height = `${(y2 - y1) * 100}%`;
    const caption = document.createElement('span');
    caption.className = 'detection-label';
    const confidence = Number.isFinite(score) ? ` ${Math.round(score * 100)}%` : '';
    caption.textContent = `${label}${confidence}${detection.range_supported === false ? ' · bearing only' : ''}`;
    element.append(caption);
    boxes.push(element);
  }
  overlay.replaceChildren(...boxes);
}

function renderTelemetry(frame) {
  if (frame === renderedFrame && frame) return;
  renderedFrame = frame;
  const lidar = frame?.lidar_xyz ?? frame?.lidar ?? [];
  const radar = (frame?.radar ?? []).concat(cornerReturns(frame).map(([x,y,z,v])=>[x,y,v]));
  setText('lidar-count', frame ? numeric(lidarCount(frame)) : '—');
  setText('radar-count', frame ? numeric(radar.length) : '—');
  setText('map', frame?.map?.split('/').pop() || 'CARLA');
  setText('frame', numeric(frame?.frame));
  setText('speed', numeric(frame?.speed_mps * 3.6, decimal));
  setText('steer', numeric(frame?.steer, signedPercent));
  setText('steer-direction', steeringDirection(frame?.steer));
  setText('throttle', numeric(frame?.throttle, percent));
  setText('brake', numeric(frame?.brake, percent));
  setText('progress', numeric(frame?.progress_m, decimal));
  const length = frame?.route_length_m;
  setText('route-total', Number.isFinite(length) ? `/ ${numeric(length)} m` : 'm');
  setText('lateral', numeric(frame?.lat_error_m, precision));
  setText('collisions', numeric(frame?.collision_count));
  setText('scenario', words(frame?.scenario));
  setText('policy', policyLabels[frame?.policy] ?? words(frame?.policy));
  setText('seed', Number.isFinite(frame?.seed) ? String(frame.seed) : '—');
  setText('phase', words(frame?.phase));
  setText('weather', words(frame?.weather));
  ui['telemetry-grip'].hidden = !Number.isFinite(frame?.grip);
  setText('grip', numeric(frame?.grip, percent));
  ui['telemetry-destination'].hidden = !frame?.destination;
  const destination = options?.destinations.find(item => item.id === frame?.destination)?.label;
  setText('destination', destination ?? words(frame?.destination));
  const showDelay = delayScenarios.has(frame?.scenario) && Number.isFinite(frame?.crossing_delay);
  ui['telemetry-delay'].hidden = !showDelay;
  setText('crossing-delay', `${numeric(frame?.crossing_delay, delayFormat)} s`);
  ui['telemetry-red-light'].hidden = !Number.isFinite(frame?.red_light_violations);
  setText('red-light-count', numeric(frame?.red_light_violations));
  ui['telemetry-yield'].hidden = !Number.isFinite(frame?.yield_violations);
  setText('yield-count', numeric(frame?.yield_violations));
  setText('live-clock', timeLabel(frame?.sim_time));
  sensor.update(lidar, radar, frame?.observer_map, frame);
  renderSensorPanel();
  renderDetections();
}

function renderSensorPanel() {
  const showingRoute = sensor.view === 'route';
  const hasRoute = Boolean(sensor.route);
  ui['sensor-controls'].hidden = !hasRoute;
  ui['sensor-unit'].hidden = hasRoute;
  ui['plot-scale'].hidden = showingRoute;
  ui['route-source'].hidden = !showingRoute;
  ui['returns-legend'].hidden = showingRoute;
  ui['route-legend'].hidden = !showingRoute;
  ui['route-traffic'].hidden = !sensor.route?.traffic;
  ui['route-hazard'].hidden = !sensor.route?.hazard;
  const hazardType = sensor.route?.hazard_type ?? '';
  if (ui['route-hazard'].dataset.kind !== hazardType) ui['route-hazard'].dataset.kind = hazardType;
  setText('route-hazard-label', {vehicle: 'Hazard vehicle', car: 'Vehicle', pedestrian: 'Pedestrian', cyclist: 'Cyclist', bus: 'Bus', barrier: 'Traffic barrier', cone: 'Traffic cone', van: 'Van'}[hazardType] ?? 'Hazard');
  const label = showingRoute
    ? 'Map ground truth showing the route and simulator actor positions'
    : 'Top-down lidar returns and radar detections around the vehicle';
  if (ui.sensor.getAttribute('aria-label') !== label) ui.sensor.setAttribute('aria-label', label);
  for (const button of sensorButtons) {
    const pressed = String(button.dataset.sensorView === sensor.view);
    if (button.getAttribute('aria-pressed') !== pressed) button.setAttribute('aria-pressed', pressed);
  }
}

async function readJson(url, signal, timeout, request = {}) {
  const response = await fetch(url, {
    ...request,
    cache: 'no-store',
    signal: AbortSignal.any([signal, AbortSignal.timeout(timeout)])
  });
  if (!response.ok) {
    const body = request.method === 'POST' ? await response.json().catch(() => ({})) : {};
    const error = new Error(body.error || `${url} returned ${response.status}`);
    error.status = response.status;
    throw error;
  }
  return response.json();
}

function frameCameras(frame) {
  return frame?.cameras ?? frame?.images ?? {};
}

function renderLiveCamera() {
  const jpeg = frameCameras(latestFrame)[selectedCamera];
  ui.stream.hidden = !jpeg;
  if (!jpeg) {
    showMessage('Waiting for a camera frame…');
    return;
  }
  const src = jpeg.startsWith('data:') ? jpeg : 'data:image/jpeg;base64,' + jpeg;
  if (ui.stream.getAttribute('src') !== src) ui.stream.src = src;
  ui.stream.alt = `Live simulation · ${cameraLabels[selectedCamera]}`;
  showMessage('');
}

function setScenarioFields(scenario) {
  ui['destination-field'].hidden = scenario !== 'trip';
  ui['delay-field'].hidden = !delayScenarios.has(scenario);
}

function installEnvironmentOptions(key, values, defaultValue, labelForId) {
  if (!Array.isArray(values) || !values.length) return;
  const field = ui[`run-${key}`];
  const choices = values.map(value => {
    const id = value?.id ?? value;
    return new Option(value?.label ?? labelForId(id), id);
  });
  field.replaceChildren(...choices);
  field.value = String(defaultValue);
  if (field.selectedIndex < 0) field.selectedIndex = 0;
  field.dataset.defaultValue = field.value;
  ui[`${key}-field`].hidden = false;
}

function environmentSettings(previous) {
  const settings = {};
  for (const key of ['weather', 'grip']) {
    const field = ui[`run-${key}`];
    if (!field.options.length) continue;
    if (previous) {
      field.value = String(previous[key] ?? field.dataset.defaultValue);
      if (field.selectedIndex < 0) field.value = field.dataset.defaultValue;
    }
    settings[key] = key === 'grip' ? Number(field.value) : field.value;
  }
  return settings;
}

function installOptions(settings) {
  if (!settings?.scenarios?.length || !settings?.destinations?.length || !settings.defaults) return;
  const toOptions = values => values.map(({label, id}) => new Option(label, id));
  options = settings;
  const defaults = settings.defaults;
  ui['run-scenario'].replaceChildren(...toOptions(settings.scenarios));
  ui['run-scenario'].value = defaults.scenario;
  ui['run-destination'].replaceChildren(...toOptions(settings.destinations));
  ui['run-destination'].value = defaults.destination;
  ui['run-seed'].value = defaults.seed;
  ui['run-delay'].value = defaults.crossing_delay;
  installEnvironmentOptions('weather', settings.weather, defaults.weather ?? 'seeded',
    id => id === 'seeded' ? 'Seeded weather' : words(id));
  installEnvironmentOptions('grip', settings.grip, defaults.grip ?? 1, id => percent.format(id));
  for (const [field, key] of [['run-seed', 'seed'], ['run-delay', 'crossing_delay']]) {
    for (const bound of ['min', 'max']) {
      if (Number.isFinite(settings.limits?.[key]?.[bound])) ui[field][bound] = settings.limits[key][bound];
    }
  }
  setScenarioFields(defaults.scenario);
  renderRunStatus();
}

function renderRunStatus() {
  const state = runStatus?.state;
  const settings = runStatus?.settings;
  const busy = submitting || state === 'queued' || Boolean(stopState);
  const canRerun = terminalStates.has(state) && delayScenarios.has(settings?.scenario) &&
    Number.isFinite(settings.crossing_delay) && settings.crossing_delay + 1 <= (Number(ui['run-delay'].max) || Infinity);
  const outcome = runStatus?.outcome;
  const outcomeCode = typeof outcome === 'string' ? outcome : outcome?.outcome ?? outcome?.outcome_code;
  const scenario = options?.scenarios.find(item => item.id === settings?.scenario)?.label;
  const runName = scenario ? `${scenario} · seed ${settings.seed}` : 'Drive';
  const outcomeLabel = outcomeCode === 'complete'
    ? 'Route complete' + (outcome?.yield_violations > 0 ? ' · Yield violation' : '')
    : words(outcomeCode);
  const messages = {
    idle: 'Ready to start a drive.',
    queued: `${runName} is queued.`,
    running: `${runName} is running.`,
    complete: outcomeCode ? `${runName} · ${outcomeLabel}` : `${runName} finished.`,
    error: runStatus?.error || `${runName} stopped with an error.`
  };
  setText('run-status', simulatorStopLabel() || (submitting ? 'Submitting drive…' : messages[state] || 'Checking simulator status…'));
  setText('run-request-error', runError ?? '');
  ui['run-request-error'].hidden = !runError;
  ui['run-start'].disabled = busy || !state;
  ui['run-stop'].disabled = busy || !state;
  ui['run-repeat'].hidden = !canRerun;
  ui['run-repeat'].disabled = busy;
}

function clearLiveFrame() {
  latestFrame = undefined;
  ui.stream.hidden = true;
  ui.stream.removeAttribute('src');
  setCameras([]);
  renderTelemetry();
  showMessage('Waiting for the first simulator frame…');
}

async function monitorStatus(signal) {
  while (!signal.aborted && stopState !== 'stopped') {
    if (stopState) {
      await new Promise(resolve => setTimeout(resolve, 500));
      continue;
    }
    const requested = submittal;
    try {
      const status = await readJson('/api/status', signal, 1800);
      if (signal.aborted) return;
      if (requested === submittal && !submitting) {
        if (status.state === 'queued' && status.run_id !== runStatus?.run_id) clearLiveFrame();
        runStatus = status;
        renderRunStatus();
      }
    } catch {
      if (signal.aborted) return;
    }
    await new Promise(resolve => setTimeout(resolve, 500));
  }
}

async function submitDrive(settings) {
  if (submitting || runStatus?.state === 'queued' || stopState) return;
  submittal++;
  submitting = true;
  runError = undefined;
  renderRunStatus();
  try {
    const result = await readJson('/api/drive', new AbortController().signal, 5000, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(settings)
    });
    runStatus = {state: 'queued', run_id: result.run_id, settings};
    if (mode === 'live') clearLiveFrame();
  } catch (error) {
    runError = error.message;
  } finally {
    submitting = false;
    renderRunStatus();
  }
}

async function stopSimulator() {
  if (!options || mode !== 'live' || !runStatus?.state || ui['run-stop'].disabled) return;
  const signal = new AbortController().signal;
  stopState = 'requesting';
  runError = undefined;
  setMode('live', '');
  renderRunStatus();
  try {
    await readJson('/api/stop', signal, 5000, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: '{}'
    });
    stopState = 'stopping';
  } catch (error) {
    stopState = '';
    runError = error.message;
    if (mode === 'live') setMode('live', latestFrame ? 'Live simulator' : 'Simulator connected');
    renderRunStatus();
    return;
  }

  // The local HTTP server closes after CARLA has cleaned up.
  let disconnectedPolls = 0;
  while (stopState === 'stopping') {
    await new Promise(resolve => setTimeout(resolve, 500));
    try {
      await readJson('/api/status', signal, 1500);
      disconnectedPolls = 0;
    } catch {
      if (++disconnectedPolls < 2) continue;
      stopState = 'stopped';
      if (mode === 'live') {
        setMode('live', '');
        if (!latestFrame) showMessage('Select Recorded drive to watch a replay.');
      }
      renderRunStatus();
    }
  }
}

async function monitorLive(signal) {
  let failedPolls = 0;
  let local = Boolean(options);
  while (!signal.aborted && stopState !== 'stopped') {
    if (stopState) {
      await new Promise(resolve => setTimeout(resolve, 500));
      continue;
    }
    try {
      const frame = await readJson('/api/frame', signal, 1800);
      if (!Number.isFinite(frame?.frame)) throw new Error('No current simulation frame');
      if (signal.aborted) return;
      failedPolls = 0;
      const cameras = frameCameras(frame);
      local = true;
      ui['source-picker'].hidden = false;
      setMode('live', 'Live simulator');
      const sameRun = frame.run_id == null || runStatus?.run_id == null || frame.run_id === runStatus.run_id;
      if (runStatus?.state !== 'queued' && !submitting && sameRun) {
        latestFrame = frame;
        setCameras(Object.keys(cameraLabels).filter(camera => typeof cameras[camera] === 'string' && cameras[camera]));
        renderLiveCamera();
        renderTelemetry(frame);
      }
    } catch (error) {
      if (signal.aborted) return;
      if (error.status === 503) {
        local = true;
        ui['source-picker'].hidden = false;
        setMode('live', 'Simulator connected');
        if (!latestFrame && !terminalStates.has(runStatus?.state)) showMessage(
          runStatus?.state === 'idle' ? 'Start a run to see the simulator cameras.' : 'Waiting for the first simulator frame…'
        );
      } else {
        failedPolls++;
        if (!local || (!options && failedPolls >= 3)) {
          if (!local) ui['source-picker'].hidden = true;
          ui.source.value = 'replay';
          await openReplay(signal);
          return;
        }
        setMode('live', 'Reconnecting');
        if (failedPolls >= 3) showMessage('The live connection was interrupted.');
      }
    }
    const interval = latestFrame && (!options || runStatus?.state === 'running') ? 100 : 500;
    await new Promise(resolve => setTimeout(resolve, interval));
  }
}

function parseReplay(recording) {
  const frames = Array.isArray(recording) ? recording : recording?.frames;
  if (!frames?.length || !frames.every(frame => Number.isFinite(frame?.frame))) {
    throw new Error('Recorded telemetry is empty or invalid');
  }
  const times = frames.map((frame, index) => {
    if (Number.isFinite(frame.video_time)) return frame.video_time;
    if (Number.isFinite(recording.fps) && recording.fps > 0) return index / recording.fps;
    return frame.sim_time - frames[0].sim_time;
  });
  if (times.some((time, index) => !Number.isFinite(time) || (index > 0 && time < times[index - 1]))) {
    throw new Error('Recorded telemetry timestamps are invalid');
  }
  return {frames, times, cameras: recording.cameras};
}

function getReplayFrame(time) {
  let low = 0;
  let high = replay.times.length - 1;
  while (low < high) {
    const middle = Math.ceil((low + high) / 2);
    if (replay.times[middle] <= time + 0.001) low = middle;
    else high = middle - 1;
  }
  return replay.frames[low];
}

function updateControls() {
  const ready = mode === 'replay' && mediaReady && Number.isFinite(ui.film.duration);
  const playing = ready && !ui.film.paused && !ui.film.ended;
  const label = playing ? 'Pause replay' : 'Play replay';
  ui.play.disabled = !ready;
  ui.play.dataset.playing = playing;
  ui.play.setAttribute('aria-label', label);
  ui.play.title = label;
  ui.seek.disabled = !ready;
  ui.loop.disabled = !ready;
  ui.seek.max = ready ? ui.film.duration : 1;
  setText('end-clock', timeLabel(ready ? ui.film.duration : NaN));
}

function syncFrame() {
  if (mode !== 'replay' || !mediaReady) return;
  const time = ui.film.currentTime;
  ui.seek.value = time;
  ui.seek.setAttribute('aria-valuetext', timeLabel(time));
  setText('clock', timeLabel(time));
  renderTelemetry(getReplayFrame(time));
}

function animatePlayback() {
  cancelAnimationFrame(animation);
  syncFrame();
  if (mode === 'replay' && !ui.film.paused && !ui.film.ended) {
    animation = requestAnimationFrame(animatePlayback);
  }
}

function loadCamera(position, resume) {
  pendingPosition = position;
  pendingPlay = resume;
  mediaReady = false;
  ui.film.pause();
  updateControls();
  showMessage('Loading ' + cameraLabels[selectedCamera].toLowerCase() + '…');
  ui.film.hidden = true;
  ui.film.setAttribute('aria-label', `${ui.encounter.selectedOptions[0].textContent} · ${cameraLabels[selectedCamera]}`);
  const movie = replay.cameraUrls?.[selectedCamera] ?? `${replay.directory}/${selectedCamera}.mp4`;
  ui.film.src = movie;
  setArtifactLink(ui['archive-mp4'], movie);
}

function loadBuild() {
  buildRequest ??= readJson('./media/build.json', new AbortController().signal, 4000).catch(() => null);
  return buildRequest;
}

function archiveHash(field, value, url) {
  const label = typeof value === 'string' ? value : '';
  const link = ui[`archive-${field}-link`];
  const code = ui[`archive-${field}-text`];
  ui[`archive-${field}-row`].hidden = !label;
  setArtifactLink(link, url);
  code.hidden = !link.hidden;
  link.textContent = code.textContent = label.slice(0, 12);
  link.title = code.title = label;
}

function clearCameraEvidence() {
  const video = ui['preview-video'];
  if (video.hasAttribute('src')) {
    video.pause();
    video.removeAttribute('src');
    video.removeAttribute('poster');
    video.load();
  }
  video.hidden = true;
  ui['preview-still'].hidden = true;
  ui['preview-still'].removeAttribute('src');
  ui['camera-evidence'].hidden = true;
  ui['sensor-panel'].hidden = false;
  ui['data-panel'].hidden = false;
  ui['capture-caption'].hidden = true;
  ui['archive-trial-row'].hidden = true;
  archiveHash('source');
  archiveHash('model');
  for (const id of ['archive-mp4', 'archive-replay', 'archive-log']) setArtifactLink(ui[id]);
  setText('archive-log', 'Run log');
}

function renderArchiveRun(build, entry, directory) {
  const run = Array.isArray(build?.runs) && build.runs.find(item => item?.recording_path === entry.path &&
    (!entry.trial_id || item.trial_id === entry.trial_id));
  const trialId = entry.trial_id || run?.trial_id;
  ui['camera-evidence'].hidden = false;
  ui['archive-trial-row'].hidden = !trialId;
  setText('archive-trial', trialId);
  archiveHash('source', run && build.source?.commit, run && build.source?.url);
  const checkpoint = build?.checkpoints?.[run?.policy];
  archiveHash('model', run?.model_sha256, checkpoint?.sha256 === run?.model_sha256 ? checkpoint?.url : undefined);
  setArtifactLink(ui['archive-replay'], entry.replay_url ?? directory + '/replay.json');
  setArtifactLink(ui['archive-log'], entry.log_url ?? run?.log_url);
}

function showPreview(build) {
  const videoUrl = artifactUrl(build?.preview_video);
  const imageUrl = previewImage(build);
  if (!videoUrl && !imageUrl) return false;
  const simulator = build.simulator?.server_version ? 'CARLA ' + build.simulator.server_version : 'CARLA';
  const title = build.preview_label || (videoUrl ? 'Training demonstration' : 'Simulator camera capture') + ' · ' + simulator;
  ui['camera-evidence'].hidden = false;
  ui['sensor-panel'].hidden = true;
  ui['data-panel'].hidden = true;
  ui['capture-caption'].hidden = false;
  const description = videoUrl ? 'Training instructor at the controls' : 'Still frame';
  setText('capture-caption', build.preview_caption || title + ' · ' + description);
  setMode('replay', videoUrl ? 'Recorded CARLA demo' : 'CARLA camera capture');
  ui.playback.hidden = true;
  showMessage('');
  ui['view-label'].hidden = true;
  ui['frame-label'].hidden = true;
  archiveHash('source', build.source?.commit, build.source?.url);
  if (build.preview_log) {
    setText('archive-log', 'Demonstration log');
    setArtifactLink(ui['archive-log'], build.preview_log);
  }
  const still = ui['preview-still'];
  if (imageUrl) {
    still.src = imageUrl;
    still.alt = build.preview_image?.alt || 'Camera frame captured inside CARLA';
  }
  if (videoUrl) {
    const video = ui['preview-video'];
    if (imageUrl) video.poster = imageUrl;
    video.src = videoUrl;
    video.hidden = false;
    setArtifactLink(ui['archive-mp4'], videoUrl);
  } else {
    still.hidden = false;
  }
  return true;
}

async function loadCatalog(signal) {
  if (catalog) return catalog;
  let entries;
  try {
    const data = await readJson('./media/catalog.json', signal, 4000);
    entries = Array.isArray(data) ? data : data.records;
  } catch {
    if (signal.aborted) return [];
    // Older releases have telemetry and video files but no catalog.
    const legacy = [
      {label: 'Pedestrian crossing', path: ''},
      {label: 'Destination trip', path: 'trip'},
      {label: 'Destination trip · delayed crossing', path: 'trip-delayed'},
      {label: 'Traffic signal', path: 'signal'},
      {label: 'Occluded pedestrian', path: 'occluded'},
      {label: 'Protected left turn', path: 'left'},
      {label: 'Intersection turn', path: 'turn'},
      {label: 'Braking vehicle', path: 'braking'}
    ];
    const available = await Promise.all(legacy.map(async entry => {
      try {
        const folder = entry.path ? `./media/${entry.path}` : './media';
        const response = await fetch(`${folder}/replay.json`, {
          method: 'HEAD',
          signal: AbortSignal.any([signal, AbortSignal.timeout(3000)])
        });
        return response.ok ? entry : null;
      } catch {
        return null;
      }
    }));
    entries = available.filter(Boolean);
  }
  if (signal.aborted || !Array.isArray(entries)) return [];
  try {
    const view = await readJson('./media/viewer-catalog.json', signal, 4000);
    const extra = Array.isArray(view) ? view : view?.records ?? [];
    const supplement = new Map(extra.map(item => [item.path, item]));
    const basePaths = new Set(entries.map(item => item.path));
    entries = entries.map(item => ({...item, ...supplement.get(item.path)}))
      .concat(extra.filter(item => !basePaths.has(item.path)));
  } catch {
    if (signal.aborted) return [];
  }
  const paths = new Set();
  catalog = entries.flatMap(entry => {
    if (typeof entry?.label !== 'string' || !entry.label.trim() || !validRecordingPath(entry.path) || paths.has(entry.path)) return [];
    paths.add(entry.path);
    return [entry];
  });
  return catalog;
}

async function openReplay(signal, trialId) {
  ui.stream.hidden = true;
  ui.stream.removeAttribute('src');
  clearCameraEvidence();
  setMode('replay', 'Loading CARLA replay');
  replay = undefined;
  resetPlayback();
  setCameras([]);
  renderTelemetry();
  showMessage('Loading recorded drive…');
  const [entries, build] = await Promise.all([loadCatalog(signal), loadBuild()]);
  if (signal.aborted) return;
  const selectedPath = ui.encounter.value;
  const requested = trialId && Array.isArray(build?.runs) && build.runs.find(run => run?.trial_id === trialId);
  const entry = trialId
    ? entries.find(item => item.trial_id === trialId || (!item.trial_id && item.path === requested?.recording_path))
    : entries.find(item => item.path === selectedPath) ?? entries[0];
  ui.encounter.replaceChildren(...entries.map(item => new Option(item.label, item.path)));
  ui['encounter-picker'].hidden = !entries.length;
  if (!entry) {
    ui.encounter.selectedIndex = -1;
    if (!trialId && showPreview(build)) return;
    setMode('replay', 'Replay unavailable');
    ui.playback.hidden = true;
    showMessage(trialId ? 'The requested CARLA replay is unavailable.' : 'CARLA recordings are awaiting publication.');
    return;
  }
  ui.encounter.value = entry.path;
  const directory = entry.path ? `./media/${entry.path}` : './media';
  try {
    const recording = recordings.get(directory) ??
      await loadReplay(entry.replay_pack_url, signal, parseReplay,
        () => readJson(entry.replay_url ?? `${directory}/replay.json`, signal, 30000));
    if (signal.aborted) return;
    recordings.set(directory, recording);
    replay = {...recording, directory, cameraUrls: entry.camera_urls};
  } catch {
    if (signal.aborted) return;
    setMode('replay', 'Replay unavailable');
    showMessage('Recorded telemetry for this encounter is unavailable.');
    return;
  }
  setMode('replay', 'Recorded CARLA drive');
  renderArchiveRun(build, entry, directory);
  const cameras = entry.cameras ?? replay.cameras;
  setCameras(Array.isArray(cameras) ? Object.keys(cameraLabels).filter(camera => cameras.includes(camera)) : Object.keys(cameraLabels));
  if (!availableCameras.length) {
    showMessage('No camera tracks are listed for this recording.');
    return;
  }
  loadCamera(0, false);
}

function connectionSignal() {
  sourceControl?.abort();
  sourceControl = new AbortController();
  return sourceControl.signal;
}

function resetPlayback() {
  mediaReady = false;
  pendingPosition = 0;
  pendingPlay = false;
  ui.film.pause();
  ui.film.hidden = true;
  ui.film.removeAttribute('src');
  ui.film.load();
  setText('clock', timeLabel(0));
  ui.seek.value = 0;
  ui.seek.setAttribute('aria-valuetext', timeLabel(0));
  updateControls();
}

async function connect() {
  const signal = connectionSignal();
  if (ui.source.value === 'replay') {
    openReplay(signal);
  } else {
    setMode('', 'Connecting');
    clearCameraEvidence();
    resetPlayback();
    ui.stream.hidden = true;
    ui.stream.removeAttribute('src');
    setCameras([]);
    renderTelemetry();
    if (stopState === 'stopped') {
      setMode('live', '');
      showMessage('Select Recorded drive to watch a replay.');
      return;
    }
    showMessage('Connecting to the drive…');
    try {
      const settings = await readJson('/api/options', signal, 1800);
      if (signal.aborted) return;
      if (!options) installOptions(settings);
    } catch {
      if (signal.aborted) return;
    }
    if (options) {
      ui['source-picker'].hidden = false;
      setMode('live', 'Simulator connected');
      monitorStatus(signal);
    }
    monitorLive(signal);
  }
}

for (const button of cameraButtons) {
  button.addEventListener('click', () => {
    if (button.dataset.camera === selectedCamera) return;
    selectedCamera = button.dataset.camera;
    setCameras(availableCameras);
    if (mode === 'live') renderLiveCamera();
    if (mode === 'replay') {
      const position = mediaReady ? ui.film.currentTime : pendingPosition;
      const resume = mediaReady ? !ui.film.paused && !ui.film.ended : pendingPlay;
      loadCamera(position, resume);
    }
  });
}

ui.source.addEventListener('change', connect);
ui.encounter.addEventListener('change', () => {
  const entry = catalog?.find(item => item.path === ui.encounter.value);
  const url = new URL(location.href);
  if (entry?.trial_id) url.searchParams.set('run', entry.trial_id);
  else url.searchParams.delete('run');
  history.replaceState(history.state, '', url);
  openReplay(connectionSignal());
});
ui['run-stop'].addEventListener('click', stopSimulator);
for (const button of sensorButtons) {
  button.addEventListener('click', () => {
    sensor.setView(button.dataset.sensorView);
    renderSensorPanel();
  });
}
ui['run-scenario'].addEventListener('change', () => setScenarioFields(ui['run-scenario'].value));
ui['run-form'].addEventListener('submit', event => {
  event.preventDefault();
  const scenario = ui['run-scenario'].value;
  submitDrive({
    scenario,
    seed: ui['run-seed'].valueAsNumber,
    crossing_delay: delayScenarios.has(scenario) ? ui['run-delay'].valueAsNumber : options.defaults.crossing_delay,
    destination: ui['run-destination'].value,
    ...environmentSettings()
  });
});
ui['run-repeat'].addEventListener('click', () => {
  const previous = runStatus.settings;
  const {scenario, seed, crossing_delay, destination} = previous;
  const delay = Number((crossing_delay + 1).toFixed(6));
  ui['run-scenario'].value = scenario;
  ui['run-seed'].value = seed;
  ui['run-delay'].value = delay;
  ui['run-destination'].value = destination ?? options.defaults.destination;
  setScenarioFields(scenario);
  submitDrive({
    scenario, seed, crossing_delay: delay, destination: ui['run-destination'].value,
    ...environmentSettings(previous)
  });
});
ui.stream.addEventListener('load', renderDetections);
new ResizeObserver(renderDetections).observe(ui.viewport);
ui.film.addEventListener('loadedmetadata', () => {
  if (mode !== 'replay' || !replay || !ui.film.hasAttribute('src')) return;
  ui.film.currentTime = Math.min(pendingPosition, ui.film.duration);
});
ui.film.addEventListener('canplay', () => {
  if (mode !== 'replay' || !replay || mediaReady || !ui.film.hasAttribute('src')) return;
  mediaReady = true;
  ui.film.hidden = false;
  showMessage('');
  syncFrame();
  updateControls();
  if (pendingPlay) ui.film.play().catch(updateControls);
});
ui.film.addEventListener('error', () => {
  if (mode !== 'replay' || !replay || !ui.film.hasAttribute('src')) return;
  const position = mediaReady ? ui.film.currentTime : pendingPosition;
  const resume = mediaReady ? !ui.film.paused && !ui.film.ended : pendingPlay;
  setCameras(availableCameras.filter(camera => camera !== selectedCamera));
  if (availableCameras.length) {
    loadCamera(position, resume);
    return;
  }
  resetPlayback();
  setArtifactLink(ui['archive-mp4']);
  renderTelemetry();
  setMode('replay', 'Replay unavailable');
  showMessage('The recorded camera files could not be played.');
});
ui['preview-video'].addEventListener('error', () => {
  const video = ui['preview-video'];
  if (video.hidden || !video.hasAttribute('src')) return;
  video.hidden = true;
  setArtifactLink(ui['archive-mp4']);
  const still = ui['preview-still'];
  ui['capture-caption'].hidden = true;
  if (still.hasAttribute('src')) {
    still.hidden = false;
    setText('capture-caption', 'CARLA camera capture · Still frame');
    ui['capture-caption'].hidden = false;
    setMode('replay', 'CARLA camera capture');
  } else {
    setMode('replay', 'Replay unavailable');
    showMessage('The demonstration video could not be played.');
  }
  ui.playback.hidden = true;
});
ui['preview-still'].addEventListener('error', () => {
  if (ui['preview-still'].hidden) return;
  ui['preview-still'].hidden = true;
  ui['capture-caption'].hidden = true;
  setMode('replay', 'Capture unavailable');
  ui.playback.hidden = true;
  showMessage('The CARLA camera capture could not be displayed.');
});
for (const event of ['play', 'pause', 'ended', 'durationchange']) {
  ui.film.addEventListener(event, () => {
    updateControls();
    animatePlayback();
  });
}
ui.film.addEventListener('seeked', syncFrame);
ui.film.addEventListener('timeupdate', syncFrame);
ui.play.addEventListener('click', () => {
  if (!ui.film.paused && !ui.film.ended) {
    ui.film.pause();
    return;
  }
  if (ui.film.ended) ui.film.currentTime = 0;
  ui.film.play().catch(updateControls);
});
ui.loop.addEventListener('click', () => {
  ui.film.loop = !ui.film.loop;
  ui.loop.setAttribute('aria-pressed', ui.film.loop);
});
ui.seek.addEventListener('input', () => {
  ui.film.currentTime = Number(ui.seek.value);
  syncFrame();
});
const trialId = new URLSearchParams(location.search).get('run');
if (trialId) {
  ui.source.value = 'replay';
  const signal = connectionSignal();
  openReplay(signal, trialId);
  readJson('/api/options', signal, 1800).then(settings => {
    if (signal.aborted) return;
    installOptions(settings);
    if (options) ui['source-picker'].hidden = false;
  }).catch(() => {});
} else {
  connect();
}
