import {SensorView} from './sensor.js';

const ids = [
  'source', 'source-state', 'encounter-picker', 'encounter', 'viewport', 'film', 'stream',
  'detections', 'camera-empty', 'view-label', 'frame-label',
  'playback', 'play', 'clock', 'seek', 'end-clock', 'live-strip', 'live-clock', 'ir-note', 'front-note',
  'sensor', 'radar-reading', 'lidar-count', 'radar-count', 'map', 'frame', 'speed',
  'steer', 'steer-direction', 'throttle', 'brake',
  'progress', 'route-total', 'lateral', 'collisions', 'scenario', 'policy', 'seed', 'phase', 'weather'
];
const ui = Object.fromEntries(ids.map(id => [id, document.getElementById(id)]));
const cameraButtons = [...document.querySelectorAll('[data-camera]')];
const cameraLabels = {chase: 'Chase camera', front: 'Front camera', infrared: 'Thermal proxy'};
const policyLabels = {fusion: 'Learned multisensor', camera: 'Learned vision'};
const replayDirectories = {pedestrian: 'media', turn: 'media/turn', braking: 'media/braking'};
const recordings = new Map();
const sensor = new SensorView(ui.sensor, ui['radar-reading']);
const integer = new Intl.NumberFormat(undefined, {maximumFractionDigits: 0});
const decimal = new Intl.NumberFormat(undefined, {minimumFractionDigits: 1, maximumFractionDigits: 1});
const precision = new Intl.NumberFormat(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
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

function setText(id, value) {
  ui[id].textContent = value ?? '—';
}

function setMode(nextMode, label) {
  mode = nextMode;
  ui['source-state'].dataset.mode = mode;
  setText('source-state', label);
  ui['encounter-picker'].hidden = mode !== 'replay';
  ui.playback.hidden = mode === 'live';
  ui['live-strip'].hidden = mode !== 'live';
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
  const detections = renderedFrame?.detections;
  const visible = selectedCamera === 'front' && ui['camera-empty'].hidden;
  const width = mode === 'live' ? ui.stream.naturalWidth : ui.film.videoWidth;
  const height = mode === 'live' ? ui.stream.naturalHeight : ui.film.videoHeight;
  const show = visible && Array.isArray(detections) && width && height;
  overlay.hidden = !show;
  ui['front-note'].hidden = !show;
  if (!show) return;
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
    element.style.left = `${x1 * 100}%`;
    element.style.top = `${y1 * 100}%`;
    element.style.width = `${(x2 - x1) * 100}%`;
    element.style.height = `${(y2 - y1) * 100}%`;
    const caption = document.createElement('span');
    caption.className = 'detection-label';
    caption.textContent = Number.isFinite(score) ? `${label} ${Math.round(score * 100)}%` : label;
    element.append(caption);
    boxes.push(element);
  }
  overlay.replaceChildren(...boxes);
}

function renderTelemetry(frame) {
  if (frame === renderedFrame && frame) return;
  renderedFrame = frame;
  const lidar = frame?.lidar ?? [];
  const radar = frame?.radar ?? [];
  setText('lidar-count', frame ? numeric(lidar.length) : '—');
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
  setText('live-clock', timeLabel(frame?.sim_time));
  sensor.update(lidar, radar);
  renderDetections();
}

async function readJson(url, signal, timeout) {
  const response = await fetch(url, {
    cache: 'no-store',
    signal: AbortSignal.any([signal, AbortSignal.timeout(timeout)])
  });
  if (!response.ok) {
    const error = new Error(`${url} returned ${response.status}`);
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

async function monitorLive(signal) {
  let failedPolls = 0;
  while (!signal.aborted) {
    try {
      const frame = await readJson('/api/frame', signal, 1800);
      if (!Number.isFinite(frame?.frame)) throw new Error('No current simulation frame');
      if (signal.aborted) return;
      failedPolls = 0;
      latestFrame = frame;
      const cameras = frameCameras(frame);
      setMode('live', 'Live simulator');
      setCameras(Object.keys(cameraLabels).filter(camera => typeof cameras[camera] === 'string' && cameras[camera]));
      renderLiveCamera();
      renderTelemetry(frame);
    } catch (error) {
      if (signal.aborted) return;
      if (error.status === 503) {
        setMode(mode, 'Waiting for simulator');
        if (mode !== 'live') showMessage('Waiting for the first simulator frame…');
      } else {
        failedPolls++;
        if (mode !== 'live' || failedPolls >= 3) {
          await openReplay(signal);
          return;
        }
        setMode('live', 'Reconnecting');
      }
    }
    await new Promise(resolve => setTimeout(resolve, mode === 'live' ? 100 : 350));
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
  return {frames, times};
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
  ui.film.src = `${replay.directory}/${selectedCamera}.mp4`;
}

async function openReplay(signal) {
  const directory = replayDirectories[ui.encounter.value];
  ui.stream.hidden = true;
  ui.stream.removeAttribute('src');
  setMode('replay', 'Loading replay');
  replay = undefined;
  resetPlayback();
  setCameras([]);
  renderTelemetry();
  showMessage('Loading recorded drive…');
  try {
    const recording = recordings.get(directory) ??
      parseReplay(await readJson(`${directory}/replay.json`, signal, 15000));
    if (signal.aborted) return;
    recordings.set(directory, recording);
    replay = {...recording, directory};
  } catch {
    if (signal.aborted) return;
    setMode('replay', 'Replay unavailable');
    showMessage('Recorded telemetry for this encounter is unavailable.');
    return;
  }
  setMode('replay', 'Recorded drive');
  setCameras(Object.keys(cameraLabels));
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

function connect() {
  const signal = connectionSignal();
  if (ui.source.value === 'replay') {
    openReplay(signal);
  } else {
    setMode('', 'Connecting');
    resetPlayback();
    ui.stream.hidden = true;
    ui.stream.removeAttribute('src');
    setCameras([]);
    renderTelemetry();
    showMessage('Connecting to the drive…');
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
ui.encounter.addEventListener('change', () => openReplay(connectionSignal()));
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
  renderTelemetry();
  setMode('replay', 'Replay unavailable');
  showMessage('The recorded camera files could not be played.');
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
ui.seek.addEventListener('input', () => {
  ui.film.currentTime = Number(ui.seek.value);
  syncFrame();
});
connect();
