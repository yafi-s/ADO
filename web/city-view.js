import * as T from './vendor/city-view/three-view.js';

const el = id => document.getElementById(id);
const finite = Number.isFinite;
const point = (p, n = 2) => Array.isArray(p) && p.length >= n && p.slice(0, n).every(finite);
const radians = Math.PI / 180;
const vec = (x, y, z) => new T.Vector3(x, y, z);
const read = async (url, signal, timeout = 12000) => {
  const result = await fetch(url, {cache: 'no-cache', signal: AbortSignal.any([signal, AbortSignal.timeout(timeout)])});
  if (!result.ok) throw new Error(String(result.status));
  return result.json();
};
function issue(id, message = '') {
  el(id).textContent = message;
  el(id).hidden = !message;
}
const colorMaterial = color => new T.MeshLambertMaterial({color});

// Night and day palettes for the map, matching the display mode chosen in the status bar.
// Only colour and light levels differ; the geometry and the telemetry it shows do not.
const PALETTES = {
  night: {
    sky: 0x16212f, ground: 0x1a2533, curb: 0x56667c, road: 0x3a4a60,
    buildings: [0x48586e, 0x415166, 0x506077, 0x3b4a5f],
    route: 0x45e2c7, marker: 0x45e2c7, vehicle: 0x9caefa, walker: 0xff9d9b,
    shadow: 0x000000, shadowOpacity: .42, locatorRing: 0x0c3d38, locator: 0x7ff3df,
    hemiSky: 0xdbe6ff, hemiGround: 0x3a4658, hemi: 1.7, sun: 1.3, environment: .6, exposure: 1.0
  },
  day: {
    sky: 0xeceef2, ground: 0xdde1e7, curb: 0xc9cfd8, road: 0xf7f8fa,
    buildings: [0xe3e6eb, 0xd9dde4, 0xe8eaee, 0xdfe2e8],
    route: 0x2563eb, marker: 0x2563eb, vehicle: 0x9fa6b2, walker: 0xe2702a,
    shadow: 0x1d2633, shadowOpacity: .26, locatorRing: 0xffffff, locator: 0x2563eb,
    hemiSky: 0xffffff, hemiGround: 0x9aa3b0, hemi: 2.0, sun: 1.7, environment: .75, exposure: 1.0
  }
};
const displayMode = () => document.documentElement.dataset.display === 'day' ? 'day' : 'night';
function mesh(geometry, material, group) {
  const item = new T.Mesh(geometry, material);
  group.add(item);
  return item;
}

// Each input polyline is one lane center, sampled in metres. Build joined ribbons
// in CARLA XY and carry the waypoint elevation into the Three.js Y coordinate.
function ribbon(lines, width, up, material) {
  const positions = [], indices = [];
  for (const line of lines) {
    const base = positions.length / 3;
    for (let i = 0; i < line.length; i++) {
      const [x, y, h] = line[i], a = line[Math.max(0, i - 1)], b = line[Math.min(i + 1, line.length - 1)];
      const dx = b[0] - a[0], dy = b[1] - a[1], scale = width / (2 * (Math.hypot(dx, dy) || 1));
      positions.push(x - dy * scale, h + up, y + dx * scale, x + dy * scale, h + up, y - dx * scale);
      if (i) {
        const j = base + 2 * i;
        indices.push(j - 2, j, j - 1, j - 1, j, j + 1);
      }
    }
  }
  const geometry = new T.BufferGeometry();
  geometry.setAttribute('position', new T.Float32BufferAttribute(positions, 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  material.side = T.DoubleSide;
  return new T.Mesh(geometry, material);
}

class CityScene {
  constructor() {
    const canvas = el('scene');
    this.renderer = new T.WebGLRenderer({canvas, antialias: true, powerPreference: 'low-power'});
    this.renderer.toneMapping = T.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.05;
    this.renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 1.6));
    this.scene = new T.Scene();
    this.look = PALETTES[displayMode()];
    this.scene.background = new T.Color(this.look.sky);
    this.scene.fog = new T.Fog(this.look.sky, 350, 2400);
    this.camera = new T.PerspectiveCamera(46, 1, .15, 3500);
    this.controls = new T.OrbitControls(this.camera, canvas);
    this.dirty = true;
    this.controls.addEventListener('change', () => { this.dirty = true; });
    this.controls.enablePan = false;
    this.controls.maxPolarAngle = Math.PI * .48;
    this.controls.minDistance = 4.5;
    this.controls.maxDistance = 1600;
    this.controls.target.set(0, 0, 0);
    this.camera.position.set(-10, 7, 9);
    this.controls.update();
    this.hemisphere = new T.HemisphereLight(0xfff9e9, 0x748477, 2.05);
    this.scene.add(this.hemisphere);
    const sunlight = this.sunlight = new T.DirectionalLight(0xfff4d9, 1.8);
    sunlight.position.set(-70, 100, -40);
    this.scene.add(sunlight);
    const room = new T.RoomEnvironment();
    const pmrem = new T.PMREMGenerator(this.renderer);
    this.scene.environment = pmrem.fromScene(room, .05).texture;
    pmrem.dispose();
    room.dispose();
    this.scene.environmentIntensity = .75;
    this.floor = mesh(new T.PlaneGeometry(6000, 6000), colorMaterial(0x93a58e), this.scene);
    this.floor.rotation.x = -Math.PI / 2;
    this.floor.position.y = -.14;
    this.ego = new T.Group();
    this.ego.visible = false;
    this.scene.add(this.ego);
    const shadow = this.shadow = mesh(new T.CylinderGeometry(1, 1, .015, 48), new T.MeshBasicMaterial({
      color: 0x1c2b25, transparent: true, opacity: .31, depthWrite: false
    }), this.ego);
    shadow.scale.set(2.54, 1, 1.13);
    shadow.position.y = .02;
    this.marker = mesh(new T.CylinderGeometry(.46, .64, .25, 16), colorMaterial(0x8af3be), this.ego);
    this.marker.position.y = .2;
    this.overhead = mesh(new T.CylinderGeometry(1, 1, .04, 32), new T.MeshBasicMaterial({
      color: 0x154c47, depthTest: false, toneMapped: false
    }), this.ego);
    const locator = this.locator = mesh(new T.CylinderGeometry(.60, .60, .05, 32), new T.MeshBasicMaterial({
      color: 0xc1ff8c, depthTest: false, toneMapped: false
    }), this.overhead);
    this.overhead.renderOrder = 10;
    locator.renderOrder = 11;
    this.overhead.visible = false;
    this.mode = 'follow';
    this.grid = new Map();
    this.models = new Map();
    this.actors = [];
    this.vehicleMat = colorMaterial(0xea9e60);
    this.walkerMat = colorMaterial(0xf87c99);
    // Ground-truth actors arrive as positions, often without a heading, so their markers stay symmetric.
    this.actorBody = new T.RoundedBoxGeometry(3.9, 1.12, 1.78, 3, .3);
    this.actorRoof = new T.RoundedBoxGeometry(2.05, .52, 1.58, 3, .2);
    this.walkerBody = new T.CapsuleGeometry(.2, .92, 5, 14);
    this.walkerHead = new T.SphereGeometry(.17, 16, 12);
    this.frameCamera = true;
    this.applyPalette();
    addEventListener('ado:display', () => this.applyPalette());
    new ResizeObserver(() => {
      const w = canvas.clientWidth, h = canvas.clientHeight;
      this.dirty = true;
      this.renderer.setSize(w, h, false);
      this.camera.aspect = w / Math.max(1, h);
      this.camera.updateProjectionMatrix();
      if (this.mode === 'overview') this.positionCamera();
    }).observe(canvas);
  }

  applyPalette() {
    const look = this.look = PALETTES[displayMode()];
    this.scene.background.set(look.sky);
    this.scene.fog.color.set(look.sky);
    this.renderer.toneMappingExposure = look.exposure;
    this.scene.environmentIntensity = look.environment;
    this.hemisphere.color.set(look.hemiSky);
    this.hemisphere.groundColor.set(look.hemiGround);
    this.hemisphere.intensity = look.hemi;
    this.sunlight.intensity = look.sun;
    this.floor.material.color.set(look.ground);
    this.shadow.material.color.set(look.shadow);
    this.shadow.material.opacity = look.shadowOpacity;
    this.marker.material.color.set(look.marker);
    this.overhead.material.color.set(look.locatorRing);
    this.locator.material.color.set(look.locator);
    this.vehicleMat.color.set(look.vehicle);
    this.walkerMat.color.set(look.walker);
    this.curbs?.material.color.set(look.curb);
    this.roads?.material.color.set(look.road);
    this.route?.material.color.set(look.route);
    if (this.buildings) {
      const tint = new T.Color();
      for (let i = 0; i < this.buildings.count; i++) this.buildings.setColorAt(i, tint.set(look.buildings[i % look.buildings.length]));
      this.buildings.instanceColor.needsUpdate = true;
    }
    this.dirty = true;
  }

  async loadTown(url = './media/town.json', mapName = null) {
    if ((this.mapUrl === url && this.mapName === mapName) ||
        (this.mapRequest?.url === url && this.mapRequest.mapName === mapName)) return;
    this.mapRequest?.control.abort();
    const request = {url, mapName, control: new AbortController()};
    this.mapRequest = request;
    this.mapUrl = this.mapName = this.bounds = null;
    for (const name of ['curbs', 'roads', 'buildings']) {
      const geometry = this[name];
      if (!geometry) continue;
      this.scene.remove(geometry);
      geometry.geometry.dispose();
      geometry.material.dispose();
      geometry.dispose?.();
      this[name] = null;
    }
    this.grid.clear();
    this.floor.position.y = -.16;
    this.routeKey = '';
    if (this.lastFrame) this.show(this.lastFrame);
    this.dirty = true;
    issue('map-issue');
    try {
      if (typeof url !== 'string' || !/^\.\/media\/(?:[\w-]+\/)*[\w-]+\.json$/i.test(url)) throw new Error('Invalid road map path');
      const data = await read(url, request.control.signal, 30000);
      if (this.mapRequest !== request) return;
      if (mapName && data.producer?.map !== mapName) throw new Error('Road geometry does not match the recording');
      const roads = (data.roads || []).map(lane => (Array.isArray(lane) ? lane : lane.points || []).filter(p => point(p, 3))).filter(lane => lane.length > 1);
      if (!roads.length) throw new Error('No lanes');
      for (const road of roads) for (let i = 1; i < road.length; i++) {
        const a = road[i - 1], b = road[i];
        const lowX = Math.floor((Math.min(a[0], b[0]) - 8) / 16), highX = Math.floor((Math.max(a[0], b[0]) + 8) / 16);
        const lowY = Math.floor((Math.min(a[1], b[1]) - 8) / 16), highY = Math.floor((Math.max(a[1], b[1]) + 8) / 16);
        for (let x = lowX; x <= highX; x++) for (let y = lowY; y <= highY; y++) {
          const key = x + ',' + y;
          if (!this.grid.has(key)) this.grid.set(key, []);
          this.grid.get(key).push([a, b]);
        }
      }
      const bounds = new T.Box3();
      for (const lane of roads) for (const [x, y, z] of lane) bounds.expandByPoint(vec(x, z, y));
      this.bounds = bounds;
      this.floor.position.y = bounds.min.y - .16;
      this.curbs = ribbon(roads, 4.65, -.065, colorMaterial(this.look.curb));
      this.roads = ribbon(roads, 3.53, 0, colorMaterial(this.look.road));
      this.scene.add(this.curbs, this.roads);
      const boxes = (data.buildings || []).filter(b => point(b.center, 3) && point(b.extent, 3) && b.extent.every(v => v > 0));
      if (boxes.length) {
        const buildings = new T.InstancedMesh(new T.BoxGeometry(), colorMaterial(0xffffff), boxes.length);
        const dummy = new T.Object3D(), palettes = this.look.buildings;
        boxes.forEach((b, i) => {
          const [x, y, z] = b.center, [sx, sy, sz] = b.extent;
          dummy.position.set(x, z, y);
          dummy.rotation.y = -(b.yaw || 0) * radians;
          dummy.scale.set(2 * sx, 2 * sz, 2 * sy);
          dummy.updateMatrix();
          buildings.setMatrixAt(i, dummy.matrix);
          buildings.setColorAt(i, new T.Color(palettes[i % palettes.length]));
        });
        this.buildings = buildings;
        this.scene.add(buildings);
      }
      this.dirty = true;
      this.mapUrl = url;
      this.mapName = mapName;
      this.routeKey = '';
      if (this.lastFrame) this.show(this.lastFrame);
      if (this.mode === 'overview' || !this.ego.visible) this.positionCamera(true);
    } catch {
      if (!request.control.signal.aborted) issue('map-issue', 'City footprint unavailable. Vehicle and route use telemetry coordinates.');
    } finally {
      if (this.mapRequest === request) this.mapRequest = null;
    }
  }

  height(x, y) {
    const list = this.grid.get(Math.floor(x / 16) + ',' + Math.floor(y / 16)) || [];
    let best = Infinity, z = 0;
    for (const [a, b] of list) {
      const dx = b[0] - a[0], dy = b[1] - a[1];
      const t = Math.max(0, Math.min(1, ((x - a[0]) * dx + (y - a[1]) * dy) / (dx * dx + dy * dy || 1)));
      const distance = (a[0] + t * dx - x) ** 2 + (a[1] + t * dy - y) ** 2;
      if (distance < best) { best = distance; z = a[2] + t * (b[2] - a[2]); }
    }
    return z;
  }

  async selectModel(name) {
    this.dirty = true;
    this.selected = name;
    if (this.car) this.ego.remove(this.car);
    this.car = null;
    this.marker.visible = true;
    issue('model-issue', 'Loading vehicle designâ€¦');
    try {
      if (!this.models.has(name)) {
        const pending = new T.GLTFLoader().loadAsync('./vehicle-assets/' + name + '.glb').then(({scene}) => {
          const bounds = new T.Box3().setFromObject(scene);
          if (bounds.isEmpty()) throw new Error('Empty model');
          scene.position.y -= bounds.min.y;
          return scene;
        }).catch(error => { this.models.delete(name); throw error; });
        this.models.set(name, pending);
      }
      const model = await this.models.get(name);
      if (name !== this.selected) return;
      this.dirty = true;
      this.car = model;
      this.ego.add(model);
      this.marker.visible = false;
      issue('model-issue');
    } catch {
      if (name === this.selected) issue('model-issue', 'Vehicle design unavailable. The green marker shows its position.');
    }
  }

  clear() {
    this.dirty = true;
    this.lastFrame = null;
    this.ego.visible = false;
    this.frameCamera = true;
    this.actors.forEach(a => { a.visible = false; });
    if (this.route) this.route.visible = false;
  }

  drawActors(map) {
    const actors = (map.others || []).filter(a => a && finite(a.x) && finite(a.y));
    for (const [p, type] of [[map.traffic, 'vehicle'], [map.hazard, map.hazard_type === 'pedestrian' ? 'walker' : 'vehicle']]) {
      if (point(p) && !actors.some(a => Math.hypot(p[0] - a.x, p[1] - a.y) < .7)) actors.push({x:p[0], y:p[1], type});
    }
    actors.forEach((a, i) => {
      const walker = a.type === 'walker' || a.type === 'pedestrian';
      let marker = this.actors[i];
      if (!marker || marker.userData.walker !== walker) {
        if (marker) this.scene.remove(marker);
        marker = new T.Group();
        marker.userData.walker = walker;
        const body = mesh(walker ? this.walkerBody : this.actorBody, walker ? this.walkerMat : this.vehicleMat, marker);
        body.position.y = walker ? .68 : .70;
        const top = mesh(walker ? this.walkerHead : this.actorRoof, walker ? this.walkerMat : this.vehicleMat, marker);
        top.position.y = walker ? 1.56 : 1.39;
        this.actors[i] = marker;
        this.scene.add(marker);
      }
      // Legacy actor telemetry supplies XY only. Use a square footprint to avoid implying a heading.
      marker.scale.set(!walker && !finite(a.yaw) ? .55 : 1, 1, 1);
      marker.rotation.y = -(finite(a.yaw) ? a.yaw : 0) * radians;
      marker.position.set(a.x, this.height(a.x, a.y) + .025, a.y);
      marker.visible = true;
    });
    this.actors.slice(actors.length).forEach(a => { a.visible = false; });
  }

  show(frame) {
    const map = frame?.observer_map;
    if (!point(map?.ego, 3)) { this.clear(); return; }
    this.lastFrame = frame;
    const path = Array.isArray(map.path) ? map.path : [];
    const routeKey = JSON.stringify(path);
    if (routeKey !== this.routeKey) {
      if (this.route) {
        this.scene.remove(this.route);
        this.route.geometry.dispose();
        this.route.material.dispose();
      }
      const points = path.filter(p => point(p)).map(([x,y]) => [x,y,this.height(x,y)]);
      this.route = ribbon([points], .32, .055, new T.MeshBasicMaterial({color:this.look.route}));
      this.routeKey = routeKey;
      this.scene.add(this.route);
    }
    if (this.route) this.route.visible = true;
    this.drawActors(map);
    this.pose(map.ego, map.ego_z);
  }

  pose([x, y, yaw], height) {
    this.dirty = true;
    const previous = this.ego.position.clone(), oldAngle = this.ego.rotation.y;
    const road = this.height(x,y);
    this.ego.position.set(x, finite(height) ? Math.max(road + .015, height) : road + .025, y);
    this.ego.rotation.y = -yaw * radians;
    this.ego.visible = true;
    if (this.frameCamera) { this.positionCamera(); this.frameCamera = false; return; }
    if (this.mode !== 'follow') return;
    const delta = this.ego.position.clone().sub(previous);
    const offset = this.camera.position.clone().sub(this.controls.target);
    offset.applyAxisAngle(vec(0,1,0), this.ego.rotation.y - oldAngle);
    this.controls.target.add(delta);
    this.camera.position.copy(this.controls.target).add(offset);
    this.controls.update();
  }

  positionCamera(forceOverview = false) {
    this.dirty = true;
    this.overhead.visible = this.mode === 'overview';
    this.camera.near = this.mode === 'overview' ? 2 : .15;
    this.camera.updateProjectionMatrix();
    if ((this.mode === 'overview' || forceOverview) && this.bounds) {
      const size = this.bounds.getSize(vec());
      this.bounds.getCenter(this.controls.target);
      const radius = Math.max(size.x, size.z) * .65 / Math.min(1, this.camera.aspect);
      this.camera.position.copy(this.controls.target).add(vec(0, 2.1 * radius, .59 * radius));
      this.overhead.scale.setScalar(radius * .025);
    } else if (this.ego.visible) {
      this.controls.target.copy(this.ego.position).add(vec(0, 1, 0));
      this.camera.position.copy(this.controls.target).add(vec(-8.3, 4.7, 6.9).applyAxisAngle(vec(0,1,0), this.ego.rotation.y));
    }
    this.controls.update();
  }
}

let scene;
try {
  scene = new CityScene();
  scene.loadTown();
  scene.selectModel(el('design').value);
} catch {
  issue('renderer-issue', '3D rendering is unavailable. Enable WebGL 2 in your browser to see the city.');
}
el('design').addEventListener('change', () => scene?.selectModel(el('design').value));
for (const name of ['follow', 'overview']) {
  el(name).addEventListener('click', () => {
    if (!scene) return;
    scene.mode = name;
    scene.positionCamera();
    for (const id of ['follow', 'overview']) el(id).setAttribute('aria-pressed', String(id === name));
  });
}

let catalog = [], recording = null, position = 0, playing = false, mode = '', current = -1;
let replayRequest, monitorRequest, activeLiveId, sourceSerial = 0;
let requestedCase = new URLSearchParams(location.search).get('case');
const requestedDesign = new URLSearchParams(location.search).get('design');
if (['waymo-ipace','meridian'].includes(requestedDesign) && requestedDesign !== el('design').value) {
  el('design').value = requestedDesign;
  scene?.selectModel(requestedDesign);
}
function setMode(value, title) {
  mode = value;
  el('mode').dataset.mode = value;
  el('mode').textContent = title;
  el('case-label').hidden = value === 'live' || !catalog.length;
}
function fmtTime(value) {
  const seconds = Math.max(0, Math.floor(value));
  return Math.floor(seconds / 60) + ':' + String(seconds % 60).padStart(2, '0');
}
function controls() {
  const ready = Boolean(recording && mode === 'recorded');
  el('play').disabled = !ready;
  el('seek').disabled = !ready;
  el('play').textContent = playing ? 'Pause' : 'Play';
  el('play').setAttribute('aria-label', playing ? 'Pause recording' : 'Play recording');
  el('seek').max = ready ? recording.end : 1;
  el('seek').value = ready ? position : 0;
  el('end').textContent = ready ? fmtTime(recording.end) : 'â€“:â€“â€“';
  if (ready) el('clock').textContent = fmtTime(position);
}
function display(frame) {
  scene?.show(frame);
  el('map').textContent = frame?.map?.split('/').pop() || 'CARLA';
  el('frame').textContent = finite(frame?.frame) ? frame.frame : 'â€”';
  el('speed').textContent = finite(frame?.speed_mps) ? (frame.speed_mps * 3.6).toFixed(1) : 'â€”';
  el('progress').textContent = finite(frame?.progress_m) ? frame.progress_m.toFixed(1) : 'â€”';
  el('length').textContent = finite(frame?.route_length_m) ? '/ ' + Math.round(frame.route_length_m) + ' m' : 'm';
  el('steer').textContent = finite(frame?.steer) ? Math.round(frame.steer * 100) : 'â€”';
  issue('data-issue', frame && !point(frame.observer_map?.ego, 3) ? 'This frame has no recorded vehicle pose.' : '');
}
function resetDrive() {
  recording = null;
  current = -1;
  position = 0;
  playing = false;
  el('clock').textContent = '0:00';
  display(null);
  controls();
}
function renderRecording() {
  if (!recording) return;
  const {frames,times} = recording;
  let lo = 0, hi = times.length - 1;
  while (lo < hi) {
    const mid = Math.ceil((lo + hi) / 2);
    if (times[mid] <= position) lo = mid;
    else hi = mid - 1;
  }
  if (current !== lo) { display(frames[lo]); current = lo; }
  const a = frames[lo].observer_map?.ego, b = frames[lo + 1]?.observer_map?.ego;
  if (scene && point(a,3) && point(b,3) && times[lo + 1] > times[lo]) {
    const t = Math.max(0, Math.min(1, (position - times[lo]) / (times[lo + 1] - times[lo])));
    const dyaw = ((b[2] - a[2] + 540) % 360) - 180;
    const za = frames[lo].observer_map.ego_z, zb = frames[lo + 1].observer_map.ego_z;
    scene.pose([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + dyaw * t],
      finite(za) && finite(zb) ? za + (zb - za) * t : za);
  }
  el('clock').textContent = fmtTime(position);
  el('seek').value = position;
}
async function startRecording() {
  replayRequest?.abort();
  replayRequest = new AbortController();
  const {signal} = replayRequest;
  setMode('recorded', 'Recorded');
  resetDrive();
  issue('data-issue', 'Loading recorded driveâ€¦');
  try {
    if (!catalog.length) {
      const [base, viewer] = await Promise.allSettled([
        read('./media/catalog.json', signal), read('./media/viewer-catalog.json', signal)
      ]);
      if (signal.aborted) return;
      const entries = result => Array.isArray(result.value) ? result.value : result.value?.records || [];
      const legacy = entries(base);
      const paths = new Set(legacy.map(entry => entry?.path));
      const extra = entries(viewer).filter(entry => entry?.map_url && !paths.has(entry.path));
      catalog = [...legacy, ...extra].filter(entry => typeof entry?.label === 'string' &&
        typeof entry.path === 'string' && (entry.path === '' || /^[\w-]+(?:\/[\w-]+)*$/.test(entry.path)));
      if (!catalog.length) throw new Error('Empty catalog');
      el('case').replaceChildren(...catalog.map(entry => new Option(entry.label, entry.path)));
      if (catalog.some(e => e.path === requestedCase)) el('case').value = requestedCase;
      el('case-label').hidden = false;
    }
    const selected = catalog.find(entry => entry.path === el('case').value);
    void scene?.loadTown(selected?.map_url, typeof selected?.map === 'string' ? selected.map : null);
    const directory = './media/' + (el('case').value ? el('case').value + '/' : '');
    const data = await read(directory + 'replay.json', signal, 30000);
    if (signal.aborted) return;
    const frames = Array.isArray(data) ? data : data.frames;
    if (!frames?.length || !frames.every(f => finite(f?.frame))) throw new Error('Invalid telemetry');
    const rawTimes = frames.map((f,i) => finite(f.video_time) ? f.video_time : finite(data.fps) && data.fps > 0 ? i / data.fps : f.sim_time);
    if (rawTimes.some((t,i) => !finite(t) || (i && t < rawTimes[i-1]))) throw new Error('Invalid timing');
    const times = rawTimes.map(t => t - rawTimes[0]);
    recording = {frames, times, end:times.at(-1)};
    playing = recording.end > 0;
    renderRecording();
    controls();
  } catch {
    if (!signal.aborted) issue('data-issue', 'Recorded drive unavailable. Check the recording catalog and telemetry files.');
  }
}
async function monitor(serial, signal) {
  let fallbackStarted = false, misses = 0;
  while (!signal.aborted && serial === sourceSerial) {
    const [statusResult, frameResult] = await Promise.allSettled([
      read('./api/status', signal, 1900), read('./api/frame', signal, 1900)
    ]);
    if (signal.aborted || serial !== sourceSerial) return;
    const status = statusResult.value, frame = frameResult.value;
    const active = finite(frame?.frame) && (!status || (status.state === 'running' &&
      (frame.run_id == null || status.run_id == null || frame.run_id === status.run_id)));
    if (active) {
      misses = 0;
      if (mode !== 'live' || activeLiveId !== frame.run_id) {
        replayRequest?.abort();
        setMode('live', 'Live');
        resetDrive();
        void scene?.loadTown();
        activeLiveId = frame.run_id;
      }
      display(frame);
      el('clock').textContent = finite(frame.sim_time) ? fmtTime(frame.sim_time) : 'â€“:â€“â€“';
      fallbackStarted = false;
    } else if (mode === 'live' && ++misses < 3 && (!status || status.state === 'running')) {
      issue('data-issue', 'Waiting for live telemetryâ€¦');
    } else if (!fallbackStarted) {
      fallbackStarted = true;
      void startRecording();
    }
    // A static deployment has neither API. Local servers expose status while idle,
    // and legacy read-only live servers return HTTP 503 before their first frame.
    const staticPage = !status && !frame && [statusResult,frameResult].every(result =>
      result.status === 'rejected' && (result.reason.message === '404' || result.reason instanceof SyntaxError));
    if (staticPage) return;
    await new Promise(resolve => setTimeout(resolve, active ? 150 : 1200));
  }
}
function chooseSource() {
  monitorRequest?.abort();
  replayRequest?.abort();
  sourceSerial++;
  if (el('source').value === 'recorded') { void startRecording(); return; }
  resetDrive();
  setMode('waiting', 'Connecting');
  monitorRequest = new AbortController();
  void monitor(sourceSerial, monitorRequest.signal);
}
el('source').addEventListener('change', chooseSource);
el('case').addEventListener('change', () => {
  requestedCase = el('case').value;
  // An explicitly selected recording stays on screen if a live run starts.
  el('source').value = 'recorded';
  chooseSource();
});
el('play').addEventListener('click', () => {
  if (!recording) return;
  if (position >= recording.end) { position = 0; renderRecording(); }
  playing = !playing;
  controls();
});
el('seek').addEventListener('input', () => {
  if (!recording) return;
  position = Number(el('seek').value);
  renderRecording();
});
let lastDraw;
function animate(stamp) {
  requestAnimationFrame(animate);
  if (document.hidden) { lastDraw = undefined; return; }
  if (lastDraw != null && stamp - lastDraw < 30) return;
  const elapsed = lastDraw == null ? 0 : (stamp - lastDraw) / 1000;
  lastDraw = stamp;
  if (playing && recording) {
    position = Math.min(recording.end, position + elapsed);
    renderRecording();
    if (position >= recording.end) { playing = false; controls(); }
  }
  if (scene?.dirty) {
    scene.renderer.render(scene.scene, scene.camera);
    scene.dirty = false;
  }
}
document.addEventListener('visibilitychange', () => { lastDraw = undefined; });
chooseSource();
requestAnimationFrame(animate);
