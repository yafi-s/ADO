import {measuredHeads, cameraLamp, LAMP_COLOURS} from './lamp-observations.mjs';

function isCoordinate(value, size) {
  return Array.isArray(value) && value.length === size && value.every(Number.isFinite);
}

function observerRoute(data) {
  if (!Array.isArray(data?.path) || data.path.length < 2 ||
      !data.path.every(point => isCoordinate(point, 2) || isCoordinate(point, 3)) || !isCoordinate(data.ego, 3)) return;
  return {
    path: data.path,
    ego: data.ego,
    traffic: isCoordinate(data.traffic, 2) ? data.traffic : null,
    hazard: isCoordinate(data.hazard, 2) ? data.hazard : null,
    hazard_type: data.hazard_type
  };
}

// Plot colours come from the display tokens in display.css, so the canvas follows the night and day modes.
const TOKENS = {
  grid: '--plot-grid', ring: '--plot-ring', label: '--plot-label', lidar: '--plot-lidar', radar: '--plot-radar',
  ego: '--plot-ego', glass: '--plot-ego-glass', route: '--plot-route', routeBase: '--plot-route-base',
  outline: '--plot-outline', traffic: '--caution', hazard: '--vru'
};
const FALLBACK = {
  grid: '#121b29', ring: '#2a3850', label: '#8793a8', lidar: '#7fd6d8', radar: '#ffbd66', ego: '#eef3fb',
  glass: '#2a3850', route: '#45e2c7', routeBase: '#1a2940', outline: '#060a11', traffic: '#ffbd66', hazard: '#ffa59d'
};

export class SensorView {
  constructor(canvas, reading) {
    this.canvas = canvas;
    addEventListener('ado:display', () => {
      this.colors = undefined;
      this.draw();
    });
    this.reading = reading;
    this.lidar = [];
    this.radar = [];
    this.view = 'returns';
    this.observer = new ResizeObserver(() => this.draw());
    this.observer.observe(canvas);
    canvas.addEventListener('pointermove', event => {
      this.pointer = {x: event.clientX, y: event.clientY};
      this.inspect();
    });
    canvas.addEventListener('pointerleave', () => {
      this.pointer = undefined;
      reading.hidden = true;
    });
  }

  update(lidar, radar, map, frame) {
    this.lidar = lidar;
    this.radar = radar;
    this.signals = measuredHeads(frame);
    this.surround = frame?.perception_version === 'surround-v1';
    this.egoPose = map?.ego;
    this.route = observerRoute(map);
    if (!this.route) this.view = 'returns';
    this.draw();
  }

  setView(view) {
    this.view = view === 'route' && this.route ? 'route' : 'returns';
    this.reading.hidden = true;
    this.draw();
  }

  palette() {
    if (!this.colors) {
      const style = getComputedStyle(this.canvas);
      this.colors = Object.fromEntries(Object.entries(TOKENS).map(
        ([name, token]) => [name, style.getPropertyValue(token).trim() || FALLBACK[name]]
      ));
    }
    return this.colors;
  }

  draw() {
    const {canvas} = this;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    if (!width || !height) return;
    const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
    if (canvas.width !== Math.round(width * pixelRatio) || canvas.height !== Math.round(height * pixelRatio)) {
      canvas.width = Math.round(width * pixelRatio);
      canvas.height = Math.round(height * pixelRatio);
    }
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    ctx.clearRect(0, 0, width, height);
    if (this.view === 'route') {
      this.drawRoute(ctx, width, height);
      return;
    }
    this.drawReturns(ctx, width, height);
  }

  drawReturns(ctx, width, height) {
    // The returns use local forward x and right y, with the ego pointing up.
    const x0 = width / 2;
    const y0 = height * (this.surround ? .52 : .72);
    const scale = this.surround ? Math.min(width,height) / 62 : Math.min(width / 47, height / 56);
    this.projection = {x0, y0, scale};
    const color = this.palette();

    ctx.lineWidth = 1;
    ctx.strokeStyle = color.grid;
    ctx.beginPath();
    for (let d = -50; d <= 50; d += 5) {
      ctx.moveTo(x0 + d * scale, 0);
      ctx.lineTo(x0 + d * scale, height);
      ctx.moveTo(0, y0 + d * scale);
      ctx.lineTo(width, y0 + d * scale);
    }
    ctx.stroke();
    ctx.strokeStyle = color.ring;
    ctx.setLineDash([3, 4]);
    ctx.beginPath();
    for (let r = 10; r <= 50; r += 10) {
      ctx.moveTo(x0 + r * scale, y0);
      ctx.arc(x0, y0, r * scale, 0, Math.PI * 2);
    }
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = color.label;
    ctx.font = '600 10px Barlow, system-ui, sans-serif';
    for (let r = 10; y0 - r * scale > 16; r += 10) {
      ctx.fillText(r + ' m', x0 + 4, y0 - r * scale - 3);
    }

    ctx.fillStyle = color.lidar;
    for (const [x, y] of this.lidar) {
      ctx.fillRect(x0 + y * scale - 1, y0 - x * scale - 1, 2, 2);
    }
    ctx.fillStyle = color.radar;
    ctx.beginPath();
    for (const [x, y] of this.radar) {
      const px = x0 + y * scale;
      const py = y0 - x * scale;
      ctx.moveTo(px, py - 2.8);
      ctx.lineTo(px + 2.8, py);
      ctx.lineTo(px, py + 2.8);
      ctx.lineTo(px - 2.8, py);
      ctx.closePath();
    }
    ctx.fill();

    const egoW = Math.max(scale * 1.85, 8);
    const egoH = Math.max(scale * 4.3, 17);
    ctx.fillStyle = color.ego;
    if (isCoordinate(this.egoPose, 3)) {
      const [ex, ey, yaw] = this.egoPose;
      const c = Math.cos(yaw * Math.PI / 180), s = Math.sin(yaw * Math.PI / 180);
      this.drawSignals(ctx, (x,y) => [
        x0 + ((y-ey)*c - (x-ex)*s)*scale, y0 - ((x-ex)*c + (y-ey)*s)*scale
      ]);
      ctx.fillStyle = color.ego;
    }
    ctx.beginPath();
    ctx.roundRect(x0 - egoW / 2, y0 - egoH / 2, egoW, egoH, 3);
    ctx.fill();
    ctx.fillStyle = color.glass;
    ctx.fillRect(x0 - egoW * .33, y0 - egoH * .28, egoW * .66, egoH * .23);
    if (this.pointer) this.inspect();
  }

  drawRoute(ctx, width, height) {
    const {path, ego, traffic, hazard, hazard_type: hazardType} = this.route;
    const color = this.palette();
    const points = [...path, ego, traffic, hazard].filter(Boolean);
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    for (const [x, y] of points) {
      minX = Math.min(minX, x);
      maxX = Math.max(maxX, x);
      minY = Math.min(minY, y);
      maxY = Math.max(maxY, y);
    }
    const padding = Math.min(34, width / 5, height / 5);
    const scale = Math.min(
      (width - 2 * padding) / Math.max(maxX - minX, 1),
      (height - 2 * padding) / Math.max(maxY - minY, 1)
    );
    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;
    // CARLA's positive yaw rotates +X toward +Y. Drawing +Y down preserves that convention.
    const project = ([x, y]) => [width / 2 + (x - centerX) * scale, height / 2 + (y - centerY) * scale];

    ctx.save();
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.beginPath();
    ctx.moveTo(...project(path[0]));
    for (const point of path.slice(1)) ctx.lineTo(...project(point));
    ctx.lineWidth = 6;
    ctx.strokeStyle = color.routeBase;
    ctx.stroke();
    ctx.lineWidth = 2.5;
    ctx.strokeStyle = color.route;
    ctx.stroke();

    const [startX, startY] = project(path[0]);
    ctx.fillStyle = color.outline;
    ctx.strokeStyle = color.route;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(startX, startY, 4.5, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();

    const [endX, endY] = project(path.at(-1));
    ctx.fillStyle = color.route;
    ctx.fillRect(endX - 4.5, endY - 4.5, 9, 9);

    if (traffic) {
      const [x, y] = project(traffic);
      ctx.fillStyle = color.traffic;
      ctx.beginPath();
      ctx.moveTo(x, y - 5);
      ctx.lineTo(x + 5, y);
      ctx.lineTo(x, y + 5);
      ctx.lineTo(x - 5, y);
      ctx.fill();
    }

    if (hazard) {
      const [x, y] = project(hazard);
      ctx.fillStyle = color.hazard;
      ctx.strokeStyle = color.outline;
      ctx.beginPath();
      if (hazardType === 'vehicle') ctx.roundRect(x - 5, y - 4, 10, 8, 2);
      else ctx.arc(x, y, 5, 0, 2 * Math.PI);
      ctx.fill();
      ctx.stroke();
    }

    this.drawSignals(ctx, (x,y) => project([x,y]));
    ctx.translate(...project(ego));
    ctx.rotate(ego[2] * Math.PI / 180);
    ctx.fillStyle = color.ego;
    ctx.strokeStyle = color.outline;
    ctx.beginPath();
    ctx.moveTo(9, 0);
    ctx.lineTo(-5, -5.5);
    ctx.lineTo(-5, 5.5);
    ctx.closePath();
    ctx.stroke();
    ctx.fill();
    ctx.restore();
  }

  drawSignals(ctx, project) {
    for (const head of this.signals || []) {
      const [x,y] = project(head.x,head.y);
      const state = cameraLamp(head);
      ctx.fillStyle = '#202b39';
      ctx.fillRect(x-3.5,y-6,7,12);
      ctx.fillStyle = LAMP_COLOURS[state];
      ctx.beginPath();
      ctx.arc(x,y+(state==='red'?-2.5:state==='green'?2.5:0),2.25,0,Math.PI*2);
      if (state==='unknown') { ctx.strokeStyle = LAMP_COLOURS.unknown; ctx.lineWidth = 1; ctx.stroke(); }
      else ctx.fill();
    }
  }

  inspect() {
    if (!this.projection || this.view !== 'returns') return;
    const {x0, y0, scale} = this.projection;
    const bounds = this.canvas.getBoundingClientRect();
    const px = this.pointer.x - bounds.left;
    const py = this.pointer.y - bounds.top;
    let selected;
    let distance = 9;
    for (const point of this.radar) {
      const d = Math.hypot(px - x0 - point[1] * scale, py - y0 + point[0] * scale);
      if (d < distance) {
        selected = point;
        distance = d;
      }
    }
    this.reading.hidden = !Number.isFinite(selected?.[2]);
    if (!this.reading.hidden) this.reading.textContent = 'Radial velocity ' + selected[2].toFixed(1) + ' m/s';
  }
}
