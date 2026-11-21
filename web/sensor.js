// CARLA uses x forward and y right. The canvas looks down on the ego vehicle.
export class SensorView {
  constructor(canvas, reading) {
    this.canvas = canvas;
    this.reading = reading;
    this.lidar = [];
    this.radar = [];
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

  update(lidar, radar) {
    this.lidar = lidar;
    this.radar = radar;
    this.draw();
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
    const x0 = width / 2;
    const y0 = height * .72;
    const scale = Math.min(width / 47, height / 56);
    this.projection = {x0, y0, scale};

    ctx.lineWidth = 1;
    ctx.strokeStyle = '#304037';
    ctx.beginPath();
    for (let d = -50; d <= 50; d += 5) {
      ctx.moveTo(x0 + d * scale, 0);
      ctx.lineTo(x0 + d * scale, height);
      ctx.moveTo(0, y0 + d * scale);
      ctx.lineTo(width, y0 + d * scale);
    }
    ctx.stroke();
    ctx.strokeStyle = '#586c5b';
    ctx.setLineDash([3, 4]);
    ctx.beginPath();
    for (let r = 10; r <= 50; r += 10) {
      ctx.moveTo(x0 + r * scale, y0);
      ctx.arc(x0, y0, r * scale, 0, Math.PI * 2);
    }
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#a1b69e';
    ctx.font = '9px ui-monospace, Consolas, monospace';
    for (let r = 10; y0 - r * scale > 16; r += 10) {
      ctx.fillText(r + ' m', x0 + 4, y0 - r * scale - 3);
    }

    ctx.fillStyle = '#89d9b6';
    for (const [x, y] of this.lidar) {
      ctx.fillRect(x0 + y * scale - 1, y0 - x * scale - 1, 2, 2);
    }
    ctx.fillStyle = '#ffc16d';
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
    ctx.fillStyle = '#e1ecca';
    ctx.beginPath();
    ctx.roundRect(x0 - egoW / 2, y0 - egoH / 2, egoW, egoH, 3);
    ctx.fill();
    ctx.fillStyle = '#374c40';
    ctx.fillRect(x0 - egoW * .33, y0 - egoH * .28, egoW * .66, egoH * .23);
    if (this.pointer) this.inspect();
  }

  inspect() {
    if (!this.projection) return;
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
