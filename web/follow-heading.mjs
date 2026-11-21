import {contextOf} from './object-memory.mjs';

// Smooth the viewing direction over ±0.8 simulation seconds. This symmetric
// kernel preserves constant turn rate without the phase lag of a trailing mean.
const radius = .8;
const radians = Math.PI / 180;

// Convolution of max(t, 0) with the unit-area biweight kernel, less max(t, 0).
// A change in angular velocity adds a hinge to the recorded heading curve.
// Outside the kernel radius the smoothed hinge and original are identical.
function roundedHinge(distance) {
  const u = Math.abs(distance) / radius;
  return radius * (1 - u) ** 4 * (u * u + 4 * u + 5) / 32;
}

export class FollowHeading {
  constructor(frames) {
    this.knots = [];
    let previous = null;
    let segment = 0;
    for (const frame of frames) {
      const index = this.knots.length;
      const ctx = contextOf(frame, previous, undefined, index / 10);
      const last = this.knots[index - 1];
      if (ctx.reset) segment = index;
      const rate = ctx.reset ? 0 : -ctx.yawRate;
      const angle = ctx.reset ? -ctx.degrees * radians : last.angle + rate * ctx.dt;
      // A segment extends horizontally beyond its first and last measurements.
      if (!ctx.reset) last.change += rate;
      this.knots.push({time: ctx.t, angle, change: -rate, segment});
      previous = ctx;
    }
  }

  at(index, fraction = 0) {
    const knots = this.knots;
    const start = knots[index];
    const next = knots[index + 1]?.segment === start.segment ? knots[index + 1] : start;
    const time = start.time + (next.time - start.time) * fraction;
    let angle = start.angle + (next.angle - start.angle) * fraction;
    for (let i = index; i >= start.segment && time - knots[i].time < radius; i--) {
      angle += knots[i].change * roundedHinge(time - knots[i].time);
    }
    for (let i = index + 1; i < knots.length && knots[i].segment === start.segment && knots[i].time - time < radius; i++) {
      angle += knots[i].change * roundedHinge(knots[i].time - time);
    }
    return angle;
  }
}
