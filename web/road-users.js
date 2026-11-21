// ADO road-user shapes for the perception display and the map.
// Each shape is authored in metres at a nominal size, then normalised to a unit box
// (x forward -0.5..0.5, y up 0..1, z across -0.5..0.5) so a renderer can scale it
// to a track's measured or estimated length, height and width.
// They are symbols for a class of road user. They are not models of the actors in the simulator.
import * as T from './vendor/city-view/three-view.js';

const V = (x = 0, y = 0, z = 0) => new T.Vector3(x, y, z);
const UP = V(0, 1, 0);

export const TRACK_TONE = {
  car: 'vehicle', bus: 'vehicle', truck: 'vehicle', motorcycle: 'rider', bicycle: 'rider', person: 'person', unknown: 'unknown'
};

// The range of sizes, in metres, over which each symbol still looks like its class: [min, max] for
// length, height and width. A renderer can keep the symbol inside this range and show the
// tracker's own estimate another way, such as a ground outline.
const PLAUSIBLE = {
  car: [[3.4, 5.4], [1.25, 1.95], [1.55, 2.1]],
  truck: [[5, 12], [2.2, 3.9], [2, 2.7]],
  bus: [[8, 13.5], [2.7, 3.5], [2.2, 2.7]],
  person: [[.35, .75], [1.1, 1.98], [.35, .75]],
  bicycle: [[1.3, 2.1], [1.3, 2], [.45, .8]],
  motorcycle: [[1.6, 2.5], [1.2, 1.9], [.55, 1]]
};
export function plausibleSize(label, size, target = size) {
  const limits = PLAUSIBLE[label];
  if (!limits) return target.copy(size);
  return target.set(
    Math.min(limits[0][1], Math.max(limits[0][0], size.x)),
    Math.min(limits[1][1], Math.max(limits[1][0], size.y)),
    Math.min(limits[2][1], Math.max(limits[2][0], size.z))
  );
}

// Geometry is shared between every instance of a class and lives for the page.
const geometries = new Map();
function shared(key, build) {
  if (!geometries.has(key)) geometries.set(key, build());
  return geometries.get(key);
}
export function disposeRoadUserGeometry() {
  for (const geometry of geometries.values()) geometry.dispose();
  geometries.clear();
}

// Writes depth without colour, so a translucent shape never shows its own inner faces.
const depthOnly = new T.MeshBasicMaterial({colorWrite: false, transparent: true});

// A smooth closed shell through cross-sections along X.
// Each section is [x, bottom, top, halfWidth, squareness]; squareness 2 is an ellipse, higher is boxier.
function loft(sections, around = 28, squareness = 4) {
  const positions = [];
  const indices = [];
  for (const [x, bottom, top, half, power = squareness] of sections) {
    const middle = (bottom + top) / 2;
    const rise = (top - bottom) / 2;
    for (let j = 0; j < around; j++) {
      const angle = j / around * Math.PI * 2;
      const c = Math.cos(angle);
      const s = Math.sin(angle);
      positions.push(x, middle + rise * Math.sign(s) * Math.abs(s) ** (2 / power), half * Math.sign(c) * Math.abs(c) ** (2 / power));
    }
  }
  const count = sections.length;
  for (let i = 0; i < count - 1; i++) {
    for (let j = 0; j < around; j++) {
      const a = i * around + j;
      const b = i * around + (j + 1) % around;
      indices.push(a, a + around, b, b, a + around, b + around);
    }
  }
  for (const [i, forward] of [[0, false], [count - 1, true]]) {
    const [x, bottom, top] = sections[i];
    const centre = positions.length / 3;
    positions.push(x, (bottom + top) / 2, 0);
    for (let j = 0; j < around; j++) {
      const a = i * around + j;
      const b = i * around + (j + 1) % around;
      if (forward) indices.push(centre, b, a);
      else indices.push(centre, a, b);
    }
  }
  const geometry = new T.BufferGeometry();
  geometry.setAttribute('position', new T.Float32BufferAttribute(positions, 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  return geometry;
}

function builder(look, label) {
  const materials = [];
  const keep = material => { materials.push(material); return material; };
  const tint = look[TRACK_TONE[label]] ?? look.unknown;
  const solid = (color, properties = {}) => keep(new T.MeshStandardMaterial({
    color, roughness: .5, metalness: .08, transparent: true, depthWrite: false, ...properties
  }));
  const flat = color => keep(new T.MeshBasicMaterial({color, toneMapped: false, transparent: true, depthWrite: false}));
  const body = solid(tint, {roughness: .42, emissive: tint, emissiveIntensity: look.glow});
  if (look.rim) {
    // A soft edge light, strongest where the surface turns away from the viewer.
    const rim = {value: look.rim};
    body.onBeforeCompile = shader => {
      shader.uniforms.uRim = rim;
      shader.fragmentShader = shader.fragmentShader
        .replace('#include <common>', '#include <common>\nuniform float uRim;')
        .replace('#include <emissivemap_fragment>', `#include <emissivemap_fragment>
          totalEmissiveRadiance += diffuseColor.rgb * uRim * pow(1. - abs(dot(normalize(normal), normalize(vViewPosition))), 2.2);`);
    };
    body.customProgramCacheKey = () => 'ado-rim';
  }
  const paints = {
    body,
    glass: solid(look.glass, {metalness: .3, roughness: .22}),
    tyre: solid(look.tyre, {roughness: .8, metalness: 0}),
    hub: solid(look.hub, {roughness: .45, metalness: .3}),
    lamp: flat(look.lamp),
    tail: flat(look.tail)
  };
  const root = new T.Group();
  const inner = new T.Group();
  root.add(inner);

  function part(geometry, material, x = 0, y = 0, z = 0) {
    const mesh = new T.Mesh(geometry, material);
    mesh.position.set(x, y, z);
    mesh.renderOrder = 3;
    const depth = new T.Mesh(geometry, depthOnly);
    depth.renderOrder = 2;
    mesh.add(depth);
    inner.add(mesh);
    return mesh;
  }
  const shell = (key, sections, material, around, squareness) =>
    part(shared(key, () => loft(sections, around, squareness)), material);
  const box = (w, h, d, r, material, x, y, z) =>
    part(shared(`box:${w}:${h}:${d}:${r}`, () => new T.RoundedBoxGeometry(w, h, d, 3, r)), material, x, y, z);
  const ball = (radius, material, x, y, z, sx = 1, sy = 1, sz = 1) => {
    const mesh = part(shared(`ball:${radius}`, () => new T.SphereGeometry(radius, 20, 14)), material, x, y, z);
    mesh.scale.set(sx, sy, sz);
    return mesh;
  };
  // A rounded limb between two points.
  function limb(start, end, radius, material) {
    const a = V(...start);
    const b = V(...end);
    const delta = b.clone().sub(a);
    const length = Math.max(.001, delta.length() - radius * 2);
    const mesh = part(shared(`limb:${radius}:${length.toFixed(3)}`, () => new T.CapsuleGeometry(radius, length, 5, 12)), material);
    mesh.position.copy(a).add(b).multiplyScalar(.5);
    mesh.quaternion.setFromUnitVectors(UP, delta.normalize());
    return mesh;
  }
  function wheel(x, z, radius, width, hubRadius = radius * .58) {
    const tyre = part(shared(`tyre:${radius}:${width}`, () => new T.CylinderGeometry(radius, radius, width, 28)), paints.tyre, x, radius, z);
    tyre.rotation.x = Math.PI / 2;
    const hub = part(shared(`hub:${hubRadius}:${width}`, () => new T.CylinderGeometry(hubRadius, hubRadius, width * 1.06, 20)), paints.hub, x, radius, z);
    hub.rotation.x = Math.PI / 2;
  }
  // Scale the metre-space parts into the unit box.
  function finish(length, height, width) {
    inner.scale.set(1 / length, 1 / height, 1 / width);
    return {group: root, materials};
  }
  return {paints, shell, box, ball, limb, wheel, finish, part};
}

function car(b) {
  const {paints: p} = b;
  b.shell('car-body', [
    [-2.25, .47, .78, .58], [-2.19, .36, .90, .78], [-1.98, .25, .98, .89], [-1.25, .21, 1.00, .925],
    [0, .21, .98, .93], [.85, .21, .95, .925], [1.5, .23, .87, .91], [1.98, .27, .77, .86],
    [2.19, .36, .69, .74], [2.25, .45, .62, .54]
  ], p.body, 28, 4.6);
  b.shell('car-glass', [
    [-1.74, .93, .95, .66], [-1.38, .92, 1.27, .755], [-.92, .92, 1.44, .78], [-.2, .92, 1.47, .785],
    [.26, .92, 1.44, .78], [.76, .92, 1.21, .765], [1.14, .91, .94, .72]
  ], p.glass, 24, 3.4);
  b.shell('car-roof', [
    [-.98, 1.36, 1.455, .60], [-.62, 1.40, 1.485, .67], [-.1, 1.42, 1.495, .68], [.24, 1.40, 1.47, .66], [.36, 1.37, 1.44, .58]
  ], p.body, 20, 5);
  for (const x of [-1.36, 1.38]) for (const z of [-.83, .83]) b.wheel(x, z, .335, .24);
  for (const side of [-1, 1]) {
    b.box(.06, .08, .34, .028, p.lamp, 2.205, .665, side * .50);
    b.box(.06, .075, .40, .028, p.tail, -2.215, .80, side * .52);
  }
  return b.finish(4.5, 1.5, 1.86);
}

function truck(b) {
  const {paints: p} = b;
  // Cab, then a cargo box behind it.
  b.shell('truck-cab', [
    [1.42, .50, 2.20, 1.08], [1.52, .42, 2.30, 1.15], [2.45, .42, 2.28, 1.16], [2.92, .42, 2.0, 1.15],
    [3.16, .44, 1.42, 1.13], [3.25, .52, 1.05, 1.02]
  ], p.body, 28, 6);
  b.shell('truck-glass', [
    [2.34, 1.48, 2.20, 1.165], [2.72, 1.46, 2.13, 1.165], [3.02, 1.44, 1.78, 1.14], [3.17, 1.42, 1.5, 1.08]
  ], p.glass, 24, 6);
  b.box(4.62, 2.28, 2.38, .09, p.body, -.92, 1.66, 0);
  b.box(4.4, .16, 1.9, .05, p.tyre, -.9, .47, 0);
  for (const x of [-2.25, -1.15, 2.3]) for (const z of [-.98, .98]) b.wheel(x, z, .47, .32);
  for (const side of [-1, 1]) {
    b.box(.06, .11, .34, .03, p.lamp, 3.215, .86, side * .72);
    b.box(.05, .16, .22, .03, p.tail, -3.235, .78, side * .98);
  }
  return b.finish(6.5, 2.8, 2.4);
}

function bus(b) {
  const {paints: p} = b;
  b.shell('bus-body', [
    [-5.5, .55, 2.85, 1.06], [-5.42, .40, 3.0, 1.2], [-4.9, .36, 3.08, 1.265], [4.7, .36, 3.08, 1.265],
    [5.3, .38, 3.02, 1.24], [5.46, .46, 2.86, 1.16], [5.5, .6, 2.6, 1.0]
  ], p.body, 28, 7);
  // One window band along both sides and a windscreen.
  b.box(9.6, .92, 2.56, .05, p.glass, -.2, 2.08, 0);
  b.box(.09, .84, 1.94, .07, p.glass, -5.47, 2.10, 0);
  // Narrow pillars break up the glazing so the long body reads at street scale.
  for (const x of [-3.55, -1.95, -.35, 1.25, 2.85]) {
    b.box(.08, .95, 2.58, .025, p.body, x, 2.08, 0);
  }
  b.box(.07, .19, .65, .04, p.tyre, -5.50, .89, 0);
  b.shell('bus-screen', [[5.02, 1.42, 2.72, 1.255], [5.34, 1.40, 2.66, 1.225], [5.49, 1.44, 2.5, 1.12]], p.glass, 24, 7);
  for (const x of [-3.7, -2.55, 3.75]) for (const z of [-1.07, 1.07]) b.wheel(x, z, .5, .34);
  for (const side of [-1, 1]) {
    b.box(.06, .12, .4, .03, p.lamp, 5.475, .86, side * .78);
    b.box(.05, .2, .2, .03, p.tail, -5.49, .9, side * .92);
  }
  return b.finish(11, 3.1, 2.55);
}

function person(b) {
  const {paints: p} = b;
  // Mid-stride, so the figure reads as a pedestrian from any angle.
  b.ball(.118, p.body, .012, 1.615, 0, 1, 1.12, .95);
  b.limb([0, 1.44, 0], [0, 1.0, 0], .15, p.body).scale.set(.74, 1, 1.12);
  b.ball(.15, p.body, 0, .95, 0, .76, .8, 1.06);
  for (const side of [-1, 1]) {
    const stride = side * .11;
    b.limb([0, .95, side * .088], [stride, .09, side * .095], .074, p.body);
    b.ball(.07, p.body, stride + .05, .05, side * .095, 1.7, .68, 1);
    b.limb([0, 1.40, side * .212], [-stride * .85, .90, side * .245], .05, p.body);
  }
  return b.finish(.52, 1.75, .6);
}

function rider(b, seat, shoulder, head, hands, knee, feet, helmet = .118) {
  const {paints: p} = b;
  b.ball(.15, p.body, seat[0], seat[1] + .05, 0, .8, .8, 1.02);
  b.limb([seat[0] + .02, seat[1] + .04, 0], [shoulder[0], shoulder[1], 0], .135, p.body).scale.set(.8, 1, 1.12);
  b.ball(helmet, p.body, head[0], head[1], 0, 1.05, 1.08, .98);
  for (const side of [-1, 1]) {
    b.limb([shoulder[0], shoulder[1] - .04, side * .19], [hands[0], hands[1], side * .2], .046, p.body);
    const foot = feet[side < 0 ? 0 : 1];
    b.limb([seat[0], seat[1], side * .09], [knee[0], knee[1], side * .12], .068, p.body);
    b.limb([knee[0], knee[1], side * .12], [foot[0], foot[1], side * .11], .056, p.body);
  }
}

function bicycle(b) {
  const {paints: p} = b;
  const rear = [-.53, .34];
  const front = [.54, .34];
  for (const [x, y] of [rear, front]) {
    // Open rims: a ring facing each side, and a small hub.
    for (const turn of [0, Math.PI]) {
      b.part(shared('cycle-rim', () => new T.RingGeometry(.275, .34, 40)), p.tyre, x, y, 0).rotation.y = turn;
    }
    b.part(shared('cycle-hub', () => new T.CylinderGeometry(.05, .05, .09, 14)), p.hub, x, y, 0).rotation.x = Math.PI / 2;
  }
  const crank = [.0, .33, 0];
  const saddle = [-.2, .9, 0];
  const stem = [.4, .86, 0];
  for (const [a, c] of [[saddle, crank], [crank, stem], [stem, saddle], [saddle, [rear[0], rear[1], 0]], [crank, [rear[0], rear[1], 0]],
    [stem, [front[0], front[1], 0]], [[.42, .98, -.22], [.42, .98, .22]]]) b.limb(a, c, .024, p.tyre);
  b.limb([.4, .86, 0], [.42, .98, 0], .024, p.tyre);
  rider(b, [-.2, .95], [.15, 1.4], [.27, 1.585], [.42, 1.0], [.12, .74], [[.06, .22], [-.04, .46]]);
  return b.finish(1.76, 1.72, .62);
}

function motorcycle(b) {
  const {paints: p} = b;
  for (const x of [-.72, .72]) b.wheel(x, 0, .31, .13, .17);
  b.shell('moto-body', [
    [-1.0, .62, .80, .07], [-.62, .46, .88, .15], [-.1, .36, .90, .19], [.3, .40, 1.0, .19], [.62, .52, 1.0, .15], [.82, .66, .94, .08]
  ], p.tyre, 20, 3);
  b.shell('moto-tank', [[-.5, .84, .90, .12], [-.2, .82, .98, .17], [.22, .84, 1.07, .17], [.5, .88, 1.02, .1]], p.hub, 18, 3);
  for (const side of [-1, 1]) b.limb([.5, 1.0, side * .07], [.72, .33, side * .07], .034, p.hub);
  b.limb([.5, 1.08, -.3], [.5, 1.08, .3], .028, p.tyre);
  b.box(.05, .1, .12, .03, p.lamp, .86, .9, 0);
  b.box(.05, .06, .12, .025, p.tail, -1.02, .74, 0);
  rider(b, [-.3, .93], [.02, 1.36], [.12, 1.56], [.5, 1.1], [.1, .76], [[.0, .42], [.0, .42]], .135);
  return b.finish(2.1, 1.7, .8);
}

function unknown(b) {
  // A cluster without a reliable class gets a footprint. The return layer shows its observed surface.
  for (const x of [-1, 1]) {
    for (const z of [-1, 1]) {
      b.box(.23, .035, .024, .009, b.paints.body, x * .375, .055, z * .48);
      b.box(.024, .035, .23, .009, b.paints.body, x * .48, .055, z * .375);
    }
  }
  return b.finish(1, 1, 1);
}

const SHAPES = {car, truck, bus, person, bicycle, motorcycle, unknown};

// Returns {group, materials}. The materials belong to this instance: fade and dispose them with it.
export function buildRoadUser(label, look) {
  const kind = Object.hasOwn(SHAPES, label) ? label : 'unknown';
  return SHAPES[kind](builder(look, kind));
}
