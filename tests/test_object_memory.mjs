import assert from 'node:assert/strict';
import test from 'node:test';

import {ObjectMemory, advance, contextOf, createState, laneAxes} from '../web/object-memory.mjs';

// A tracker sample as recordings carry it. Positions are in the vehicle frame, x ahead and y to the right.
const car = (id, x, y, more = {}) => ({id, label: 'car', x, y, z: .76, length: 4.4, width: 1.85, height: 1.52,
  heading: 0, speed_mps: 0, confidence: .5, coasting: false, sources: ['rgb', 'lidar'], ...more});
const person = (id, x, y, more = {}) => ({...car(id, x, y), label: 'person', length: .5, width: .5, height: 1.7, z: .85, ...more});
const coasting = sample => ({...sample, coasting: true, sources: []});
// A detection box across the camera image wherever a vehicle-frame point projects.
const boxOver = (x, y, label = 'car') => {
  const u = .5 + .5 * y / (x - 1.3);
  return {label, score: .5, box: [u - .05, .4, u + .05, .6]};
};
const returnsOn = (x, y) => [[x - 1, y], [x, y], [x + 1, y]];

// Play frames ten times a second with the vehicle stopped at the origin unless `ego` says otherwise.
function play(frames, deps = {}) {
  const state = createState();
  let previous = null;
  return frames.map((content, index) => {
    const frame = {frame: index, sim_time: index / 10, scenario: 'test', run_id: 'r', seed: 1,
      observer_map: {ego: content.ego || [0, 0, 0], ego_z: 0}, tracks: content.tracks || [],
      detections: content.detections || [], lidar: content.lidar || []};
    previous = contextOf(frame, previous);
    return advance(state, previous, deps);
  });
}
const repeat = (count, make) => Array.from({length: count}, (_, i) => make(i));
// A lane centre line along the world x axis, sampled every 2 m as a town export is.
const lane = repeat(101, i => [i * 2 - 100, 0, 0]);

test('a single detection is never drawn', () => {
  const drawn = play([{tracks: [car(1, 20, -4, {speed_mps: null})]}, ...repeat(8, () => ({tracks: [coasting(car(1, 20, -4, {speed_mps: null}))]}))]);
  assert.ok(drawn.every(objects => objects.length === 0));
});

test('a second measurement confirms an object, and it fades in', () => {
  // The drive has been running for a second when the car is first measured.
  const drawn = play([...repeat(10, () => ({})),
    ...repeat(8, i => ({tracks: [i % 2 ? coasting(car(1, 20, -4)) : car(1, 20, -4, {speed_mps: i ? 0 : null})]}))]).slice(10);
  assert.equal(drawn[0].length, 0);
  assert.equal(drawn[2].length, 1);
  assert.ok(drawn[2][0].opacity > 0 && drawn[2][0].opacity < 1);
  assert.equal(drawn[7][0].opacity, 1);
  // Joining a drive part-way shows what is already tracked at once.
  const joined = play(repeat(4, i => ({tracks: [i % 2 ? coasting(car(1, 20, -4)) : car(1, 20, -4)]})));
  assert.ok(joined[0][0].opacity > .99);
});

test('the detector idling on alternate frames does not dim an object', () => {
  const drawn = play(repeat(20, i => ({tracks: [i % 2 ? coasting(car(1, 20, -4)) : car(1, 20, -4)]})));
  assert.ok(drawn.slice(8).every(objects => objects[0].opacity === 1 && !objects[0].held));
});

test('a re-detected object under a new id and class is the same drawn object', () => {
  const frames = repeat(40, i => {
    if (i < 10) return {tracks: [i % 2 ? coasting(car(1, 20, -4)) : car(1, 20, -4)]};
    if (i < 22) return {detections: [boxOver(20, -4)]};
    return {tracks: [i % 2 ? coasting(car(7, 20.4, -4.2, {label: 'truck', confidence: .2})) : car(7, 20.4, -4.2, {label: 'truck', confidence: .2})]};
  });
  const drawn = play(frames);
  const keys = new Set(drawn.flat().map(object => object.key));
  assert.equal(keys.size, 1);
  assert.ok(drawn.slice(4).every(objects => objects.length === 1), 'it stays on screen through the gap');
  assert.ok(drawn[15][0].held && drawn[15][0].opacity < 1, 'drawn as remembered while unmeasured');
  assert.equal(drawn[39][0].label, 'car', 'the class with the most confidence names it');
  assert.ok(Math.abs(drawn[39][0].z - -4) < .2, 'a box of another class does not move it');
});

test('a stopped object in clear view leaves once nothing recorded vouches for it', () => {
  const seen = repeat(10, i => ({tracks: [i % 2 ? coasting(car(1, 20, -4)) : car(1, 20, -4)]}));
  const gone = play([...seen, ...repeat(40, () => ({}))]);
  assert.equal(gone[25].length, 1, 'held for two seconds');
  assert.equal(gone[45].length, 0);
  const detected = play([...seen, ...repeat(60, () => ({detections: [boxOver(20, -4, 'bus')]}))]);
  assert.equal(detected[69].length, 1, 'a box of its family over it keeps it');
  const returned = play([...seen, ...repeat(60, () => ({lidar: returnsOn(20, -4)}))]);
  assert.equal(returned[69].length, 1, 'lidar returns on it keep it for a while');
  assert.equal(play([...seen, ...repeat(140, () => ({lidar: returnsOn(20, -4)}))])[149].length, 0, 'but not for ever');
});

test('a stopped object stays put in the world as the vehicle passes, then leaves behind it', () => {
  const frames = repeat(60, i => {
    const ego = [i * .5, 0, 0];   // 5 m/s
    const ahead = 20 - i * .5;
    return {ego, tracks: i < 20 ? [i % 2 ? coasting(car(1, ahead, 4)) : car(1, ahead, 4)] : [], lidar: returnsOn(ahead, 4)};
  });
  const drawn = play(frames);
  const xs = drawn.filter(objects => objects.length).map(objects => objects[0].x);
  assert.ok(Math.max(...xs) - Math.min(...xs) < .3, 'world position holds');
  assert.equal(drawn[35].length, 1, 'still drawn beside the vehicle');
  assert.equal(drawn[59].length, 0, 'gone once well behind');
});

test('a travelling object follows its course for a moment, then goes', () => {
  // An oncoming car closing at 8 m/s in the next lane.
  const frames = repeat(40, i => ({tracks: i < 16 ? [i % 2 ? coasting(car(1, 40 - i * .8, -4, {speed_mps: 8})) : car(1, 40 - i * .8, -4, {speed_mps: 8})] : []}));
  const drawn = play(frames);
  assert.ok(drawn[20][0].x < drawn[15][0].x - 2, 'it keeps moving after the last measurement');
  assert.equal(drawn[39].length, 0);
});

test('a remembered object the vehicle drives through is dropped at once', () => {
  const frames = repeat(60, i => ({ego: [i * .5, 0, 0], tracks: i < 10 ? [i % 2 ? coasting(person(1, 20 - i * .5, 0)) : person(1, 20 - i * .5, 0)] : [],
    detections: [boxOver(20 - i * .5, 0, 'person')]}));
  const drawn = play(frames);
  const last = drawn.findLastIndex(objects => objects.some(object => object.opacity > .3));
  assert.ok(last < 38, `still drawn at frame ${last}`);   // the vehicle reaches the spot at frame 40
});

test('a vehicle-class detection far from every lane is not drawn', () => {
  const lookup = laneAxes([lane]);
  const deps = {laneAxis: lookup, laneDistance: lookup.distance};
  const frames = repeat(12, i => ({tracks: [i % 2 ? coasting(car(1, 20, 9)) : car(1, 20, 9), i % 2 ? coasting(car(2, 20, 3)) : car(2, 20, 3),
    i % 2 ? coasting(person(3, 15, 9)) : person(3, 15, 9)]}));
  const drawn = play(frames, deps)[11];
  assert.deepEqual(drawn.map(object => object.label).sort(), ['car', 'person']);
  assert.ok(Math.abs(drawn.find(object => object.label === 'car').z - 3) < .1);
});

test('a vehicle first seen in a lane faces the way the lane is driven', () => {
  // One lane stored west to east, driven against its stored order.
  const against = laneAxes([lane], [false]), unknown = laneAxes([lane]);
  assert.equal(Math.abs(against(10, 0, 0, true)), 180);
  assert.equal(against(10, 0, 0), 0, 'the plain axis still agrees with the hint');
  assert.equal(unknown(10, 0, 0, true), 0);
  const frames = repeat(6, i => ({tracks: [i % 2 ? coasting(car(1, 20, 0)) : car(1, 20, 0)]}));
  const heading = play(frames, {laneAxis: against, laneDistance: against.distance})[5][0].heading;
  assert.equal(Math.abs(((heading % 360) + 540) % 360 - 180), 180);
});

test('joining a drive part-way takes up what the tracker is coasting', () => {
  const drawn = play([{tracks: [coasting(car(1, 20, -4, {age_s: .3}))], detections: [boxOver(20, -4)]},
    ...repeat(10, () => ({detections: [boxOver(20, -4)]}))]);
  assert.equal(drawn[0].length, 1);
  assert.ok(drawn[0][0].opacity > .9 && drawn[10].length === 1, 'shown at once and kept while a box vouches for it');
  // Later in the same drive a coast alone starts nothing.
  const later = play([...repeat(10, () => ({})), ...repeat(6, () => ({tracks: [coasting(car(1, 20, -4))]}))]);
  assert.ok(later.every(objects => objects.length === 0));
});

test('a frame shows the same objects however playback reached it', () => {
  const frames = repeat(60, i => ({frame: i, sim_time: i / 10, scenario: 'test', run_id: 'r', seed: 1,
    observer_map: {ego: [0, 0, 0], ego_z: 0}, detections: [boxOver(20, -4)],
    tracks: i < 12 ? [i % 2 ? coasting(car(1, 20, -4)) : car(1, 20, -4)] : []}));
  const memory = new ObjectMemory();
  const straight = frames.map(frame => memory.step(frame).objects);
  assert.equal(straight[40].length, 1);
  assert.deepEqual(memory.step(frames[40]).objects, straight[40], 'seeking back');
  assert.deepEqual(memory.step(frames[41]).objects, straight[41], 'and playing on from there');
  // A fresh page that jumps straight to frame 40 has no history and shows nothing remembered.
  const cold = new ObjectMemory();
  assert.equal(cold.step(frames[40]).objects.length, 0);
  // Playing from the start afterwards replaces that with the full picture.
  frames.slice(0, 41).forEach(frame => cold.step(frame));
  assert.deepEqual(cold.step(frames[40]).objects, straight[40]);
});
