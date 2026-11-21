import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

import {recordedChapters} from '../web/encounter-chapters.mjs';

const exposed = (kind, stamp, extra = {}) => ({kind, triggered: true, exposed: true, exposed_s: stamp, ...extra});
const frame = (sim_time, events) => ({sim_time, events});

test('legacy, missing and unreached encounters do not produce chapters', () => {
  const frames = [frame(0), frame(.1, [null, {kind: 'lead', station_m: 15, triggered: false},
    {kind: 'pedestrian', triggered: true, exposed: false, exposed_s: null}])];
  assert.deepEqual(recordedChapters(frames), []);
  assert.deepEqual(recordedChapters(frames, {events: [exposed('bus', 32)]}), []);
});

test('repeated evidence uses the first actual exposure frame and accurate encounter labels', () => {
  const frames = [
    frame(4),
    frame(4.1, [{kind: 'signal', triggered: false, exposed: false, queue_exposed_s: 4.2}]),
    frame(4.2, [exposed('signal', 4.3, {queue_exposed_s: 4.2})]),
    frame(4.3, [exposed('bus', 4.4), exposed('signal', 4.3, {queue_exposed_s: 4.2})]),
    frame(4.4, [exposed('bus', 4.4, {passed: true})])
  ];
  assert.deepEqual(recordedChapters(frames), [
    {kind: 'signal', label: 'Signal queue', index: 1}, {kind: 'bus', label: 'Bus pass', index: 3}
  ]);
  assert.equal(recordedChapters(frames.slice(0, 4))[1].label, 'Stopped bus');
});

test('manifest-only evidence maps simulation times to nearest recorded frame', () => {
  const frames = [frame(3), frame(4), frame(5), frame(6)];
  assert.deepEqual(recordedChapters(frames, {summary: {events: [
    exposed('lead', 4.5), exposed('braking', 7), exposed('bus', 1)
  ]}}), [{kind: 'lead', label: 'Stopped lead vehicle', index: 2}]);
});

test('chapter button seeks to the frame video time, including before metadata is ready', () => {
  const source = readFileSync(new URL('../web/ado-home.js', import.meta.url), 'utf8');
  const extract = (start, end) => source.slice(source.indexOf(`function ${start}(`), source.indexOf(`function ${end}(`));
  const film = {currentTime: 0}, clock = {video: false}, buttons = [], attrs = {};
  let visible, syncs = 0;
  const state = {source: 'recorded', replay: null};
  const context = vm.createContext({recordedChapters, state, film, clock,
    finite: Number.isFinite, show: (_, value) => { visible = value; },
    $: () => ({replaceChildren: (...items) => buttons.splice(0, buttons.length, ...items)}),
    document: {createElement: () => ({setAttribute: (key, value) => { attrs[key] = value; },
      addEventListener(type, handler) { this[type] = handler; }})},
    duration: () => 2, clamp: (value, min, max) => Math.min(max, Math.max(min, value)),
    syncNow: () => { syncs++; }
  });
  vm.runInContext(extract('renderChapters', 'renderRun') + extract('seekTo', 'resetPlayback') +
    extract('parseReplay', 'parseTrainingLog'), context);
  state.replay = context.parseReplay([
    {frame: 1200, sim_time: 51, video_time: 0},
    {frame: 1201, sim_time: 51.1, video_time: .7, events: [exposed('bus', 51.2, {passed: true})]},
    {frame: 1202, sim_time: 51.2, video_time: 1.8}
  ]);
  context.renderChapters();
  assert.equal(visible, true);
  assert.equal(buttons.length, 1);
  assert.equal(buttons[0].type, 'button');
  assert.equal(attrs['aria-label'], 'Jump to Bus pass');
  buttons[0].click();
  assert.equal(clock.pendingT, .7);
  clock.video = true;
  buttons[0].click();
  assert.equal(film.currentTime, .7);
  assert.equal(syncs, 2);
  state.source = 'live';
  state.replay = null;
  context.renderChapters();
  assert.equal(visible, false);
  assert.deepEqual(buttons, []);
});
