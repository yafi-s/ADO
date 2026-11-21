import assert from 'node:assert/strict';
import test from 'node:test';
import {gzipSync} from 'node:zlib';

import {loadReplay, unpackReplay} from '../web/replay-pack.mjs';

const path = [[1, 2], [3, 4]];
const fixture = () => ({format: 'ado-replay-pack-v1', shared_path: path, path_ranges: [[0, 2]], replay: {
  fps: 10, frames: [{frame: 1, observer_map: {}}, {frame: 2, observer_map: {ego: [0, 0, 0]}},
    {frame: 3, observer_map: {path: [[1, 3]]}}, {frame: 4}, {frame: 5, observer_map: {path: null}}]
}});

test('unpacks by reference and keeps dynamic, missing and null routes', () => {
  const {frames, fps} = unpackReplay(fixture());
  assert.equal(fps, 10);
  assert.equal(frames[0].observer_map.path, frames[1].observer_map.path);
  assert.deepEqual(frames[2].observer_map.path, [[1, 3]]);
  assert.equal(Object.hasOwn(frames[3], 'observer_map'), false);
  assert.equal(frames[4].observer_map.path, null);
  assert.throws(() => unpackReplay({...fixture(), path_ranges: [[0, 3]]}), /Invalid packed path reference/);
});

test('loads gzip from the same origin and falls back for missing or rejected packs', async t => {
  const previousFetch = globalThis.fetch, previousLocation = globalThis.location;
  t.after(() => {
    globalThis.fetch = previousFetch;
    globalThis.location = previousLocation;
  });
  globalThis.location = new URL('https://example.test/ADO/');
  let requests = 0, rawLoads = 0;
  globalThis.fetch = async (url, options) => {
    requests++;
    assert.equal(url.href, 'https://example.test/ADO/media/replay-pack.json.gz');
    assert.equal(options.redirect, 'error');
    return requests === 1 ? new Response(gzipSync(JSON.stringify(fixture()))) : new Response('broken gzip');
  };
  const raw = async () => { rawLoads++; return ['original']; };
  const parse = replay => replay.frames ?? replay;
  const signal = new AbortController().signal;
  const frames = await loadReplay('./media/replay-pack.json.gz', signal, parse, raw);
  assert.equal(frames.length, 5);
  assert.equal(frames[0].observer_map.path, frames[1].observer_map.path);
  assert.equal(rawLoads, 0);
  assert.deepEqual(await loadReplay('./media/replay-pack.json.gz', signal, parse, raw), ['original']);
  assert.deepEqual(await loadReplay('https://elsewhere.test/replay-pack.json.gz', signal, parse, raw), ['original']);
  assert.deepEqual(await loadReplay(undefined, signal, parse, raw), ['original']);
  assert.equal(requests, 2);
  assert.equal(rawLoads, 3);
});

test('source cancellation aborts pack download without loading the raw replay', async t => {
  const previousFetch = globalThis.fetch, previousLocation = globalThis.location;
  t.after(() => {
    globalThis.fetch = previousFetch;
    globalThis.location = previousLocation;
  });
  globalThis.location = new URL('https://example.test/');
  globalThis.fetch = (_, {signal}) => new Promise((resolve, reject) => {
    signal.addEventListener('abort', () => reject(signal.reason));
  });
  const controller = new AbortController();
  const pending = loadReplay('/media/replay-pack.json.gz', controller.signal, () => assert.fail('parsed canceled data'),
    () => assert.fail('started a raw fallback for the old source'));
  controller.abort();
  assert.equal(await pending, null);
});
