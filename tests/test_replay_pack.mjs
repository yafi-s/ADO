import assert from 'node:assert/strict';
import test from 'node:test';
import {gzipSync} from 'node:zlib';

import {loadReplay, unpackReplay} from '../web/replay-pack.mjs';
import {applyBarrierReview,barrierReviewer,loadBarrierReview} from '../web/barrier-review.mjs';
import {ReplayMemory} from '../web/object-memory.mjs';
import {cameraBoxes,bearingObservations} from '../web/lamp-observations.mjs';
import {createHash} from 'node:crypto';

test('reviewed chain labels require two qualified posts in the same camera and an owned range track',()=>{
  const box={label:'barrier',barrier_subtype:'chain',image_id:11,track_id:6,range_supported:true,box:[.1,.2,.4,.5]};
  const unsupported={...box,image_id:18,track_id:null,range_supported:false};
  const board={...box,image_id:72,track_id:8,barrier_subtype:'striped_board'};
  const track={id:6,label:'barrier',barrier_subtype:'chain',x:6,y:0,z:.4,length:1.9,width:.2,height:.8,
    heading:0,confidence:.95,sources:['rgb','lidar'],cameras:['front','left'],rgb_boxes:{front:box.box,left:box.box}};
  const native=Array.from({length:7},(_,i)=>{
    const coasting=i===0||i===3||i===6;
    return {frame:20+i,sim_time:i/10,perception_version:'surround-v1',run_id:'r',observer_map:{ego:[0,0,0]},
      tracks:[{...track,coasting,sources:coasting?[]:track.sources}],detections:coasting?[]:[box,unsupported,board],
      camera_detections:coasting?{}:{front:[box,unsupported,board],left:[box]}};
  });
  const decision=confirmed=>({6:{confirmed,observations:['front','left'].map(view=>({...box,view,confirmed:confirmed&&view==='front'}))},
    untracked:{observations:[{...unsupported,view:'front',confirmed:true}]}});
  const review={schema:'ado-barrier-post-review-v1',first_frame:20,last_frame:26,
    frames:{21:decision(true),22:decision(true),24:decision(false)}};
  const reviewed=applyBarrierReview({frames:native},review).frames;
  const inline=native.map(frame=>({...frame,barrier_review:review.frames[frame.frame]??{}}));
  assert.deepEqual(applyBarrierReview(inline,null).map(f=>f.tracks),reviewed.map(f=>f.tracks));
  const live=barrierReviewer();
  for(const [i,frame] of inline.entries()){
    const result=live(frame);
    assert.deepEqual(result.tracks,reviewed[i].tracks);
    assert.equal(live({...frame}),result,'a repeated poll must not lose confirmation');
  }
  assert.deepEqual(reviewed.map(f=>f.tracks.length),[0,1,1,1,0,0,0]);
  assert.deepEqual(reviewed[1].tracks[0].cameras,['front']);
  assert.deepEqual(cameraBoxes(reviewed[1],'left'),[]);
  assert.deepEqual(cameraBoxes(reviewed[1],'front').map(d=>d.image_id),[11,72]);
  assert.deepEqual(bearingObservations(reviewed[1],reviewed[0],['front']),[]);
  assert.deepEqual(cameraBoxes(reviewed[4],'front').map(d=>d.image_id),[72]);
  assert.equal(reviewed[4].barrier_review.tracks[0],native[4].tracks[0]);
  assert.equal(native[1].camera_detections.front.length,3);
  const memory=new ReplayMemory({});
  assert.equal(memory.at(reviewed,2).objects[0].barrierSubtype,'chain');
  assert.deepEqual(memory.at(reviewed,4).objects,[]);
  assert.deepEqual(new ReplayMemory({}).at(reviewed,5).objects,[]);
  assert.throws(()=>applyBarrierReview(native,{...review,last_frame:99}),/does not match/);
});

test('the declared post review must match its byte digest before it can qualify a replay',async t=>{
  const previousFetch=globalThis.fetch,previousLocation=globalThis.location;
  t.after(()=>{globalThis.fetch=previousFetch;globalThis.location=previousLocation;});
  globalThis.location=new URL('https://example.test/');
  const bytes='{"schema":"ado-barrier-post-review-v1"}';
  globalThis.fetch=async()=>new Response(bytes);
  const entry={barrier_review_url:'./media/urban-pace/barrier-review.json',
    barrier_review_sha256:createHash('sha256').update(bytes).digest('hex')};
  assert.deepEqual(await loadBarrierReview(entry),JSON.parse(bytes));
  await assert.rejects(loadBarrierReview({...entry,barrier_review_sha256:'0'.repeat(64)}),/checksum/);
  assert.equal(await loadBarrierReview({}),null);
});

test('a depth-only board update can continue an optically confirmed board and a failed span withdraws it',()=>{
  const box={label:'barrier',barrier_subtype:'striped_board',box:[.1,.2,.4,.5],track_id:7,image_id:3,range_supported:true};
  const tr={id:7,label:'barrier',barrier_subtype:'striped_board',x:8,y:1,z:.5,length:1.1,width:.4,height:1.1,
    confidence:.9,heading:90,coasting:false};
  const frames=Array.from({length:5},(_,i)=>{
    const rgb=i===1;
    return {frame:200+i,sim_time:i/10,run_id:'r',perception_version:'surround-v1',observer_map:{ego:[0,0,0]},
      tracks:[{...tr,cameras:rgb?['front']:[],sources:rgb?['lidar','rgb']:['lidar'],rgb_boxes:rgb?{front:box.box}:{}}],
      camera_detections:rgb?{front:[box]}:{},barrier_review:{7:{confirmed:i!==3,range_only:!rgb,barrier_subtype:'striped_board',
        observations:rgb?[{...box,view:'front',confirmed:true}]:[]}}};
  });
  const reviewed=applyBarrierReview(frames);
  assert.deepEqual(reviewed.map(f=>f.tracks.length),[0,1,1,0,0]);
  assert.deepEqual(reviewed[2].tracks[0].cameras,[]);
  assert.deepEqual(cameraBoxes(reviewed[2],'front'),[]);
  const memory=new ReplayMemory({});
  assert.equal(memory.at(reviewed,2).objects[0].barrierSubtype,'striped_board');
  assert.deepEqual(memory.at(reviewed,3).objects,[]);
});

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
