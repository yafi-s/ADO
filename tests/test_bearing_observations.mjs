import assert from 'node:assert/strict';
import test from 'node:test';
import {bearingObservations, cameraBoxes} from '../web/lamp-observations.mjs';

const person = {label: 'person', box: [.83, .31, .94, .88], image_id: 7, range_supported: false, track_id: null};
const fresh = {
  run_id: 'surround', frame: 827, sim_time: 10,
  camera_detections: {left: [person], front: [{...person, image_id: 7, range_supported: true, track_id: 1}]},
  camera_inference_age_s: {left: 0, front: 0, right: .1}
};
const skipped = {
  run_id: 'surround', frame: 828, sim_time: 10.1,
  camera_detections: {right: []}, camera_inference_age_s: {left: .1, front: .1, right: 0}
};
const views = ['front', 'left', 'right'];

test('a fresh unsupported person gets its own identifiable image observation, then holds the recorded bounds for the next skipped frame', () => {
  const items = bearingObservations(fresh, null, views);
  assert.deepEqual(items, [{key:'["left",7]', view:'left', label:'person', box:person.box, age_s:0}]);
  assert.deepEqual(bearingObservations(skipped, fresh, views), [{...items[0], age_s:.1}]);
  assert.deepEqual(cameraBoxes(skipped, 'left'), []);
});

test('a new inference replaces the observation even when empty or already localized', () => {
  const rerun = {...skipped, camera_inference_age_s: {left: 0}};
  assert.deepEqual(bearingObservations({...rerun, camera_detections:{left:[]}}, fresh, views), []);
  const localized = {...person, range_supported: true, track_id: 1};
  assert.deepEqual(bearingObservations({...rerun, camera_detections:{left:[localized]}}, fresh, views), []);
});

test('seek boundaries, run changes, gaps and missing evidence cannot retain an observation', () => {
  const expired = {...skipped, sim_time: 10.2, camera_inference_age_s:{left:.2}};
  for (const frame of [expired, {...skipped, run_id:'other'}, {...skipped, frame:830},
    {...skipped, sim_time:9.9}, {...skipped, camera_inference_age_s:{}}, {...skipped, camera_detections:undefined}]) {
    assert.deepEqual(bearingObservations(frame, fresh, views), []);
  }
  assert.deepEqual(bearingObservations(skipped, null, views), []);
  assert.deepEqual(bearingObservations({...skipped, frame:829, sim_time:10.2}, skipped, views), []);
});

test('legacy front overlays retain unlabeled range evidence without creating an unsupported row or applying hold', () => {
  const legacy = {...person, image_id:undefined, range_supported:undefined};
  const front = {frame:3, sim_time:.3, detections:[legacy]};
  assert.deepEqual(cameraBoxes(front, 'front'), [legacy]);
  assert.deepEqual(bearingObservations(front, null, views), []);
  const recordedBearing = {...front, detections:[{...legacy, range_supported:false}], detection_age_s:0};
  assert.equal(bearingObservations(recordedBearing, null, views)[0].view, 'front');
  assert.deepEqual(bearingObservations({...recordedBearing, frame:4, sim_time:.4, detection_age_s:.1}, recordedBearing, views), []);
});
