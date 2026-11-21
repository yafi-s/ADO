import assert from 'node:assert/strict';
import test from 'node:test';
import {FollowHeading} from '../web/follow-heading.mjs';

const degrees = 180 / Math.PI;
const wrap = a => (a + 540) % 360 - 180;
const frame = (t, yaw, run_id='a') => ({sim_time:t, run_id, observer_map:{ego:[0,0,wrap(yaw)]}});

test('camera follows a constant rate through the signed wrap without lag at irregular exposures', () => {
  const frames = Array.from({length:60}, (_,i) => {
    const t = .1 * i + (i % 2) * .025;
    return frame(t, 153 + 20*t);
  });
  const heading = new FollowHeading(frames);
  for (const i of [21,14,38,11,30]) for (const fraction of [0,.3,.85,1]) {
    const time = frames[i].sim_time + fraction * (frames[i+1].sim_time - frames[i].sim_time);
    assert.ok(Math.abs(-heading.at(i,fraction)*degrees - (153+20*time)) < 1e-9);
  }
});

test('filter has finite support and stays within each uninterrupted run', () => {
  const frames = Array.from({length:50}, (_,i) => frame(i/10, i===20 ? 35 : 0));
  const heading = new FollowHeading(frames);
  assert.equal(heading.at(9),0);
  assert.equal(heading.at(30),0);
  assert.ok(-heading.at(20)*degrees > 0 && -heading.at(20)*degrees < 10);
  const cut = new FollowHeading([frame(0,90),frame(.1,90),frame(.2,-85,'b'),frame(2,45,'b')]);
  assert.ok(Math.abs(-cut.at(1,.95)*degrees - 90) < 1e-9);
  assert.ok(Math.abs(-cut.at(2,.6)*degrees + 85) < 1e-9);
  assert.ok(Math.abs(-cut.at(3)*degrees - 45) < 1e-9);
});
