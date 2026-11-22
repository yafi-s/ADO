import test from 'node:test';
import assert from 'node:assert/strict';
import {maneuverActivity} from '../web/maneuver-status.mjs';

test('only explicit maneuver telemetry supplies an activity', () => {
  assert.equal(maneuverActivity(undefined), null);
  assert.equal(maneuverActivity({maneuver: 'passing', committed: true}), null);
  const plan = {planner: 'route-maneuvers-v1', maneuver_phase: 'keep'};
  assert.equal(maneuverActivity(plan), null);
  assert.equal(maneuverActivity({...plan, maneuver_phase: 'passing', committed: false}), null);
  assert.equal(maneuverActivity({...plan, maneuver_phase: 'yield'}).state, 'waiting');
  assert.equal(maneuverActivity({...plan, maneuver_phase: 'route_merge'}).label, 'Merging for the route');
  assert.equal(maneuverActivity({...plan, maneuver_phase: 'passing', committed: true, lead_kind: 'static'}).label, 'Passing a stationary object');
  assert.equal(maneuverActivity({...plan, maneuver_phase: 'wait_rejoin'}).label, 'Waiting to return');
  assert.equal(maneuverActivity({...plan, maneuver_phase: 'rejoined'}).label, 'Pass complete');
  assert.equal(maneuverActivity({...plan, maneuver_phase: 'change_right', return_reason: 'departing_lead'}).state, 'following');
  assert.equal(maneuverActivity({...plan, maneuver_phase: 'resumed_following'}).label, 'Following the vehicle ahead');
});
