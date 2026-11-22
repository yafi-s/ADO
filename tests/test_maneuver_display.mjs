import assert from 'node:assert/strict';
import {spawnSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import test from 'node:test';

test('display audit distinguishes a linked rectangle from an ambiguous actor attribution', t => {
  const stage = fs.mkdtempSync(path.join(os.tmpdir(), 'ado-display-test-'));
  t.after(() => fs.rmSync(stage, {recursive:true, force:true}));
  fs.mkdirSync(path.join(stage, 'raw'));
  const hashes = {};
  function write(name, value) {
    const bytes = typeof value === 'string' ? value : JSON.stringify(value);
    fs.writeFileSync(path.join(stage, 'raw', name),bytes);
    hashes[name] = createHash('sha256').update(bytes).digest('hex');
  }
  const cone = {actor_id:201, type:'cone', role:'in-lane-obstruction', x:10, y:0, z:0,
    yaw:0, dimensions_m:[.7,1,.7], bbox_offset_m:[0,0,.5]};
  // This separate body projects to the same rectangle. It must block unique attribution.
  const competitor = {...cone, actor_id:202, role:'roadworks', x:18.7, z:-1.65,
    dimensions_m:[1.4,2,1.4], bbox_offset_m:[0,0,1]};
  const observation = {label:'traffic_cone', camera:'front', box:[.479,.563,.521,.676],
    confidence:.9, x:10, y:0, lidar_hits:5, sources:['lidar']};
  const event = {kind:'in-lane-obstruction', exposed_s:.1};
  const raw = Array.from({length:3}, (_,i) => ({frame:100+i, sim_time:i/10,
    observer_map:{ego:[0,0,0], ego_z:0, others:i===1?[cone,competitor]:[cone]},
    cone_observations:i===2?[]:[observation], camera_inference_age_s:{front:.1}}));
  const scores = raw.map(({frame,sim_time},i) => ({
    frame, sim_time, camera_frame:frame, video_index:i, video_time:i/10,
    ego_pose:{x:0,y:0,z:0,pitch:0,yaw:0,roll:0},
    events:i>0?[event]:[], tactical_pass:{planner:'route-maneuvers-v1', maneuver_phase:'yield'}
  }));
  write('telemetry.json', scores);
  write('town.json', {producer:{map:'Town03_Opt'},roads:[[[0,0,0],[50,0,0]]]});
  write('perception.jsonl',raw.map(row=>JSON.stringify(row)+'\n').join(''));
  fs.writeFileSync(path.join(stage, 'capture.json'),JSON.stringify({
    trial_id:'synthetic', first_frame:100, last_frame:102, perimeter_views:['left','right','rear_left','rear_right'],
    summary:{frames:3,events:[event]}, raw_sha256:hashes
  }));
  const output = path.join(stage, 'display.json');
  const check = fileURLToPath(new URL('../tools/check_maneuver_display.mjs', import.meta.url));
  const result = spawnSync(process.execPath,[check,stage,output],{encoding:'utf8',timeout:10000});
  assert.equal(result.status,0,result.stderr);
  const report = JSON.parse(fs.readFileSync(output));
  const role = report.roles['in-lane-obstruction'];
  assert.equal(role.nearby_actor_frames,3);
  assert.equal(role.matched_drawings,2);
  assert.equal(role.fresh_camera_and_range,2);
  assert.equal(role.camera_geometry_confirmed,1);
  assert.equal(role.actors['201'].first_by_camera.front.frame,100);
  const highlight=report.highlights['in-lane-obstruction/exposed_s'];
  assert.equal(highlight.planner.state,'waiting');
  assert.equal(highlight.view,'front');
  assert.equal(highlight.seconds,.1);
  assert.equal(highlight.objects[0].selected_camera_linked,false);
  assert.equal(highlight.objects[0].attribution[0].other_body_iou,highlight.objects[0].attribution[0].body_iou);
});
