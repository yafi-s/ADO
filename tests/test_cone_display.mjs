import assert from 'node:assert/strict';
import test from 'node:test';
import {readFileSync} from 'node:fs';
import {gunzipSync} from 'node:zlib';
import {applyConeReview, coneReviewer} from '../web/cone-observations.mjs';
import {bearingObservations, cameraBoxes} from '../web/lamp-observations.mjs';
import {ReplayMemory} from '../web/object-memory.mjs';

const fresh={label:'traffic_cone',camera:'front',box:[.3,.4,.4,.7],sources:['rgb','lidar'],
  lidar_hits:3,x:4,y:2,z:.6,length:.6,width:.6,height:1.2};
function exposure(i,extras={}){
  return {frame:100+i,run_id:'trial',sim_time:10+i*.1,observer_map:{ego:[0,0,0]},...extras};
}
test('a measured stationary cone stays registered through a turn, range-only support and short occlusion',()=>{
  const frames=[exposure(0,{cone_observations:[fresh]}),
    exposure(1,{observer_map:{ego:[2,2,90]},cone_observations:[{...fresh,camera:'rear_right',x:0,y:-2}]}),
    exposure(2,{observer_map:{ego:[2,2,90]},lidar_xyz:[[0,-2,.4],[.1,-2,.7]]}),
    exposure(3),exposure(5),exposure(17,{lidar_xyz:[[4,2,.4],[4.1,2,.7]]})];
  const qualified=applyConeReview(frames);
  assert.deepEqual(qualified[1].cone_tracks.map(t=>[t.id,t.world_x,t.world_y]),[['C1',4,2]]);
  assert.equal(qualified[2].cone_tracks[0].range_source,'footprint');
  assert.equal(qualified[2].cone_tracks[0].lidar_hits,2);
  assert.equal(qualified[3].cone_tracks[0].coasting,true);
  assert.deepEqual(cameraBoxes(qualified[3],'front'),[]);
  assert.equal(qualified[4].cone_tracks,undefined);
  assert.equal(qualified[5].cone_tracks,undefined);
  const memory=new ReplayMemory();
  assert.equal(memory.at(qualified,5).objects.length,0);
  const obj=memory.at(qualified,2).objects[0];
  assert.equal(obj.key,'C1'); assert.equal(obj.held,false); assert.equal(obj.z,2);
  assert.equal(memory.at(qualified,3).objects[0].held,true);
});
test('range requires fresh image and LiDAR evidence and cannot come from the observer actors',()=>{
  const frame=exposure(0,{cone_observations:[{...fresh,lidar_hits:0}],
    observer_map:{ego:[0,0,0],others:[{type:'cone',x:4,y:2}]}});
  const qualified=coneReviewer()(frame);
  assert.deepEqual(qualified.cone_tracks,[]);
  assert.equal(bearingObservations(qualified,null,['front'])[0].label,'traffic_cone');
  assert.equal(cameraBoxes(qualified,'front')[0].range_supported,false);
  assert.deepEqual(new ReplayMemory().at([qualified],0).objects,[]);
});
test('identical live polls retain identity and distinct measurements never merge in a single view',()=>{
  const qualify=coneReviewer(), rows=[fresh,{...fresh,y:2.5}];
  const f=exposure(0,{cone_observations:rows}), result=qualify(f);
  assert.deepEqual(result.cone_tracks.map(t=>t.id),['C1','C2']);
  assert.equal(qualify({...f}),result);
  const cross=qualify(exposure(1,{cone_observations:[...rows,{...fresh,camera:'right'}]}));
  assert.deepEqual(cross.cone_tracks.map(t=>t.id),['C1','C2']);
  assert.deepEqual(cross.cone_tracks[0].cameras,['front','right']);
});
test('a sidecar must match recording, frame count, interval and unique native frame numbers',()=>{
  const frames=[exposure(0),exposure(1)];
  const review={schema_version:1,trial_id:'trial',first_frame:100,last_frame:101,capture_frames:2,
    frames:[{frame:101,cone_observations:[fresh]}]};
  assert.equal(applyConeReview(frames,review)[1].cone_tracks.length,1);
  for(const invalid of [{trial_id:'other'},{capture_frames:3},{first_frame:99},{last_frame:102},
    {frames:[...review.frames,...review.frames]},{frames:[{frame:0,cone_observations:[]}]}])
    assert.throws(()=>applyConeReview(frames,{...review,...invalid}));
});

test('only a weak car fitted to the complete known cone surface is withdrawn',()=>{
  const qualify=coneReviewer();
  const weak={id:11,label:'car',cameras:['front'],confidence:.5,
    surface_xyz:[[3.9,2,.4],[4.1,2,.7],[3.95,2.1,.5],[4,1.85,.3]]};
  const strong={...weak,id:12,confidence:.85};
  const nearby={...weak,id:13,surface_xyz:[...weak.surface_xyz,[4,3.5,.9]]};
  qualify(exposure(0,{cone_observations:[fresh],tracks:[strong]}));
  const raw=exposure(1,{cone_observations:[fresh],tracks:[weak,{...strong,confidence:.4},nearby],
    camera_detections:{front:[{label:'car',track_id:11,box:fresh.box},{label:'car',track_id:13,box:fresh.box}]}});
  const result=qualify(raw);
  assert.deepEqual(result.tracks.map(t=>t.id),[12,13]);
  assert.deepEqual(result.cone_conflicts,[{track:weak,cone_id:'C1'}]);
  assert.deepEqual(result.camera_detections.front.map(b=>b.track_id),[13]);
  assert.equal(raw.camera_detections.front.length,2);
  assert.equal(qualify(exposure(2,{tracks:[{...weak,coasting:true,cameras:[]}]})).cone_conflicts[0].cone_id,'C1');
  const recovered=qualify(exposure(3,{tracks:[{...weak,surface_xyz:nearby.surface_xyz}]}));
  assert.equal(recovered.tracks.length,1);
});

test('the archived urban drive keeps all six cone identities, including coast and return-only frames',
  {skip:process.env.ADO_RECORDING_QA!=='1'},()=>{
  const media=new URL('../web/media/urban-pace/',import.meta.url);
  const packed=JSON.parse(gunzipSync(readFileSync(new URL('replay-pack.json.gz',media))));
  const review=JSON.parse(readFileSync(new URL('cone-observations.json',media)));
  const evidence=JSON.parse(readFileSync(new URL('../results/cone-perception/display-association.json',import.meta.url)));
  const observations=new Map(review.frames.map(row=>[row.frame,row.cone_observations]));
  const qualify=coneReviewer(),counts={rgb_lidar:0,lidar_continued:0,remembered:0},ids=new Map(),conflicts=new Map();
  for(const raw of packed.replay){
    const pose=raw.observer_map;
    const qualified=qualify({...raw,observer_map:{ego:pose.ego,ego_z:pose.ego_z}},observations.get(raw.frame)??[]);
    const used=new Set();
    for(const cone of qualified.cone_tracks??[]){
      const actor=pose.others.find(a=>a.type==='cone'&&Math.hypot(cone.world_x-a.x,cone.world_y-a.y)<=.65);
      assert.ok(actor,`Unmatched ${cone.id} at ${raw.frame}`);
      assert.ok(!used.has(actor.actor_id));used.add(actor.actor_id);
      if(ids.has(cone.id))assert.equal(ids.get(cone.id),actor.actor_id);
      ids.set(cone.id,actor.actor_id);
      counts[cone.coasting?'remembered':cone.range_source==='footprint'?'lidar_continued':'rgb_lidar']++;
    }
    for(const {track} of qualified.cone_conflicts??[])conflicts.set(track.id,(conflicts.get(track.id)??0)+1);
  }
  assert.deepEqual([...ids.values()].sort(),[609,610,611,612,613,614]);
  assert.deepEqual(counts,Object.fromEntries(Object.entries(evidence.drawn_by_evidence).map(([k,v])=>[k,v.frames])));
  assert.deepEqual(Object.fromEntries(conflicts),Object.fromEntries(Object.entries(evidence.surface_conflicts).map(([k,v])=>[k,v.count])));
});
