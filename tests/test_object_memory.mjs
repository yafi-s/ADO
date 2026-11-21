import assert from 'node:assert/strict';
import test from 'node:test';
import {readFileSync} from 'node:fs';

import {ObjectMemory, ReplayMemory, advance, contextOf, createState, laneAxes} from '../web/object-memory.mjs';
import {measuredHeads, cameraLamp, cameraBoxes, inspectionCamera} from '../web/lamp-observations.mjs';
import {cornerReturns, lidarCount} from '../web/sensor-returns.mjs';

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

test('only a coast whose repeated surface belongs to a newly measured striped board loses its car symbol',()=>{
  const patch=[[-.4,0,.2],[.4,0,.2],[-.4,0,.75],[.4,0,.75],[0,0,.55]];
  function replay({separation=0,score=.46,fresh=false,approved=true,extraPoint=false}={}){
    const frames=repeat(6,i=>{
      const points=patch.map(([u,v,z])=>[8+u,v+separation,z]);
      if(extraPoint)points.push([10,.3,.7]);
      const vehicle=car(1,9.9,separation,{confidence:score,cameras:['front'],lidar_hits:9,surface_xyz:points});
      const board=car(2,8,0,{label:'barrier',barrier_subtype:'striped_board',confidence:.9,
        length:1.2,width:.35,height:1,z:.5,cameras:['front'],lidar_hits:9,surface_xyz:patch.map(([u,v,z])=>[8+u,v,z])});
      const observed=i<5||fresh?[vehicle]:[];
      if(i===5)observed.push(board);
      const boxes=observed.map(t=>({label:t.label,track_id:t.id,barrier_subtype:t.barrier_subtype,score:t.confidence,
        image_id:t.id,range_supported:true,box:[.45,.45,.55,.65]}));
      return {frame:i,sim_time:i/10,perception_version:'surround-v1',observer_map:{ego:[0,0,0]},
        tracks:observed,camera_detections:{front:boxes},barrier_review:{cameras:i===5?[{view:'front',box:boxes.at(-1),confirmed:approved}]:[]}};
    });
    const memory=new ReplayMemory({});
    assert.ok(memory.at(frames,4).objects.some(o=>o.label==='car'),'future board classification cannot change earlier history');
    return memory.at(frames,5).objects.map(o=>o.label);
  }
  assert.deepEqual(replay(),['barrier']);
  for(const control of [{separation:1.15},{extraPoint:true},{score:.85},{fresh:true},{approved:false}]){
    assert.deepEqual(replay(control),['car','barrier']);
  }
});

test('a single ranged rider keeps one identity through intermittent mounted bicycle classifications',()=>{
  const frames=repeat(13,i=>{
    const mounted=i===4||i===8;
    const sample=person(73,8,.5,{confidence:.9,lidar_hits:8,cameras:['front'],sources:['lidar','rgb']});
    if(mounted)Object.assign(sample,{label:'bicycle',length:1.7,width:.55,cycle_association:{
      range_source:'shared rider/bicycle lidar',paired_observations:4,surface_hits:8,age_s:0}});
    return {frame:i,sim_time:i/10,perception_version:'surround-v1',observer_map:{ego:[0,0,0]},
      tracks:[sample,person(74,8,2.8,{confidence:.9})]};
  });
  const memory=new ReplayMemory({});
  const baseline=memory.at(frames,3).objects.find(o=>o.sourceId===73).key;
  for(let i=4;i<frames.length;i++){
    const outputs=memory.at(frames,i).objects,owned=outputs.filter(o=>o.sourceId===73);
    assert.equal(owned.length,1);
    assert.equal(owned[0].key,baseline);
    assert.equal(owned[0].label,'bicycle');
    assert.equal(outputs.find(o=>o.sourceId===74).label,'person');
  }
});

test('a weak car box on a long measured surface stays inspectable as an unclassified footprint',()=>{
  const frames=repeat(8,i=>{
    const clear=i>=4;
    const surface=clear?[[9,-.7,0],[10,.8,.7],[11,0,1.2]]:[[6,0,0],[10,.1,.5],[15,.1,.7]];
    const score=clear?.82:.44;
    return {frame:i,sim_time:i/10,perception_version:'surround-v1',observer_map:{ego:[0,0,0]},
      tracks:[car(2,10,0,{confidence:score,cameras:['front'],surface_xyz:surface,lidar_hits:8})],
      camera_detections:{front:[{label:'car',image_id:10,score,range_supported:true,track_id:2}]}};
  });
  const memory=new ReplayMemory({});
  assert.equal(memory.at(frames,3).objects[0].label,'unknown');
  assert.equal(memory.at(frames,3).objects[0].sourceId,2);
  assert.equal(memory.at(frames,7).objects[0].label,'car');
  assert.equal(frames[3].tracks[0].label,'car');
});

test('the first ranging observation can confirm a repeatedly seen pedestrian while image-only boxes stay unplaced',()=>{
  const frames=repeat(10,i=>({frame:i,sim_time:i/10,perception_version:'surround-v1',observer_map:{ego:[0,0,0]},
    camera_detections:{front:[{image_id:11,label:'person',score:i===9?.46:.73,track_id:i===9?9:null,range_supported:i===9}]},
    tracks:i===9?[person(9,24,4,{cameras:['front'],confidence:.46,lidar_hits:4,
      sources:['rgb','lidar','radar'],radial_mps:-3})]:[]}));
  const memory=new ReplayMemory({});
  assert.deepEqual(memory.at(frames,8).objects,[]);
  assert.equal(memory.at(frames,9).objects[0].label,'person');
});

test('a confirmed post-and-chain subtype reaches the renderer with measured confidence',()=>{
  const frames=repeat(4,i=>({frame:i,sim_time:i/10,perception_version:'surround-v1',observer_map:{ego:[0,0,0]},
    camera_detections:{front:[]},tracks:[car(2,8,0,{label:'barrier',barrier_subtype:'chain',subtype_confidence:.68,
      length:1.64,width:.24,height:.70,cameras:['front'],sources:['rgb','lidar'],confidence:.68,lidar_hits:8})]}));
  const drawn=new ReplayMemory({}).at(frames,3).objects[0];
  assert.equal(drawn.label,'barrier');
  assert.equal(drawn.barrierSubtype,'chain');
  assert.equal(drawn.barrierSubtypeConfidence,.68);
});

test('a range-supported work-zone barrier stays distinct from people and vehicles and expires when its depth stops',()=>{
  const frames=repeat(19,i=>({frame:i,sim_time:i/10,perception_version:'surround-v1',observer_map:{ego:[0,0,0]},
    camera_detections:{front:[]},tracks:i<4?[car(2,8,0,{label:'barrier',length:1.19,width:.40,height:1.09,
      cameras:['front'],sources:['rgb','lidar'],confidence:.8,lidar_hits:8}),person(5,8,1)]:[]}));
  const memory=new ReplayMemory({});
  assert.equal(memory.at(frames,3).objects.filter(p=>p.family==='barrier').length,1);
  assert.equal(memory.at(frames,3).objects.filter(p=>p.family==='person').length,1);
  assert.ok(!memory.at(frames,18).objects.some(p=>p.family==='barrier'));
});

test('continuous playback, cold chapter jumps and reverse seeks reconstruct identical rear tracks', () => {
  const frames = repeat(235, i => ({
    frame: i, sim_time: i / 10, scenario: 'test', perception_version: 'surround-v1',
    observer_map: {ego: [0, 0, 0]}, tracks: i < 200 ? [car(8,-14,3,{cameras:['rear_right']})] : []
  }));
  const memory = new ReplayMemory({});
  const continuous = frames.map((_,i) => memory.at(frames,i).objects);
  assert.equal(continuous[182].length, 1);
  const seek = new ReplayMemory({});
  for (const i of [182,8,234,214,50,181,102,99,100]) assert.deepEqual(seek.at(frames,i).objects, continuous[i]);
  assert.ok(!continuous.at(-1).length || continuous.at(-1)[0].leaving);
});

test('a range-supported nearby pedestrian appears on its first RGB and Doppler observation', () => {
  const output = play([...repeat(15,()=>({})),{tracks:[person(6,9,.1,{sources:['rgb','lidar','radar'],speed_mps:null,confidence:.39})]}]);
  assert.equal(output.at(-1)[0].label,'person');
  assert.ok(output.at(-1)[0].opacity>.65);
});

test('a resolved pedestrian is shown through the next coast outside the close-range score gate', () => {
  const measured=person(18,15.314,2.228,{confidence:.74,speed_mps:null,cameras:['front'],sources:['rgb','lidar','radar'],
    lidar_hits:5,radial_mps:-5.62});
  const observe=t=>play([...repeat(12,()=>({})),{tracks:[t]},{tracks:[coasting(t)]}]);
  assert.equal(observe(measured).at(-1)[0].sourceId,18);
  assert.deepEqual(observe({...measured,lidar_hits:2}).at(-1),[]);
  assert.deepEqual(observe({...measured,radial_mps:null}).at(-1),[]);
});

test('track inspection uses a recorded camera association and no rectangle during a coast', () => {
  const track={id:6,cameras:['front','left'],last_cameras:['rear_left'],rgb_boxes:{left:[.1,.2,.3,.8]}};
  assert.equal(inspectionCamera(track,['chase','front','left'],'left'),'left');
  assert.equal(inspectionCamera(track,['chase','front'],'chase'),'front');
  assert.equal(inspectionCamera(track,['chase'],'chase'),null);
  assert.deepEqual(cameraBoxes({tracks:[track]},'left').map(item=>item.track_id),[6]);
  const coast={...track,cameras:[],coasting:true};
  assert.equal(inspectionCamera(coast,['rear_left'],'chase'),'rear_left');
  assert.deepEqual(cameraBoxes({tracks:[coast]},'rear_left'),[]);
  assert.deepEqual(cameraBoxes({tracks:[coast]},'left'),[]);
});

test('wheel contact follows the grade at the object and ignores changes in visible body surface', () => {
  const slope=laneAxes([[[0,0,0],[4,0,.4],[8,0,.8],[12,0,1.2]]]);
  const frames=repeat(5,i=>({frame:i,sim_time:i*.1,observer_map:{ego:[0,0,0],ego_z:0},
    tracks:[car(7,6,0,{z:.8+i*.1,height:1.5})]}));
  const replay=new ReplayMemory({laneAxis:slope});
  for(let i=0;i<frames.length;i++)assert.ok(Math.abs(replay.at(frames,i).objects[0].y-.615)<1e-6);
  assert.equal(slope.ground(6,4,1),null,'an unmeasured sidewalk does not get the adjacent lane elevation');
});

test('a wheel track cannot snap to a newly exposed face while Doppler still reports a parked car', () => {
  const lanes=laneAxes([lane]);
  const frames=repeat(24,i=>{
    const x=16+(i<16?0:3.8),radar=[[x,-.2,0],[x,0,0],[x,.2,0]];
    return {frame:i,sim_time:i*.1,perception_version:'surround-v1',observer_map:{ego:[0,0,0],ego_z:0},radar,
      tracks:[car(1,x,0,{doppler:{radar:{hits:10,radial_mps:0}},lidar_hits:12,confidence:.8})]};
  });
  const replay=new ReplayMemory({laneAxis:lanes});
  const drawn=frames.map((_,i)=>replay.at(frames,i).objects[0]);
  assert.ok(drawn.slice(1).every((obj,i)=>Math.abs(obj.x-drawn[i].x)<.051));
});

test('two aligned overlapping car symbols are deduplicated without merging separate nearby lanes', () => {
  const frames=repeat(5,i=>({frame:i,sim_time:i*.1,perception_version:'surround-v1',observer_map:{ego:[0,0,0]},
    tracks:[car(2,11,1,{cameras:['front'],lidar_hits:7}),car(8,11.8,1.2,{cameras:['right'],lidar_hits:25}),
      car(9,11,4.5,{cameras:['right'],lidar_hits:8})]}));
  const objects=new ReplayMemory({}).at(frames,4).objects;
  assert.equal(objects.length,2);
  assert.deepEqual(objects.find(o=>o.sourceIds).sourceIds,[2,8]);
});

test('a new unknown range surface fades promptly when fresh point support stops', () => {
  const frames=repeat(14,i=>({frame:i,sim_time:i/10,perception_version:'surround-v1',
    observer_map:{ego:[0,0,0]},tracks:i<4?[car(10,8,3,{label:'unknown',sources:['lidar'],cameras:[],
      confidence:.8,lidar_hits:8})]:[],lidar:[]}));
  const m=new ReplayMemory({});
  assert.equal(m.at(frames,3).objects[0].label,'unknown');
  assert.equal(m.at(frames,9).objects.length,0);
});

test('the clear-pass rear-camera class handoff keeps one physical lead with both inspectable source IDs', () => {
  const read=path=>JSON.parse(readFileSync(new URL(path,import.meta.url),'utf8'));
  const frames=read('../web/media/overtake-clear/replay.json');
  const lanes=read('../web/media/overtake-clear/town.json').roads;
  const features=read('../web/media/map-features/Town03_Opt.json');
  const ends=(a,b)=>[a,b].map(p=>p.slice(0,2).map(n=>n.toFixed(2)).join(',')).join('|');
  const directions=new Map(features.lanes.map(l=>[ends(l.a,l.b),l.forward]));
  const axis=laneAxes(lanes,lanes.map(l=>directions.get(ends(l[0],l.at(-1)))));
  const memory=new ReplayMemory({laneAxis:axis,laneDistance:axis.distance});
  for(const n of [7882,7884,7888,7891]){
    const objects=memory.at(frames,frames.findIndex(r=>r.frame===n)).objects;
    const lead=objects.filter(o=>o.sourceIds?.some(id=>id===4||id===5)||[4,5].includes(o.sourceId));
    assert.equal(lead.length,1,`frame ${n}`);
    assert.ok(lead[0].sourceIds.includes(4)&&lead[0].sourceIds.includes(5));
    assert.equal(lead[0].label,'car');
  }
});

test('a lidar-only mover gains a camera class under its original display identity', () => {
  const frames = repeat(20, i => ({
    frame:i,sim_time:i/10,perception_version:'surround-v1',observer_map:{ego:[0,0,0]},
    tracks:[car(31,-12,2,{label:i<10?'unknown':'car',cameras:i<10?[]:['rear_right'],
      sources:i<10?['lidar']:['rgb','lidar']})]
  }));
  const memory = new ReplayMemory({});
  const first = memory.at(frames,8).objects;
  const later = memory.at(frames,15).objects;
  assert.equal(first.length,1); assert.equal(later.length,1);
  assert.equal(first[0].label,'unknown'); assert.equal(later[0].label,'car');
  assert.equal(first[0].key,later[0].key);
});

test('signals retain distinct surveyed head locations and require an explicit camera color', () => {
  const heads = [{id:'l/1',map_id:7,x:4,y:5,z:4,state:'green',source:'camera'},
    {id:'l/2',map_id:7,x:4,y:9,z:4,state:'green',source:'map'}];
  const frame = {phase:'red signal',signals:[...heads,{id:8,x:1,y:2}],observer_map:{signals:[{id:7,state:'red'}]}};
  assert.deepEqual(measuredHeads(frame),heads);
  assert.deepEqual(measuredHeads(frame).map(cameraLamp),['green','unknown']);
  assert.deepEqual(measuredHeads({phase:'red signal'}),[]);
});

test('a colored head and a bearing-only object overlay only their contributing camera', () => {
  const measurement={id:'h1',x:10,y:3,z:5,view:'right',state:'yellow',source:'camera',
    pixel_size:[320,180],pixel_box:[96,18,112,36]};
  const frame={signals:[measurement],camera_detections:{right:[{label:'person',box:[.8,.4,.9,.7],range_supported:false}]}};
  assert.deepEqual(cameraBoxes(frame,'front'),[]);
  assert.deepEqual(cameraBoxes(frame,'chase'),[]);
  const right=cameraBoxes(frame,'right');
  assert.equal(right[0].range_supported,false);
  assert.equal(right[1].lamp,'yellow');
  assert.deepEqual(right[1].box,[.3,.1,.35,.2]);
});

test('corner radar height and signed Doppler stay distinct and XYZ takes precedence for lidar counts', () => {
  const frame = {lidar:[[1,2]],lidar_xyz:[[1,2,.5],[3,4,1.1]],
    corner_radar:{radar_rear_left:[[-5,-3,1.4,-3.2]],radar_front_right:[[2,3,NaN,4]]}};
  assert.deepEqual(cornerReturns(frame),[[-5,-3,1.4,-3.2]]);
  assert.equal(lidarCount(frame),2);
});

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

test('a remembered object leaves when something farther off is seen where it stood', () => {
  // A car waits 8 m ahead, then pulls away unmeasured. The detector still boxes it, ever smaller.
  const waiting = repeat(10, i => ({tracks: [i % 2 ? coasting(car(1, 8, 0)) : car(1, 8, 0)]}));
  const boxAt = range => ({label: 'car', score: .5, box: [.5 - .46 / (range - 1.3), .4, .5 + .46 / (range - 1.3), .6]});
  const stays = play([...waiting, ...repeat(15, () => ({detections: [boxAt(8)]}))]);
  assert.equal(stays[24].length, 1, 'a box its own size keeps it');
  const leaves = play([...waiting, ...repeat(15, i => ({detections: [boxAt(8 + (i + 1) * 1.5)]}))]);
  const gone = leaves.findIndex((objects, index) => index >= 10 && !objects.some(object => !object.leaving));
  assert.ok(gone > 0 && gone <= 20, `still held at frame ${gone}`);
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
