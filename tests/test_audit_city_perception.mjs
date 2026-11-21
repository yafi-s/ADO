import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtempSync,writeFileSync,rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import path from 'node:path';
import {audit,match,sector} from '../tools/audit_city_perception.mjs';
import {align} from '../tools/compare_city_encounters.mjs';

const at=(id,x,label='car')=>({id,x,y:0,label});
const ids=pairs=>pairs.map(p=>[p.actor.id,p.prediction.id]);

test('assignment finds the maximum number of distinct actors before minimizing residual',()=>{
  // The first prediction can explain either actor; the second can only explain A.
  const actors=[at('A',0),at('B',3)];
  assert.deepEqual(ids(match(actors,[at(1,1),at(2,-2)])),[['A',2],['B',1]]);
  assert.deepEqual(ids(match(actors,[at(1,.9),at(2,1.6)])),[['A',1],['B',2]]);
});

test('duplicates, family mismatches and distances above the inclusive gate do not earn hits',()=>{
  assert.equal(match([at('A',0)],[at(1,.1),at(2,.2)]).length,1);
  assert.deepEqual(ids(match([at('A',0,'person'),at('B',4,'bus')],[at(1,0),at(2,7)])),[['B',2]]);
  assert.deepEqual(match([at('B',4)],[at(1,7.001)]),[]);
  assert.deepEqual(match([at('A',0)],[at(1,0,'unknown')]),[]);
  assert.deepEqual(ids(match([at('A',0)],[at(1,0,'unknown')],3,false)),[['A',1]]);
  assert.deepEqual(match([],[]),[]);
  assert.deepEqual(match([],[at(1,0)]),[]);
});

test('CARLA local Y points right, with 90-degree mutually exclusive bearing sectors',()=>{
  assert.deepEqual([[1,1],[1,-1],[0,-1],[-1,1],[-1,-1],[0,1]].map(([x,y])=>sector(x,y)),
    ['front','front','left','rear','rear','right']);
});

test('journal scoring handles rotated ego poses and separates missing, switched and out-of-range frames',async t=>{
  const dir=mkdtempSync(path.join(tmpdir(),'ado-audit-test-'));
  t.after(()=>rmSync(dir,{recursive:true,force:true}));
  const town={producer:{map:'test'},roads:[]};
  writeFileSync(path.join(dir,'town.json'),JSON.stringify(town));
  const rows=[1,null,2,3,null,4].map((id,index)=>({
    frame:100+index,sim_time:index/10,tracks:[{id:'parked-track',label:'car',x:10,y:0},
      ...(id==null?[]:[{id,label:'person',x:0,y:-10}])],
    observer_map:{ego:[50,70,90],others:[{actor_id:10,type:'pedestrian',role:'crossing',x:index===4?95:60,
      y:70,yaw:0,dimensions_m:[.6,1.8,.4]}]},
  }));
  writeFileSync(path.join(dir,'perception.jsonl'),rows.map(JSON.stringify).join('\n')+'\n{"frame"');
  const staticFile=path.join(dir,'parked.json');
  writeFileSync(staticFile,JSON.stringify({map:'test',actors:[{actor_id:'level-test',type:'car',x:50,y:80,
    dimensions_m:[4,1.5,2],yaw:90,role:'parked-map-vehicle'}]}));
  const result=await audit(dir,{staticFile});
  assert.equal(result.frames,6);
  assert.equal(result.partial_tail_ignored,true);
  assert.deepEqual(result.tracker.sector.left,{actor_frames:5,matched:4,exact_label_hits:4,
    prediction_frames:4,unmatched_predictions:0});
  assert.equal(result.tracker.reacquisitions,1);
  assert.equal(result.tracker.direct_id_switches,1);
  assert.equal(result.tracker.actors[0].distinct_ids,4);
  assert.equal(result.display.overall.matched,0,'the observer list must never create display objects');
  assert.equal(result.tracker.overall.unmatched_predictions,6,'scenario-only counts are preserved');
  assert.equal(result.with_static_road_users.tracker.overall.unmatched_predictions,0);
  assert.equal(result.with_static_road_users.display.overall.matched,0,'parked references must never create drawings');
});

test('encounter pairing is chronological, one-to-one, and excludes different vehicle or actor poses',()=>{
  const sample=(frame,x,sector='front')=>({frame,ego:[x,0,359],progress:x,actor:{x:12,y:0,sector}});
  const base=[sample(1,0),sample(2,.3),sample(3,1),sample(4,4)];
  const next=[sample(10,.2),sample(11,1),sample(12,4,'left'),sample(13,4)];
  next[0].ego[2]=1;
  next[3].actor.x=14;
  assert.deepEqual(align(base,next).map(([a,b])=>[a.frame,b.frame]),[[2,10],[3,11]]);
});
