import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {gunzipSync} from 'node:zlib';
import test from 'node:test';
import {bicycleAssociations} from '../web/rider-association.mjs';

const folder=new URL('../results/perception/rider-association/',import.meta.url);
const archive=readFileSync(new URL('high-samples.json.gz',folder));
const manifest=JSON.parse(readFileSync(new URL('manifest.json',folder)));
const native=JSON.parse(gunzipSync(archive));
const selected=native.findIndex(f=>f.frame===3435);

test('native High sample combines only one bike and person, after their joint motion is observed',()=>{
  assert.equal(createHash('sha256').update(archive).digest('hex'),manifest.sample_sha256);
  const active=native.flatMap((f,i)=>[...bicycleAssociations(native,i)].map(([id,e])=>{
    assert.equal(id,143); assert.equal(e.image_id,496); assert.equal(e.camera,'front');
    assert.ok(e.paired_observations>=3);
    assert.ok(f.tracks.some(t=>t.id===id && t.label==='person'));
    return f.frame;
  }));
  assert.deepEqual(active,Array.from({length:23},(_,i)=>3435+i));
  const e=bicycleAssociations(native,selected).get(143);
  assert.equal(e.depth_source,'person lidar');
  assert.deepEqual(bicycleAssociations(native.slice(0,selected+1)),bicycleAssociations(native,selected));
});

test('ambiguous people, independently ranged bikes and absent sensor depth cannot create a composite',()=>{
  for(const change of [
    f=>{for(const boxes of Object.values(f.camera_detections))for(const p of [...boxes])
      if(p.track_id===143) boxes.push({...p,track_id:null,range_supported:false});},
    f=>{for(const boxes of Object.values(f.camera_detections))for(const b of boxes)
      if(b.label==='bicycle') b.range_supported=true;},
    f=>{for(const t of f.tracks)t.sources=[];},
    f=>{f.perception_version='old';},
  ]){
    const frames=structuredClone(native.slice(selected-8,selected+1));
    frames.forEach(change);
    assert.equal(bicycleAssociations(frames).size,0);
  }
});

test('losing temporal continuity or seeing the partner separate immediately rejects the association',()=>{
  const frames=structuredClone(native.slice(selected-8,selected+1));
  frames.at(-1).frame+=20;
  assert.equal(bicycleAssociations(frames).size,0);
  const f=native.findIndex(r=>r.frame===3439);
  const separated=structuredClone(native.slice(f-12,f+1));
  const bike=separated.at(-1).camera_detections.front.find(d=>d.label==='bicycle');
  bike.box=[.75,.55,.79,.69];
  assert.equal(bicycleAssociations(separated).size,0);
});

test('one native return below the person-only region is consistent with measured rider depth',()=>{
  const f=native.find(row=>row.frame===3449), boxes=f.camera_detections.front;
  const person=boxes.find(b=>b.track_id===143).box, bike=boxes.find(b=>b.image_id===496).box;
  const inside=(u,v,b)=>u>=b[0]&&u<=b[2]&&v>=b[1]&&v<=b[3];
  const points=f.lidar_xyz.filter(([x,y,z])=>{
    const d=x-1.3,u=.5+.5*y/d,v=.5-(320/360)*(z-1.65)/d;
    return d>0 && inside(u,v,bike) && !inside(u,v,person);
  });
  assert.deepEqual(points,[[7.99,-3.05,.28]]);
  const rider=f.tracks.find(t=>t.id===143).surface_xyz.map(p=>p[0]-1.3).sort((a,b)=>a-b);
  const middle=rider[Math.floor(rider.length/2)];
  assert.ok(Math.abs(points[0][0]-1.3-middle)<.3);
});
