// Compare two displays on exactly the same release measurements and references,
// even after the website's featured recording changes.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {audit,match,sector} from './audit_city_perception.mjs';
import {motion} from './audit_city_motion.mjs';
import * as currentRider from '../web/rider-association.mjs';

const root=fileURLToPath(new URL('../',import.meta.url));
const release='a7a00120ff6e3f2e266db7de1880e31c38289553';
const archive=file=>execFileSync('git',['show',release+':'+file],{cwd:root,maxBuffer:96*1024*1024});
const digest=value=>createHash('sha256').update(value).digest('hex');
const hashSource=value=>digest(value.toString().replaceAll('\r\n','\n'));
const close=p=>Math.hypot(p.x,p.y)<=35;
const add=(map,key)=>{map[key]=(map[key]??0)+1;};
const sourceCases={hedge:[23,24],residential_frontage:[50,55],shelter:[70]};
const output=process.argv[2];
if(!output)throw Error('Pass a JSON output path');

async function score(directory,staticFile,revision,rider,isHigh){
  const history=[], associations=[], cases={},overlay={scenario:{hits:0,unmatched:0},with_parked:{hits:0,unmatched:0}};
  const options={staticFile,memoryRevision:revision};
  const coverage=await audit(directory,{...options,onFrame({frame,actors,tracker,display}){
    const {frame:number,sim_time,perception_version,run_id,scenario,seed,camera_detections,tracks}=frame;
    history.push({frame:number,sim_time,perception_version,run_id,scenario,seed,camera_detections,tracks});
    if(history.length>(rider.RIDER_HISTORY_FRAMES??13))history.shift();
    const hints=rider.bicycleAssociations(history);
    const shown=display.map(p=>p.label==='person'&&hints.has(p.sourceId)?{...p,label:'bicycle',hint:hints.get(p.sourceId)}:p);
    for(const [name,truth] of [['scenario',actors.filter(a=>a.role!=='parked-map-vehicle')],['with_parked',actors]]){
      const pairs=match(truth,shown);
      const matched=new Map(pairs.map(p=>[p.prediction,p.actor]));
      overlay[name].hits+=pairs.filter(p=>close(p.actor)).length;
      overlay[name].unmatched+=shown.filter(p=>close(p)&&!matched.has(p)).length;
      if(name==='scenario')continue;
      for(const p of shown.filter(p=>p.hint)){
        associations.push({frame:number,source_id:p.sourceId,display_id:p.id,sector:sector(p.x,p.y),
          reference_role:matched.get(p)?.role??null,optical_camera:p.hint.camera,
          last_pair_age_s:+(sim_time-p.hint.last_s).toFixed(2)});
      }
      if(!isHigh)continue;
      for(const [key,ids] of Object.entries(sourceCases)){
        const row=cases[key]??={source_ids:ids,native_measurements:0,symbol_ticks:{},matched:0,frames:[]};
        row.native_measurements+=tracker.filter(p=>ids.includes(p.id)&&p.status==='measured'&&close(p)).length;
        for(const p of shown.filter(p=>ids.includes(p.sourceId)&&close(p))){
          add(row.symbol_ticks,p.label+'/'+p.status);
          if(matched.has(p))row.matched++;
          row.frames.push([number,p.sourceId,p.id,p.label,p.status]);
        }
      }
    }
  }});
  const metrics=await motion(directory,options);
  const groups={};
  for(const [key,g] of Object.entries(metrics.groups)){
    if(!g.display.matched)continue;
    const d=g.display;
    groups[key]={within_35m_frames:g.within_35m_frames,matched:d.matched,distinct_ids:d.distinct_ids,
      duplicate_frames:d.duplicate_frames,footprint_duplicates:d.footprint_duplicate_frames,
      first_spatial_track:g.first_spatial_track,first_draw:g.first_draw,first_half_opacity:g.first_half_opacity,
      position_error_m:d.position_error_m,
      same_id_translation_error_m:d.same_id.translation_m,same_id_extent_error_m:d.same_id.extent_m};
  }
  const core=s=>({overall:s.overall,sector:s.sector,by_status:s.by_status,position_error:s.matched_position_residual,
    direct_id_switches:s.direct_id_switches,reacquisitions:s.reacquisitions});
  return {input_sha256:coverage.sha256,frames:coverage.frames,first:coverage.first_frame,last:coverage.last_frame,
    town_sha256:coverage.town_sha256,parked_sha256:coverage.with_static_road_users.source_sha256,
    memory_sha256_lf:coverage.memory_sha256_lf,tracker:coverage.tracker.overall,
    spatial:core(coverage.display),spatial_with_parked:core(coverage.with_static_road_users.display),overlay,
    associations,source_cases:cases,all_wheeled_steps:metrics.all_nearby_wheeled_same_id_steps.display,encounters:groups};
}

const oldBytes=archive('web/rider-association.mjs');
const oldRider=await import('data:text/javascript;base64,'+oldBytes.toString('base64'));
const temp=fs.mkdtempSync(path.join(os.tmpdir(),'ado-object-replay-'));
const report={release,association_sha256_lf:{before:hashSource(oldBytes),
  after:hashSource(fs.readFileSync(path.join(root,'web/rider-association.mjs')))},scoring_sha256_lf:{},replays:{}};
for(const file of ['tools/audit_city_perception.mjs','tools/audit_city_motion.mjs','web/road-users.js',
  'tools/compare_rider_object_replays.mjs'])report.scoring_sha256_lf[file]=hashSource(fs.readFileSync(path.join(root,file)));
try{
  const staticFile=path.join(temp,'parked.json');
  fs.writeFileSync(staticFile,archive('results/perception/town03-static-vehicles.json'));
  for(const name of ['surround-city','overtake-clear','overtake-yield']){
    const directory=path.join(temp,name);
    fs.mkdirSync(directory);
    for(const file of ['replay-pack.json.gz','town.json'])
      fs.writeFileSync(path.join(directory,file),archive('web/media/'+name+'/'+file));
    const before=await score(directory,staticFile,release,oldRider,name==='surround-city');
    const after=await score(directory,staticFile,null,currentRider,name==='surround-city');
    if(before.input_sha256!==after.input_sha256||before.frames!==after.frames)throw Error('Inputs differed');
    report.replays[name]={before,after};
  }
  fs.writeFileSync(output,JSON.stringify(report,null,2)+'\n');
}finally{
  fs.rmSync(temp,{recursive:true,force:true});
}
