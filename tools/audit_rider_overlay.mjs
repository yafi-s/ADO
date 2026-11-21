// Score the rider overlay independently of the spatial tracker's original classes.
import fs from 'node:fs';
import {parseArgs} from 'node:util';
import {audit,match,sector} from './audit_city_perception.mjs';
import {bicycleAssociations,RIDER_HISTORY_FRAMES} from '../web/rider-association.mjs';
const {values,positionals}=parseArgs({allowPositionals:true,options:{'static-map':{type:'string'},town:{type:'string'},
  'memory-file':{type:'string'},'memory-revision':{type:'string'},output:{type:'string'}}});
if(positionals.length!==1||!values.output)throw new Error('Pass one recording and an output JSON file');
const history=[],associations=[],sectors={},unknown={predictions:0,position_matched:0},targets={};
const close=v=>Math.hypot(v.x,v.y)<=35;
const stats=await audit(positionals[0],{staticFile:values['static-map'],townFile:values.town,memoryFile:values['memory-file'],
  memoryRevision:values['memory-revision'],onFrame({frame,actors,tracker,display}){
    history.push(frame);if(history.length>RIDER_HISTORY_FRAMES)history.shift();
    const hints=bicycleAssociations(history);
    const shown=display.map(p=>p.label==='person'&&hints.has(p.sourceId)?{...p,label:'bicycle',hint:hints.get(p.sourceId)}:p);
    const base=match(actors,display),overlay=match(actors,shown);
    const assigned=new Map(overlay.map(p=>[p.prediction,p.actor]));
    for(const p of shown.filter(p=>p.hint))
      associations.push({frame:frame.frame,time_s:frame.sim_time,source_id:p.sourceId,display_id:p.id,
        reference_actor:assigned.get(p)?.id??null,reference_role:assigned.get(p)?.role??null,camera:p.hint.camera,range_age_s:p.hint.range_age_s});
    for(const a of actors.filter(close)){
      const s=sectors[sector(a.x,a.y)]??={core:0,overlay:0,actors:0};s.actors++;
      s.core+=Number(base.some(p=>p.actor===a));s.overlay+=Number(overlay.some(p=>p.actor===a));
      if(a.type==='cyclist'){
        const c=targets[a.id]??={frames:0,tracker_person_near:0,core_person_near:0,raw_bicycle_near:0,overlay_bicycle:0};
        c.frames++;
        const near=(arr,label)=>Number(arr.some(p=>p.label===label&&Math.hypot(p.x-a.x,p.y-a.y)<=3));
        c.tracker_person_near+=near(tracker,'person');c.core_person_near+=near(display,'person');
        c.raw_bicycle_near+=near(tracker,'bicycle');c.overlay_bicycle+=Number(overlay.some(p=>p.actor===a&&p.prediction.label==='bicycle'));
      }
    }
    unknown.predictions+=display.filter(p=>p.label==='unknown'&&close(p)).length;
    unknown.position_matched+=match(actors,display,3,false).filter(p=>p.prediction.label==='unknown'&&close(p.prediction)).length;
  }});
fs.writeFileSync(values.output,JSON.stringify({source_sha256:stats.sha256,memory_sha256_lf:stats.memory_sha256_lf,
  frames:stats.frames,unknown,targets,sector:sectors,associations},null,2)+'\n');
