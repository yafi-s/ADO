// Final single-pass evaluation. Actor references enter after the native track
// and actual rider-display paths have produced their independent outputs.
import fs from 'node:fs';
import {createHash} from 'node:crypto';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import {audit,match,sector,newLayer,scoreLayer,layerSummary} from './audit_city_perception.mjs';
import {bicycleAssociations,RIDER_HISTORY_FRAMES} from '../web/rider-association.mjs';

const [directory,reviewFile,output,vruRowFile]=process.argv.slice(2);
if(!output)throw Error('Pass a native recording directory, full barrier-review.json and output JSON path');
const root=fileURLToPath(new URL('../',import.meta.url));
const hash=b=>createHash('sha256').update(b).digest('hex');
const staticFile=path.join(root,'results/perception/town03-static-vehicles.json');
const review=JSON.parse(fs.readFileSync(reviewFile));
if(review.partial!==false)throw Error('Final evaluation requires a complete post review.');
const shown={scenario:newLayer(),parked:newLayer()},history=[],roles={},bikeSources={},unmatched=new Map();
const evidence={board:null,cyclist:null,unmatched_current_examples:[]};
const labelCounts={tracker:{},rider:{}},barriers={raw_chain_proposals:0,raw_chain_ranged_boxes:0,reviewed_chain_boxes:0,
  all_reviewed_subtype_proposals:0,raw_board_proposals:0,reviewed_board_boxes:0,qualified_measured_board_tracks:0,
  raw_measured_chain_tracks:0,qualified_measured_chain_tracks:0,visible_chain_symbol_ticks:0,
  raw_measured_board_tracks:0,visible_board_symbol_ticks:0,visible_board_matched_correct_subtype:0};
const round=n=>+n.toFixed(3),close=p=>Math.hypot(p.x,p.y)<=35;
const inc=(o,k)=>{o[k]=(o[k]??0)+1;};
const modules=['web/object-memory.mjs','web/rider-association.mjs','web/barrier-review.mjs','web/perception-scene.js',
  'web/road-users.js','tools/audit_city_perception.mjs','tools/audit_urban_hero.mjs'];
const code=Object.fromEntries(modules.map(f=>[f,hash(fs.readFileSync(path.join(root,f),'utf8').replaceAll('\r\n','\n'))]));
const chain=p=>p.label==='barrier'&&p.barrier_subtype==='chain';
const vruFd=vruRowFile?fs.openSync(vruRowFile,'w'):null;
const raw=await audit(directory,{staticFile,barrierReviewFile:reviewFile,onFrame({frame,riderFrame,actors,tracker,display}){
  history.push(riderFrame);if(history.length>RIDER_HISTORY_FRAMES)history.shift();
  const hints=bicycleAssociations(history),native=new Map(tracker.map(t=>[t.id,t]));
  const visible=display.map(p=>{
    const source=native.get(p.sourceId), hint=p.label==='person'?hints.get(p.sourceId):null;
    return {...p,label:hint?.label??p.label,classification:hint?'optical association':p.mountedRider?
      (source?.cycle_association?'native mounted-cycle association':'remembered native mounted association'):null,
      cameras:source&&!source.coasting?source.cameras:[],sources:source&&!source.coasting?source.sources:[]};
  });
  const scenario=actors.filter(a=>a.role!=='parked-map-vehicle');
  scoreLayer(shown.scenario,scenario,visible,frame.frame);
  scoreLayer(shown.parked,actors,visible,frame.frame);
  const predictions={tracker,rider:visible};
  const writeVRU=vruFd!==null && (actors.some(a=>a.type==='cyclist'&&Math.hypot(a.x,a.y)<=60) ||
    [...tracker,...visible].some(p=>p.label==='bicycle'&&Math.hypot(p.x,p.y)<=60));
  const refs=writeVRU?actors.filter(a=>['cyclist','pedestrian'].includes(a.type)&&Math.hypot(a.x,a.y)<=60)
    .map(({id,role,type,x,y,sector})=>({id,role,type,x,y,sector})):[];
  const vru={frame:frame.frame,sim_time:frame.sim_time,ego:frame.observer_map.ego,reference:refs};
  for(const [layer,preds] of Object.entries(predictions)){
    const pairs=match(actors,preds), assignment=new Map(pairs.map(p=>[p.actor,p]));
    if(writeVRU){
      const name=layer==='rider'?'display':'tracker';
      const slim=p=>({id:p.id,sourceId:p.sourceId,label:p.label,x:p.x,y:p.y,status:p.status,classification:p.classification});
      vru[name]=preds.filter(p=>Math.hypot(p.x,p.y)<=60).map(slim);
      for(const ref of refs){
        const pair=pairs.find(pair=>pair.actor.id===ref.id);
        ref[name+'_match']=pair?{...slim(pair.prediction),error_m:round(pair.distance)}:null;
      }
    }
    for(const a of actors.filter(close)){
      const row=roles[a.role]??={};
      const counts=row[layer]??={actor_frames:0,family_matches:0,exact_label_matches:0,predicted_labels:{}};
      counts.actor_frames++;
      const pair=assignment.get(a);if(!pair)continue;
      counts.family_matches++;inc(counts.predicted_labels,pair.prediction.label);
      if(pair.prediction.label===a.label)counts.exact_label_matches++;
      if(a.role==='cyclist'&&pair.prediction.label==='bicycle'){
        const p=pair.prediction,source=layer==='tracker'?p:native.get(p.sourceId);
        const method=source?.cycle_association?.range_source??p.classification??'native bicycle class or its spatial memory';
        inc(bikeSources,layer+' / '+method);
      }
    }
    const matched=new Set(pairs.map(p=>p.prediction));
    for(const p of preds.filter(close)){
      const label=labelCounts[layer][p.label]??={predictions:0,unmatched:0};
      label.predictions++;if(!matched.has(p))label.unmatched++;
    }
    if(layer!=='rider')continue;
    for(const p of preds.filter(close)){
      if(matched.has(p)||p.label==='unknown')continue;
      const entry=unmatched.get(p.id)??{display_id:p.id,label:p.label,unmatched_ticks:0,current_ticks:0,example:null};
      entry.unmatched_ticks++;if(p.status==='current')entry.current_ticks++;
      const s=native.get(p.sourceId);
      const observations=Object.entries(frame.camera_detections??{}).flatMap(([camera,boxes])=>boxes.filter(b=>b.track_id===s?.id&&b.range_supported===true).map(b=>({camera,...b})));
      if(observations.length&&s&&!s.coasting){
        const nearest=actors.reduce((best,a)=>{
          const d=Math.hypot(a.x-p.x,a.y-p.y);
          return !best||d<best.distance_m?{source_id:a.id,role:a.role,distance_m:round(d)}:best;
        },null);
        const obs=observations.sort((a,b)=>b.score-a.score)[0],quality=(nearest?.distance_m??35)*obs.score;
        if(quality>(entry.example?.quality??0))entry.example={frame:frame.frame,sim_time:frame.sim_time,source_id:s.id,
          camera:obs.camera,box:obs.box,camera_class:obs.label,score:round(obs.score),nearest,quality:round(quality),
          display_xy:[round(p.x),round(p.y)],range_m:round(Math.hypot(p.x,p.y))};
      }
      unmatched.set(p.id,entry);
    }
    for(const {actor:a,prediction:p,distance} of pairs){
      if(!close(a))continue;
      const board=a.blueprint==='static.prop.streetbarrier'&&p.barrierSubtype==='striped_board';
      const cyclist=a.role==='cyclist'&&p.label==='bicycle';
      if(board)barriers.visible_board_matched_correct_subtype++;
      if(!board&&!cyclist)continue;
      const s=native.get(p.sourceId);if(!s||s.coasting)continue;
      const key=board?'board':'cyclist';
      for(const [view,boxes] of Object.entries(riderFrame.camera_detections??{})){
        for(const b of boxes.filter(b=>b.track_id===s.id&&b.range_supported&&(board?b.barrier_subtype==='striped_board':['person','bicycle'].includes(b.label)))){
          const range=Math.hypot(p.x,p.y),area=(b.box[2]-b.box[0])*(b.box[3]-b.box[1]);
          // Prefer a resolved front view where the road user is visible on the follow camera.
          const quality=area*b.score*(view==='front'?5:1)*(range>=5&&range<=18?2:1);
          if(quality>(evidence[key]?.quality??-1))evidence[key]={frame:frame.frame,sim_time:frame.sim_time,role:a.role,
            actor_id:a.id,display_id:p.id,source_id:s.id,display_label:p.label,subtype:p.barrierSubtype,
            reference_residual_m:round(distance),camera:view,box:b.box,camera_label:b.label,score:round(b.score),range_m:round(range),quality};
        }
      }
    }
  }
  if(writeVRU)fs.writeSync(vruFd,JSON.stringify(vru)+'\n');
  const rawBoxes=Object.values(frame.camera_detections??{}).flat(), qualifiedBoxes=Object.values(riderFrame.camera_detections??{}).flat();
  const det=rawBoxes.filter(chain);
  barriers.raw_chain_proposals+=det.length;barriers.raw_chain_ranged_boxes+=det.filter(b=>b.range_supported).length;
  barriers.reviewed_chain_boxes+=qualifiedBoxes.filter(chain).length;
  barriers.all_reviewed_subtype_proposals+=rawBoxes.filter(b=>(review.reviewed_subtypes??['chain']).includes(b.barrier_subtype)).length;
  barriers.raw_board_proposals+=rawBoxes.filter(b=>b.barrier_subtype==='striped_board').length;
  barriers.reviewed_board_boxes+=qualifiedBoxes.filter(b=>b.barrier_subtype==='striped_board').length;
  barriers.qualified_measured_board_tracks+=riderFrame.tracks.filter(t=>t.barrier_subtype==='striped_board'&&!t.coasting).length;
  barriers.raw_measured_chain_tracks+=tracker.filter(t=>chain(t)&&!t.coasting).length;
  barriers.qualified_measured_chain_tracks+=riderFrame.tracks.filter(t=>chain(t)&&!t.coasting).length;
  barriers.raw_measured_board_tracks+=tracker.filter(t=>t.barrier_subtype==='striped_board'&&!t.coasting).length;
  barriers.visible_chain_symbol_ticks+=visible.filter(t=>t.barrierSubtype==='chain').length;
  barriers.visible_board_symbol_ticks+=visible.filter(t=>t.barrierSubtype==='striped_board').length;
}});
if(vruFd!==null)fs.closeSync(vruFd);
if(raw.partial_tail_ignored||raw.frames!==review.capture_frames||barriers.all_reviewed_subtype_proposals!==review.source_camera_observations)
  throw Error('Native source or review is incomplete.');
if(raw.input==='perception.jsonl'&&review.native_journal_sha256&&raw.sha256!==review.native_journal_sha256)
  throw Error('Post review was built from a different native journal.');
evidence.unmatched_current_examples=[...unmatched.values()].filter(x=>x.example&&x.example.nearest?.distance_m>5)
  .sort((a,b)=>b.unmatched_ticks-a.unmatched_ticks).slice(0,6);
const result={schema:'ado-urban-hero-perception-v1',source:{file:raw.input,sha256:raw.sha256,frames:raw.frames,
  first_frame:raw.first_frame,last_frame:raw.last_frame,town_sha256:raw.town_sha256,barrier_review_sha256:raw.barrier_review_sha256,
  parked_sha256:raw.with_static_road_users.source_sha256,code_sha256_lf:code},
  radius_m:raw.radius_m,match_gate_m:raw.match_gate_m,scenario:{raw_tracker:raw.tracker,core:raw.display,rider:layerSummary(shown.scenario)},
  including_registered_parked:{raw_tracker:raw.with_static_road_users.tracker,core:raw.with_static_road_users.display,
    rider:layerSummary(shown.parked)},roles_including_parked:roles,unmatched_by_label_including_parked:labelCounts,
  correct_cycle_sources:bikeSources,barriers,evidence};
fs.writeFileSync(output,JSON.stringify(result,null,2)+'\n');
