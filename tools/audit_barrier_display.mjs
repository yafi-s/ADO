// Score the rider's camera overlays and chain symbols against a reviewed native
// journal, including proposals kept for inspection after their label is rejected.
import fs from 'node:fs';
import {createInterface} from 'node:readline';
import {createHash} from 'node:crypto';
import {barrierReviewer} from '../web/barrier-review.mjs';
import {advance,createState,contextOf} from '../web/object-memory.mjs';
import {cameraBoxes,bearingObservations} from '../web/lamp-observations.mjs';

const [journal,sidecar,out]=process.argv.slice(2);
if(!out)throw Error('Pass perception.jsonl, barrier-review.json and output.json');
const bytes=fs.readFileSync(sidecar),review=JSON.parse(bytes),qualify=barrierReviewer(review);
const chain=t=>t?.label==='barrier'&&t.barrier_subtype==='chain';
const result={first_frame:null,last_frame:null,frames:0,partial:review.partial,
  review_sha256:createHash('sha256').update(bytes).digest('hex'),raw_camera_proposals:0,qualified_camera_proposals:0,
  raw_measured:0,qualified_measured:0,raw_coasts:0,qualified_coasts:0,untracked_boxes_drawn:0,
  raw_display_symbol_ticks:0,qualified_display_symbol_ticks:0,qualified_observations:[],qualified_display_runs:[]};
const rawState=createState(),qualifiedState=createState();
let before=null,after=null,lastRun=null;
const stream=fs.createReadStream(journal);
const lines=createInterface({input:stream,crlfDelay:Infinity});
try{
  for await(const line of lines){
    const native=JSON.parse(line);
    if(native.frame!==(result.last_frame??review.first_frame-1)+1)throw Error('Review and journal frames differ');
    const {frame,sim_time,perception_version,tracks,run_id,seed,scenario,camera_detections,detections,lidar,radar,corner_radar}=native;
    const {ego,ego_z}=native.observer_map;
    const raw={frame,sim_time,perception_version,run_id,seed,scenario,tracks,camera_detections,detections,lidar,radar,corner_radar,
      observer_map:{ego,ego_z}};
    const qualified=qualify(raw),views=['front','left','right','rear_left','rear_right'];
    const proposals=Object.values(raw.camera_detections??{}).flat().filter(chain);
    const depth=raw.tracks.filter(chain),kept=qualified.tracks.filter(chain);
    if(proposals.length!==(qualified.barrier_review?.cameras?.filter(c=>chain(c.box)).length??0))throw Error('Lost raw camera audit evidence');
    if(depth.length!==(qualified.barrier_review?.tracks?.filter(chain).length??0))throw Error('Lost raw range audit evidence');
    result.raw_camera_proposals+=proposals.length;
    result.raw_measured+=depth.filter(t=>!t.coasting).length;result.raw_coasts+=depth.filter(t=>t.coasting).length;
    result.qualified_measured+=kept.filter(t=>!t.coasting).length;result.qualified_coasts+=kept.filter(t=>t.coasting).length;
    for(const view of views){
      for(const box of cameraBoxes(qualified,view).filter(chain)){
        result.qualified_camera_proposals++;
        if(!box.range_supported||!kept.some(t=>t.id===box.track_id))throw Error('Rider camera label lacks reviewed depth');
        result.qualified_observations.push({frame,view,source_id:box.track_id,image_id:box.image_id});
      }
    }
    const chainKeys=new Set();
    for(const source of [raw,before?.frame]){
      for(const [view,boxes] of Object.entries(source?.camera_detections??{})){
        boxes.forEach((box,index)=>{
          if(chain(box))chainKeys.add(JSON.stringify(box.image_id==null?[view,source.frame,index]:[view,box.image_id]));
        });
      }
    }
    result.untracked_boxes_drawn+=bearingObservations(qualified,after?.frame,views).filter(b=>chainKeys.has(b.key)).length;
    before=contextOf(raw,before);after=contextOf(qualified,after);
    result.raw_display_symbol_ticks+=advance(rawState,before).filter(o=>o.barrierSubtype==='chain').length;
    const symbols=advance(qualifiedState,after).filter(o=>o.barrierSubtype==='chain');
    result.qualified_display_symbol_ticks+=symbols.length;
    for(const s of symbols){
      if(lastRun?.source_id===s.sourceId&&lastRun.last_frame===frame-1)lastRun.last_frame=frame;
      else {lastRun={source_id:s.sourceId,first_frame:frame,last_frame:frame};result.qualified_display_runs.push(lastRun);}
    }
    result.first_frame??=frame;result.last_frame=frame;result.frames++;
    if(frame===review.last_frame)break;
  }
}finally{lines.close();stream.destroy();}
if(result.last_frame!==review.last_frame||result.raw_camera_proposals!==(review.counts_by_subtype?.chain?.raw??review.source_camera_observations))
  throw Error('Review is incomplete for this journal');
fs.writeFileSync(out,JSON.stringify(result,null,2)+'\n');
