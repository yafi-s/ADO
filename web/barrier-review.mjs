// Post appearance is reviewed on the exact native camera exposure used for
// each two-post proposal. The raw detector and lidar observations stay
// available in the frame inspector and the untouched recording.
const subtypes=['chain','striped_board'];

export function barrierName(item){
  if(item?.label!=='barrier')return null;
  const subtype=item.barrier_subtype??item.barrierSubtype;
  return subtype==='chain'?'chain barrier':subtype==='striped_board'?'striped barrier':null;
}

function sameObservation(observation,view,box){
  if(observation.view!==view || (observation.track_id??null)!==(box.track_id??null))return false;
  const subtype=observation.barrier_subtype??observation.subtype;
  if(subtype!=null&&subtype!==box.barrier_subtype)return false;
  if(observation.image_id!=null && box.image_id!=null && observation.image_id!==box.image_id)return false;
  return Array.isArray(observation.box) && observation.box.length===4 && Array.isArray(box.box) &&
    observation.box.every((v,i)=>Math.abs(v-box.box[i])<.00001);
}

export function barrierReviewer(review=null){
  if(review && (review.schema!=='ado-barrier-post-review-v1'||!review.frames||!Number.isInteger(review.first_frame)||
     !Number.isInteger(review.last_frame)||review.first_frame>review.last_frame))throw Error('Invalid barrier review.');
  const approved=new Map();
  const enabled=review?(review.reviewed_subtypes??['chain']):subtypes;
  const reviewedClass=item=>item?.label==='barrier'&&enabled.includes(item.barrier_subtype);
  let previous=null,lastResult=null;
  return frame=>{
    const rows=review ? review.frames[frame.frame]??{} : frame.barrier_review;
    if(!rows){approved.clear();previous=null;return frame;}
    if(previous?.frame===frame.frame&&previous.sim_time===frame.sim_time&&previous.run_id===frame.run_id&&
       previous.seed===frame.seed&&previous.scenario===frame.scenario)return lastResult;
    if(!previous || previous.frame+1!==frame.frame || previous.run_id!==frame.run_id ||
       previous.seed!==frame.seed || previous.scenario!==frame.scenario || frame.sim_time<=previous.sim_time)approved.clear();
    previous=frame;
    const rawTracks=(frame.tracks??[]).filter(reviewedClass);
    const present=new Map(rawTracks.map(t=>[t.id,t]));
    for(const id of approved.keys())if(!present.has(id))approved.delete(id);
    if(!rawTracks.length&&!Object.keys(rows).length&&
       !Object.values(frame.camera_detections??{}).some(boxes=>boxes.some(reviewedClass))&&
       !frame.detections?.some(reviewedClass))return lastResult=frame;
    const cameras=[], revoked=[], observations=Object.values(rows).flatMap(row=>row.observations??[]);
    const qualifyBox=(view,box)=>{
      if(!reviewedClass(box))return true;
      const matches=observations.filter(o=>sameObservation(o,view,box));
      const confirmed=matches.length===1 && matches[0].confirmed===true &&
        box.range_supported===true && present.get(box.track_id)?.barrier_subtype===box.barrier_subtype &&
        rows[box.track_id]?.confirmed===true;
      cameras.push({view,box,confirmed,observation:matches.length===1?matches[0]:null});
      return confirmed;
    };
    const camera_detections=Object.fromEntries(Object.entries(frame.camera_detections??{}).map(([view,boxes])=>
      [view,boxes.filter(box=>qualifyBox(view,box))]));
    const tracks=(frame.tracks??[]).flatMap(track=>{
      if(!reviewedClass(track))return [track];
      if(track.coasting){
        const prior=approved.get(track.id);
        if(prior?.subtype===track.barrier_subtype)return [{...track,last_cameras:prior.views}];
        revoked.push(track.id);
        return [];
      }
      const decision=rows[track.id],prior=approved.get(track.id);
      if(decision?.range_only===true && decision.confirmed===true && track.barrier_subtype==='striped_board' &&
         prior?.subtype===track.barrier_subtype && !track.cameras?.length && track.sources?.includes('lidar')){
        return [{...track,cameras:[],rgb_boxes:{},last_cameras:prior.views}];
      }
      const views=(track.cameras??[]).filter(view=>cameras.some(c=>c.confirmed &&
        c.view===view && c.box.track_id===track.id && c.box.barrier_subtype===track.barrier_subtype && c.box.range_supported===true));
      if(decision?.confirmed!==true || !views.length){
        approved.delete(track.id);
        revoked.push(track.id);
        return [];
      }
      approved.set(track.id,{subtype:track.barrier_subtype,views});
      const rgb_boxes=Object.fromEntries(Object.entries(track.rgb_boxes??{}).filter(([view])=>views.includes(view)));
      return [{...track,cameras:views,last_cameras:views,rgb_boxes}];
    });
    // Some archives repeat the front camera boxes in the legacy field.
    const detections=frame.detections?.filter(box=>!reviewedClass(box)||
      cameras.some(c=>c.view==='front' && c.confirmed && sameObservation(c.observation,'front',box)));
    return lastResult={...frame,tracks,camera_detections,detections,
      barrier_review:{tracks:rawTracks,cameras,decisions:rows,revoked}};
  };
}

export function applyBarrierReview(replay,review){
  const frames=Array.isArray(replay)?replay:replay?.frames;
  if(!review&&!frames?.some(f=>f?.barrier_review))return replay;
  if(!Array.isArray(frames)||!frames.length||review&&(review.first_frame!==frames[0].frame||review.last_frame!==frames.at(-1).frame))
    throw Error('Barrier review does not match this recording.');
  const qualified=frames.map(barrierReviewer(review));
  return Array.isArray(replay)?qualified:{...replay,frames:qualified};
}

export async function loadBarrierReview(entry,signal){
  const url=entry?.barrier_review_url,expected=entry?.barrier_review_sha256;
  if(!url&&!expected)return null;
  if(typeof url!=='string'||!/^[a-f0-9]{64}$/i.test(expected??''))throw Error('Incomplete barrier review reference.');
  const address=new URL(url,location.href);
  if(address.origin!==location.origin||!['http:','https:'].includes(address.protocol))throw Error('Invalid barrier review URL.');
  const integrity='sha256-'+btoa(String.fromCharCode(...expected.match(/../g).map(pair=>parseInt(pair,16))));
  const response=await fetch(address,{signal,cache:'no-cache',redirect:'error',integrity});
  if(!response.ok)throw Error('Could not load the recorded barrier review.');
  const bytes=await response.arrayBuffer();
  if(globalThis.crypto?.subtle){
    const hash=new Uint8Array(await crypto.subtle.digest('SHA-256',bytes));
    const actual=Array.from(hash,b=>b.toString(16).padStart(2,'0')).join('');
    if(actual!==expected.toLowerCase())throw Error('Barrier review checksum mismatch.');
  }
  return JSON.parse(new TextDecoder().decode(bytes));
}
