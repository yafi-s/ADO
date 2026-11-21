// Per-head lamp estimates, separate from map landmark IDs and scenario events.
// CARLA uses world X/Y for ground position and Z for elevation.
export function measuredHeads(frame) {
  if (!Array.isArray(frame?.signals)) return [];
  return frame.signals.filter(head => head && head.id != null &&
    [head.x, head.y, head.z].every(Number.isFinite));
}

export function cameraLamp(head) {
  if (head?.source !== 'camera') return 'unknown';
  const name = String(head.state).toLowerCase();
  if (name === 'yellow' || name === 'amber') return 'yellow';
  return name === 'green' || name === 'red' ? name : 'unknown';
}

export const LAMP_COLOURS = {red: '#ff5d50', yellow: '#ffc148', green: '#38e48a', unknown: '#8195a5'};

export function inspectionCamera(track, available, current) {
  if (track?.id == null) return null;
  const live = Array.isArray(track.cameras) ? track.cameras.filter(name=>available.includes(name)) : [];
  const previous = Array.isArray(track.last_cameras) ? track.last_cameras.filter(name=>available.includes(name)) : [];
  const views=live.length?live:previous;
  return views.includes(current)?current:views[0]??null;
}

export const MAX_CAMERA_HOLD_S = .15;

// A small optical target can be just a few pixels wide in the rider's inset.
// Label one registered bearing there until range is measured, without creating a world position.
export function frontBearingCue(items) {
  const eligible=items.filter(item=>item.view==='front' && item.key && validBox(item)
    && ['person','car','bus','truck'].includes(item.label)
    && item.box[2]-item.box[0]<=.12 && item.box[3]-item.box[1]<=.12);
  const centrality=item=>Math.abs((item.box[0]+item.box[2])/2-.5);
  return eligible.sort((a,b)=>(b.label==='person')-(a.label==='person') || centrality(a)-centrality(b))[0]??null;
}

function viewDetections(frame, view) {
  return frame?.camera_detections ? frame.camera_detections[view] : view === 'front' ? frame?.detections : null;
}

function validBox(item) {
  return Array.isArray(item?.box) && item.box.length===4 && item.box.every(Number.isFinite) &&
    item.box[2]>item.box[0] && item.box[3]>item.box[1];
}

function isSkippedView(frame, previous, view) {
  if (!frame.camera_detections || !previous || previous.run_id !== frame.run_id || previous.frame + 1 !== frame.frame ||
      previous.camera_inference_age_s?.[view] !== 0 ||
      !Array.isArray(previous.camera_detections?.[view]) || Object.hasOwn(frame.camera_detections, view)) return false;
  const age = frame.camera_inference_age_s?.[view];
  const elapsed = frame.sim_time - previous.sim_time;
  return Number.isFinite(age) && age > 0 && age <= MAX_CAMERA_HOLD_S &&
    Number.isFinite(frame.sim_time) && Number.isFinite(previous.sim_time) && elapsed > 0 && Math.abs(elapsed - age) < .001;
}

export function bearingObservations(frame, previous, views) {
  if (!frame) return [];
  const cones = (frame.cone_boxes??[]).filter(b=>!b.range_supported && views.includes(b.view)).map((b,i)=>({
    key:JSON.stringify(['cone',b.view,frame.frame,i]),view:b.view,label:b.label,box:b.box,age_s:0
  }));
  return cones.concat(views.flatMap(view => {
    const held = isSkippedView(frame, previous, view);
    const measured = held ? previous : frame;
    const detections = viewDetections(measured, view);
    const age = measured.camera_inference_age_s?.[view] ?? (view === 'front' ? measured.detection_age_s : null);
    if (!Array.isArray(detections) || (age != null && age !== 0)) return [];
    return detections.flatMap((item, index) => {
      if (item?.range_supported !== false || item.track_id != null || !validBox(item)) return [];
      const imageId = item.image_id;
      const key = JSON.stringify(imageId == null ? [view, measured.frame, index] : [view, imageId]);
      return [{key, view, label:item.label, box:item.box, age_s:held ? frame.camera_inference_age_s[view] : 0}];
    });
  }));
}

// Normalized image bounds retain registration when the selected feed is resized or letterboxed.
export function cameraBoxes(frame, view) {
  const detections = viewDetections(frame, view);
  const boxes = Array.isArray(detections) ? [...detections] : [];
  boxes.push(...(frame?.cone_boxes??[]).filter(b=>b.view===view));
  // Track-local bounds are recorded by camera/lidar association on this frame.
  // A retained class or coast has no fresh camera rectangle.
  for (const track of frame?.tracks??[]) {
    const box=track?.rgb_boxes?.[view];
    if (track?.id==null || track.coasting || !track.cameras?.includes(view) || !Array.isArray(box) ||
        boxes.some(item=>item.track_id===track.id && item.range_supported!==false)) continue;
    boxes.push({label:track.label,box,track_id:track.id,range_supported:true});
  }
  for (const head of measuredHeads(frame)) {
    const color = cameraLamp(head);
    if (head.view !== view || color === 'unknown' || !Array.isArray(head.pixel_size) ||
        !head.pixel_size.every(v => Number.isFinite(v) && v>0) || head.pixel_size.length !== 2 ||
        !Array.isArray(head.pixel_box) || head.pixel_box.length !== 4) continue;
    const [w,h] = head.pixel_size;
    const [x0,y0,x1,y1] = head.pixel_box;
    boxes.push({label:`${color} light`,lamp:color,box:[x0/w,y0/h,x1/w,y1/h]});
  }
  return boxes.filter(validBox);
}
