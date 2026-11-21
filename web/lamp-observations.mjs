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

// Normalized image bounds retain registration when the selected feed is resized or letterboxed.
export function cameraBoxes(frame, view) {
  const detections = frame?.camera_detections ? frame.camera_detections[view] : view === 'front' ? frame?.detections : null;
  const boxes = Array.isArray(detections) ? [...detections] : [];
  for (const head of measuredHeads(frame)) {
    const color = cameraLamp(head);
    if (head.view !== view || color === 'unknown' || !Array.isArray(head.pixel_size) ||
        !head.pixel_size.every(v => Number.isFinite(v) && v>0) || head.pixel_size.length !== 2 ||
        !Array.isArray(head.pixel_box) || head.pixel_box.length !== 4) continue;
    const [w,h] = head.pixel_size;
    const [x0,y0,x1,y1] = head.pixel_box;
    boxes.push({label:`${color} light`,lamp:color,box:[x0/w,y0/h,x1/w,y1/h]});
  }
  return boxes.filter(item => Array.isArray(item?.box) && item.box.length===4 && item.box.every(Number.isFinite) &&
    item.box[2]>item.box[0] && item.box[3]>item.box[1]);
}
