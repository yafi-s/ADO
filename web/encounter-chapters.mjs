const labels = {
  pedestrian: 'Pedestrian crossing',
  'pedestrian-delayed': 'Delayed crossing',
  occluded: 'Occluded crossing',
  lead: 'Stopped lead vehicle',
  braking: 'Braking traffic',
  left: 'Unprotected left turn',
  signal: 'Traffic signal',
  cyclist: 'Cyclist',
  roadworks: 'Roadworks',
  bus: 'Stopped bus'
};

function exposure(event) {
  if (!event || !Object.hasOwn(labels, event.kind)) return null;
  const measured = stamp => Number.isFinite(stamp) && stamp >= 0;
  const {kind} = event;
  let label = labels[kind];
  let field = 'exposed_s';
  if (kind === 'signal' && measured(event.queue_exposed_s)) {
    label = 'Signal queue';
    field = 'queue_exposed_s';
  } else if (kind === 'occluded' && measured(event.emerged_s)) {
    field = 'emerged_s';
  } else if (event.triggered === false || event.exposed === false || !measured(event.exposed_s)) {
    return null;
  }
  if (kind === 'bus' && event.passed && !event.violation) label = 'Bus pass';
  return {kind, label, field, stamp: event[field]};
}

export function recordedChapters(frames, manifest = null) {
  const selected = new Map(), firstFrame = new Map();
  for (const [index, frame] of frames.entries()) {
    if (!Array.isArray(frame.events)) continue;
    for (const event of frame.events) {
      const chapter = exposure(event);
      if (!chapter) continue;
      const previous = selected.get(chapter.kind);
      if (!previous || previous.field === 'exposed_s' || chapter.field !== 'exposed_s') selected.set(chapter.kind, chapter);
      const id = `${chapter.kind}/${chapter.field}`;
      if (!firstFrame.has(id)) firstFrame.set(id, index);
    }
  }

  const manifestEvents = manifest?.events ?? manifest?.summary?.events;
  if (Array.isArray(manifestEvents)) {
    for (const event of manifestEvents) {
      const chapter = exposure(event);
      if (chapter && !selected.has(chapter.kind)) selected.set(chapter.kind, chapter);
    }
  }

  const chapters = [];
  for (const {kind, label, field, stamp} of selected.values()) {
    let index = firstFrame.get(`${kind}/${field}`);
    if (index == null) {
      // A manifest uses simulation time. Resolve it against telemetry before seeking on the video clock.
      if (stamp < frames[0]?.sim_time) continue;
      index = frames.findIndex(frame => Number.isFinite(frame.sim_time) && frame.sim_time + .001 >= stamp);
    }
    if (index >= 0) chapters.push({kind, label, index});
  }
  return chapters.sort((a, b) => a.index - b.index);
}
