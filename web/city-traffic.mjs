// The chart samples measured frames. The density scores use the complete trace.
export async function loadTrafficStudy() {
  const response = await fetch('./media/city-traffic.json');
  if (!response.ok) return;
  const {drives} = await response.json();
  if (!drives?.length) return;
  const byId = id => document.getElementById(id);
  const svg = byId('traffic-plot');
  const selector = byId('traffic-drive');
  const scrub = byId('traffic-time');
  const stamp = t => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, '0')}`;
  for (const drive of drives) selector.add(new Option(drive.label, drive.trial));
  const left = 30, right = 689, top = 10, bottom = 125;
  const make = (tag, attrs) => {
    const item = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (const [k, v] of Object.entries(attrs)) item.setAttribute(k, v);
    svg.append(item);
    return item;
  };
  let run, end, marker;
  function showTick() {
    const i = Number(scrub.value);
    const [t, frame, progress, speed, target, limit, cars, walkers] = run.samples[i];
    byId('traffic-timestamp').textContent = stamp(t);
    const parts = [`${Math.round(speed)} km/h actual`];
    if (target !== null) parts.push(`${Math.round(target)} km/h requested`);
    if (limit !== null) parts.push(`${limit} km/h limit`);
    if (cars !== null) parts.push(`${cars} moving vehicle${cars === 1 ? '' : 's'} · ${walkers} moving pedestrian${walkers === 1 ? '' : 's'} within 50 m`);
    byId('traffic-reading').replaceChildren(...parts.map(value => {
      const span = document.createElement('span'); span.textContent = value; return span;
    }));
    scrub.setAttribute('aria-valuetext', `${stamp(t)}, ${Math.round(progress)} metres, ${parts.join(', ')}`);
    const x = left + t / end * (right - left);
    marker.setAttribute('x1', x); marker.setAttribute('x2', x);
    byId('traffic-watch').href = `./?run=${encodeURIComponent(run.trial)}&at=${t}`;
    byId('traffic-frame').href = `./inspect.html?run=${encodeURIComponent(run.trial)}&frame=${frame}&camera=front`;
  }
  function draw() {
    run = drives[selector.selectedIndex];
    end = run.samples.at(-1)[0];
    const max = Math.max(10, 10 * Math.ceil(Math.max(...run.samples.map(s => Math.max(s[3], s[4] ?? 0))) / 10));
    svg.replaceChildren();
    const y = v => bottom - (bottom - top) * v / max;
    for (let value = 0; value <= max; value += max > 50 ? 20 : 10) {
      make('line', {x1:left, y1:y(value), x2:right, y2:y(value), class:'traffic-grid'});
      make('text', {x:left-6, y:y(value)+4, 'text-anchor':'end', class:'traffic-axis'}).textContent = value;
    }
    for (const [column, name] of [[4, 'traffic-request'], [3, 'traffic-speed']]) {
      let move = true, path = '';
      for (const row of run.samples) {
        if (row[column] === null) { move = true; continue; }
        const x = left + row[0] / end * (right - left);
        path += `${move ? 'M' : 'L'}${x.toFixed(2)} ${y(row[column]).toFixed(2)} `;
        move = false;
      }
      make('path', {d:path, class:name});
    }
    marker = make('line', {y1:top, y2:bottom, class:'traffic-marker'});
    scrub.max = run.samples.length - 1;
    scrub.value = 0;
    byId('traffic-cones').hidden = run.cones_at_s == null;
    const d = run.nearby;
    byId('traffic-summary').textContent = `A moving vehicle and a moving pedestrian were both within 50 m for ${(d.moving_car_and_ped_frames * .1).toFixed(1)} of ${(d.eligible_frames * .1).toFixed(1)} scored seconds. These counts come from physical actor positions and include people or vehicles hidden from the sensors.`;
    showTick();
  }
  function seek(t) {
    let closest = 0;
    run.samples.forEach((s, i) => { if (Math.abs(s[0] - t) < Math.abs(run.samples[closest][0] - t)) closest = i; });
    scrub.value = closest;
    showTick();
  }
  svg.addEventListener('click', e => {
    const point = svg.createSVGPoint();
    point.x = e.clientX; point.y = e.clientY;
    const x = point.matrixTransform(svg.getScreenCTM().inverse()).x;
    seek((x - left) / (right - left) * end);
  });
  byId('traffic-cones').addEventListener('click', () => seek(run.cones_at_s));
  scrub.addEventListener('input', showTick);
  selector.addEventListener('change', draw);
  draw();
  byId('traffic-study').hidden = false;
}
