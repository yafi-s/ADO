import {ReplayMemory, laneAxes} from './object-memory.mjs';
import {loadReplay} from './replay-pack.mjs';

const $ = id => document.getElementById(id);
const finite = Number.isFinite;
const rad = Math.PI / 180;
const query = new URLSearchParams(location.search);
const views = {chase:'Chase', front:'Front', left:'Left', right:'Right', rear_left:'Rear left', rear_right:'Rear right', infrared:'Infrared'};
let catalog = [], session, request, movie, playing = false, timer, frameStarted = 0, index = 0;

async function json(url, signal) {
  const response = await fetch(url, {signal});
  if (!response.ok) throw new Error(`Could not load ${url} (${response.status})`);
  return response.json();
}

function parseReplay(data) {
  const frames = Array.isArray(data) ? data : data?.frames;
  if (!frames?.length || !frames.every(f => finite(f?.frame))) throw new Error('This recording has no frame telemetry.');
  const stamped = frames.every(f => finite(f.video_time));
  const times = frames.map((f,i) => stamped ? f.video_time : data.fps > 0 ? i/data.fps : f.sim_time - frames[0].sim_time);
  if (times.some((t,i) => !finite(t) || t < 0 || (i && t <= times[i-1]))) throw new Error('Frame timestamps are missing or out of order.');
  return {frames, times, stamped, cameras:data.cameras};
}

const point = p => Array.isArray(p) && p.length >= 2 && p.slice(0,2).every(finite);
const ends = (a,b) => [a,b].map(p => p.slice(0,2).map(v => v.toFixed(2)).join(',')).join('|');

async function roadContext(entry, signal) {
  // Use the same road geometry and driven directions as the rider scene.
  const town = await json(entry.map_url ?? './media/town.json', signal).catch(() => null);
  const name = town?.producer?.map;
  if (!town || (entry.map && name !== entry.map)) return {deps:{}, road:''};
  const [features,bodies] = await Promise.all([
    typeof name === 'string' && /^[\w-]+$/.test(name)
      ? json(`./media/map-features/${name}.json`,signal).catch(() => null) : null,
    name === 'Town03_Opt'
      ? json('./media/map-features/Town03_Opt-bodies.json',signal).catch(() => null) : null
  ]);
  const lanes = (town.roads || []).map(l => (Array.isArray(l) ? l : l.points ?? []).filter(p => point(p) && finite(p[2]))).filter(l => l.length > 1);
  const directions = new Map((features?.map === name ? features.lanes ?? [] : []).map(l => [ends(l.a,l.b),l.forward]));
  const axis = laneAxes(lanes, lanes.map(l => directions.get(ends(l[0],l.at(-1)))));
  const road = lanes.map(l => 'M' + l.map(p => p.slice(0,2).join(',')).join('L')).join('');
  return {deps:{laneAxis:axis, laneDistance:axis.distance}, road, bodies:bodies?.map===name ? bodies.blueprints : null};
}

function stop() {
  clearTimeout(timer);
  const wasPlaying = playing;
  playing = false;
  $('play').textContent = 'Play';
  if (wasPlaying && session) link();
}

function discardMovie() {
  if (!movie) return;
  const old = movie;
  movie = null;
  old.removeAttribute('src');
  old.load();
  old.remove();
}

async function openRun(entry, options = new URLSearchParams()) {
  stop();
  request?.abort();
  request = new AbortController();
  const {signal} = request;
  discardMovie();
  session = null;
  $('workspace').hidden = true;
  $('loading').hidden = false;
  $('loading').textContent = 'Loading recording…';
  $('back').href = './';
  const dir = entry.path ? `./media/${entry.path}` : './media';
  try {
    const [replay, map] = await Promise.all([
      loadReplay(entry.replay_pack_url, signal, parseReplay, () => json(`${dir}/replay.json`, signal)),
      roadContext(entry, signal)
    ]);
    if (signal.aborted) return;
    const cameras = (entry.cameras ?? replay.cameras ?? ['front','chase']).filter(c => Object.hasOwn(views,c));
    session = {...replay, entry, dir, memory:new ReplayMemory(map.deps), road:map.road, bodies:map.bodies, cameras};
    const number = options.has('frame') ? Number(options.get('frame')) : NaN;
    const seconds = options.has('at') ? Number(options.get('at')) : 0;
    index = finite(number) ? replay.frames.findIndex(f => f.frame === number) : -1;
    if (index < 0) index = Math.max(0,replay.times.findLastIndex(t => t <= seconds));
    $('scale').value=entry.path.startsWith('overtake-') ? '24' : '36';
    $('camera').replaceChildren(...cameras.map(c => new Option(views[c],c)));
    $('camera').disabled = !cameras.length;
    if (cameras.includes(options.get('camera'))) $('camera').value = options.get('camera');
    $('scrub').max = String(replay.frames.length-1);
    $('workspace').hidden = false;
    $('loading').hidden = true;
    selectCamera();
    show(index);
  } catch (error) {
    if (!signal.aborted) $('loading').textContent = error.message;
  }
}

function movieMessage(message) {
  if (movie) movie.style.visibility = message ? 'hidden' : 'visible';
  $('movie-status').textContent = message;
  $('movie-status').hidden = !message;
}

function exposureTime(i) {
  // Pre-timestamp exports encoded one CFR picture per telemetry row.
  if (!session.stamped && finite(movie?.duration) && movie.duration > 0) return i * movie.duration / session.frames.length;
  return session.times[i];
}

function selectCamera() {
  discardMovie();
  const view = $('camera').value;
  if (!view) { movieMessage('No footage is listed for this recording.'); return; }
  const v = document.createElement('video');
  movie = v;
  v.muted = true;
  v.playsInline = true;
  v.preload = 'auto';
  v.setAttribute('aria-label', views[view] + ' camera exposure');
  $('movie-stage').prepend(v);
  v.addEventListener('loadedmetadata', () => { if (movie === v) show(index); });
  v.addEventListener('seeked', () => {
    if (movie !== v || v.seeking || Math.abs(v.currentTime - Number(v.dataset.target)) > .002) return;
    movieMessage('');
    schedule();
  });
  v.addEventListener('error', () => {
    if (movie !== v) return;
    movieMessage('This camera movie could not be decoded.');
    schedule();
  });
  v.src = session.entry.camera_urls?.[view] ?? `${session.dir}/${view}.mp4`;
  movieMessage('Loading camera…');
}

function seekMovie() {
  if (!movie || movie.error) { schedule(); return; }
  if (movie.readyState < 1) return;
  const t = exposureTime(index);
  const next = index+1 < session.frames.length ? exposureTime(index+1) : movie.duration;
  if (!finite(next) || next <= t || t >= movie.duration) {
    movieMessage('No movie exposure covers this timestamp.');
    schedule(); return;
  }
  // Seek inside the exposure to avoid rounding onto the previous H.264 presentation timestamp.
  const target = t + (Math.min(next,movie.duration) - t)/2;
  movie.dataset.target = String(target);
  if (!movie.seeking && movie.readyState >= 2 && Math.abs(movie.currentTime-target) < .0001) {
    movieMessage(''); schedule(); return;
  }
  movieMessage('Seeking exposure…');
  movie.currentTime = target;
}

function schedule() {
  clearTimeout(timer);
  if (!playing) return;
  if (index >= session.frames.length-1) { stop(); return; }
  const period = 1000*(exposureTime(index+1)-exposureTime(index));
  timer = setTimeout(() => show(index+1), Math.max(0,period-(performance.now()-frameStarted)));
}

function local(x,y,ctx) {
  const dx=x-ctx.x, dy=y-ctx.z, a=ctx.degrees*rad;
  return [dx*Math.cos(a)+dy*Math.sin(a), -dx*Math.sin(a)+dy*Math.cos(a)];
}

function observations(frame, ctx, objects) {
  const drawn = objects.map(o => ({id:o.key, name:`Display ${o.key} · ${o.label}`, xy:local(o.x,o.z,ctx),
    length:o.length, width:o.width, angle:o.heading-ctx.degrees, opacity:o.opacity, source:o.sourceId}));
  const measured = (frame.tracks || []).filter(t => !t.coasting && finite(t.x) && finite(t.y) &&
    t.sources?.some(s => s === 'lidar' || s === 'radar')).map(t => ({
      id:t.id, name:`Track ${t.id} · ${t.label}`, xy:[t.x,t.y]
    }));
  const observer = [], map = frame.observer_map;
  if (ctx.located && map) {
    for (const actor of map.others || []) {
      if (!finite(actor.x) || !finite(actor.y)) continue;
      const shape=session.bodies?.[actor.blueprint], dims=actor.dimensions_m;
      const origin=local(actor.x,actor.y,ctx);
      const label=[actor.role?.replaceAll('_',' ') || actor.type, actor.actor_id].filter(v => v != null).join(' ');
      const row={id:actor.actor_id, name:label+' · origin', xy:origin, origin, angle:finite(actor.yaw) ? actor.yaw-ctx.degrees : null};
      if (finite(row.angle) && point(shape?.offset_xyz) && finite(shape.rotation?.[1]) && dims?.length===3 &&
          dims.every((v,i)=>finite(v) && v>0 && Math.abs(v-shape.dimensions_m?.[i])<.02)) {
        const [dx,dy]=shape.offset_xyz, yaw=row.angle*rad;
        row.xy=[origin[0]+Math.cos(yaw)*dx-Math.sin(yaw)*dy,origin[1]+Math.sin(yaw)*dx+Math.cos(yaw)*dy];
        row.angle+=shape.rotation[1]; row.length=dims[0]; row.width=dims[2]; row.name=label+' · body';
      }
      observer.push(row);
    }
    for (const [key,label] of [['traffic','Traffic'],['hazard',map.hazard_type || 'Hazard']]) {
      if (!point(map[key])) continue;
      const xy=local(...map[key].slice(0,2),ctx);
      // Some archives repeat the actor in both the rounded scenario position and the actor list.
      if (!observer.some(o => Math.hypot(xy[0]-o.origin[0],xy[1]-o.origin[1]) < .025)) observer.push({name:label+' · origin', xy, origin:xy});
    }
  }
  return {drawn,measured,observer};
}

function svg(tag, attributes, content) {
  const el=document.createElementNS('http://www.w3.org/2000/svg',tag);
  for (const [k,v] of Object.entries(attributes)) el.setAttribute(k,String(v));
  if (content != null) el.textContent=content;
  return el;
}

function renderMap(ctx, layers) {
  const map=$('map'), range=Number($('scale').value), scale=300/range;
  const width=490*map.clientWidth/Math.max(1,map.clientHeight), ox=width/2, oy=303;
  const step=range===72 ? 10 : range===10 ? 1 : range===24 ? 2 : 5;
  map.setAttribute('viewBox',`0 0 ${width} 490`);
  const children=[];
  if (ctx.located && session.road) {
    const c=Math.cos(ctx.degrees*rad)*scale, s=Math.sin(ctx.degrees*rad)*scale;
    const transform=`matrix(${-s} ${-c} ${c} ${-s} ${ox+s*ctx.x-c*ctx.z} ${oy+c*ctx.x+s*ctx.z})`;
    const road=session.roadNode ??= svg('path',{d:session.road,stroke:'#2b404e','stroke-width':1.8,'vector-effect':'non-scaling-stroke',fill:'none'});
    road.setAttribute('transform',transform);
    children.push(road);
  }
  const grid=[];
  for(let x=ox% (scale*step);x<=width;x+=scale*step) grid.push(`M${x},0v490`);
  for(let y=oy% (scale*step);y<=490;y+=scale*step) grid.push(`M0,${y}h${width}`);
  children.push(svg('path',{d:grid.join(''),stroke:'#2c3844',fill:'none'}));
  const annotation={fill:'#c6d4dd','font-size':16};
  for(const forward of [step*2,step*4,-step*2]) {
    const y=oy-forward*scale;
    if(y>15 && y<475) children.push(svg('text',{x:8,y:y-5,...annotation},`${Math.abs(forward)} m ${forward>0?'ahead':'behind'}`));
  }
  const colors={drawn:'#74dfff',observer:'#ffbf74',measured:'#b6eb8d'};
  for (const layer of ['drawn','observer','measured']) {
    if (!$(layer).checked) continue;
    for(const item of layers[layer]) {
      const [forward,right]=item.xy, x=ox+right*scale,y=oy-forward*scale;
      const group=svg('g',{'data-layer':layer,'data-id':item.id ?? '',transform:`translate(${x} ${y}) rotate(${item.angle ?? 0})`,
        stroke:colors[layer],fill:'none','stroke-width':2.3,opacity:item.opacity ?? 1});
      group.append(svg('title',{},item.name));
      if (item.length>0 && item.width>0 && finite(item.angle)) {
        const l=item.length*scale, w=item.width*scale;
        const box=svg('rect',{x:-w/2,y:-l/2,width:w,height:l,fill:layer==='drawn'?colors.drawn:'none','fill-opacity':.24});
        if(layer==='observer') box.setAttribute('stroke-dasharray','5 3');
        group.append(box,svg('path',{d:`M0,0V${-l/2}`}));
      } else if(layer==='measured') group.append(svg('path',{d:'M-5,0H5M0,-5V5'}));
      else {
        group.append(svg('circle',{r:4}));
        if(finite(item.angle)) group.append(svg('path',{d:'M0,0V-9'}));
      }
      children.push(group);
    }
  }
  children.push(svg('path',{d:`M${ox},${oy-9}l6,14 -6,-3 -6,3z`,fill:'#f3f6f9'}));
  children.push(svg('text',{x:ox+10,y:oy+16,...annotation},'Ego'));
  children.push(svg('text',{x:width-11,y:479,'text-anchor':'end',...annotation},`${step} m grid`));
  map.replaceChildren(...children);
}

function renderPositions(ctx,layers) {
  $('pose').textContent = ctx.located ? `Ego world position (${ctx.x.toFixed(2)}, ${ctx.z.toFixed(2)}) m · yaw ${ctx.degrees.toFixed(2)}°. Positions below are forward and right of ego, in metres.`
    : 'Ego world pose is unavailable. Recorded sensor positions are relative to ego.';
  const titles={drawn:'Rider display',observer:'CARLA actors',measured:'Sensor measurements'};
  $('objects').replaceChildren(...Object.entries(titles).map(([key,title]) => {
    const div=document.createElement('div'), h=document.createElement('h3'), list=document.createElement('ul');
    h.textContent=title; h.className=key;
    for(const row of layers[key]) {
      const item=document.createElement('li');
      const source = row.source != null ? ` · track ${row.source}` : '';
      item.textContent=`${row.name}${source} · (${row.xy.map(v => v.toFixed(2)).join(', ')})`;
      list.append(item);
    }
    if (!list.children.length) list.textContent='None recorded for this frame.';
    div.append(h,list); return div;
  }));
}

function link() {
  const q=new URLSearchParams({run:session.entry.trial_id ?? session.entry.path,frame:session.frames[index].frame});
  if ($('camera').value) q.set('camera',$('camera').value);
  const search=`?${q}`;
  if (!playing && location.search!==search) history.replaceState(null,'',search);
  q.delete('frame'); q.set('at',exposureTime(index).toFixed(4));
  $('back').href=`./?${q}`;
}

function show(value) {
  if (!session) return;
  frameStarted=performance.now();
  index=Math.max(0,Math.min(session.frames.length-1,value));
  const frame=session.frames[index], {ctx,objects}=session.memory.at(session.frames,index,()=>0);
  const layers=observations(frame,ctx,objects);
  session.comparison={ctx,layers};
  renderMap(ctx,layers);
  if ($('objects').parentElement.open) renderPositions(ctx,layers);
  $('scrub').value=index;
  $('scrub').setAttribute('aria-valuetext',`Frame ${frame.frame}`);
  $('frame').textContent=`CARLA frame ${frame.frame}`;
  $('count').textContent=`Exposure ${index+1} / ${session.frames.length}`;
  const movieT=exposureTime(index), stamps=[`Movie ${movieT.toFixed(3)} s`];
  if(finite(frame.sim_time)) stamps.push(`Simulation ${frame.sim_time} s`);
  if(finite(frame.camera_timestamp_s)) stamps.push(`Camera timestamp ${frame.camera_timestamp_s.toFixed(3)} s`);
  $('clock').textContent=stamps.join(' · ');
  $('clock').title=finite(frame.camera_timestamp_s) ? `Recorded camera timestamp ${frame.camera_timestamp_s} s` : '';
  $('movie-caption').textContent=session.stamped
    ? (finite(frame.camera_frame) ? `Recorded camera frame ${frame.camera_frame} · ${movieT.toFixed(3)} s` : `Recorded movie time ${movieT.toFixed(3)} s`)
    : 'Legacy recording · aligned by exposure order.';
  $('prev').disabled=index===0;
  $('next').disabled=index===session.frames.length-1;
  link();
  seekMovie();
}

function step(delta) { stop(); show(index+delta); }
$('prev').onclick=()=>step(-1);
$('next').onclick=()=>step(1);
$('scrub').oninput=()=>{stop();show(Number($('scrub').value));};
$('play').onclick=()=>{
  if(playing){stop();return;}
  playing=true; $('play').textContent='Pause';
  show(index===session.frames.length-1 ? 0 : index);
};
$('camera').onchange=()=>{stop();selectCamera();show(index);};
function redrawMap() {
  if (session?.comparison) renderMap(session.comparison.ctx,session.comparison.layers);
}
new ResizeObserver(redrawMap).observe($('map'));
for (const id of ['scale','observer','drawn','measured']) $(id).onchange=redrawMap;
$('objects').parentElement.ontoggle=()=>{
  if ($('objects').parentElement.open && session?.comparison) renderPositions(session.comparison.ctx,session.comparison.layers);
};
$('run').onchange=()=>openRun(catalog[Number($('run').value)]);
addEventListener('keydown',e=>{
  if(!session || e.altKey || e.metaKey || e.ctrlKey || e.shiftKey || e.target.closest('input,select,textarea')) return;
  if(e.key==='ArrowLeft'||e.key==='ArrowRight'){e.preventDefault();step(e.key==='ArrowLeft'?-1:1);}
});
addEventListener('pagehide',()=>{stop();request?.abort();});
async function start() {
  for (const url of ['./media/viewer-catalog.json','./media/catalog.json']) {
    try {
      const data=await json(url), rows=Array.isArray(data)?data:data.records;
      catalog=rows.filter(r=>typeof r?.path==='string' && (!r.path || /^[\w-]+(?:\/[\w-]+)*$/.test(r.path)) && r.label);
      if(catalog.length) break;
    } catch { /* Earlier archives may only have the primary catalog. */ }
  }
  if(!catalog.length){$('loading').textContent='No recorded drives could be loaded.';return;}
  $('run').replaceChildren(...catalog.map((r,i)=>new Option(r.label,i)));
  $('run').disabled=false;
  const requested=query.get('run'), selected=requested===null ? 0 : catalog.findIndex(r=>r.trial_id===requested || r.path===requested);
  if(selected<0){$('run').selectedIndex=-1;$('loading').textContent='This recording is unavailable. Choose a recording above.';return;}
  $('run').value=selected;
  await openRun(catalog[selected],query);
}
start();
