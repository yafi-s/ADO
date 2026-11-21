// Run from the repo root. The optional argument selects a local replay pack.
import {readFileSync} from 'node:fs';
import {gunzipSync} from 'node:zlib';
import {createHash} from 'node:crypto';
import {FollowHeading} from '../web/follow-heading.mjs';
import {unpackReplay} from '../web/replay-pack.mjs';

const input = readFileSync(process.argv[2] ?? 'web/media/surround-city/replay-pack.json.gz');
const replay = unpackReplay(JSON.parse(gunzipSync(input)));
const frames = replay.frames ?? replay;
const heading = new FollowHeading(frames);
const wrap = deg => ((deg % 360) + 540) % 360 - 180;
const deg = 180/Math.PI;
const hz = 60;
const fps = 10;
const round = n => Math.round(n*1000)/1000;
const quantile = (xs, q) => [...xs].sort((a,b)=>a-b)[Math.floor(q*(xs.length-1))];
const stats = xs => ({rms:round(Math.sqrt(xs.reduce((s,x)=>s+x*x,0)/xs.length)),
  p95:round(quantile(xs.map(Math.abs), .95)), max:round(Math.max(...xs.map(Math.abs)))});
const yaws = frames.map(f=>f.observer_map.ego[2]);
const unwrapped = [];
for (const [i,y] of yaws.entries()) unwrapped.push(i ? unwrapped[i-1]+wrap(y-yaws[i-1]) : y);
const d = (xs, freq) => xs.slice(1).map((v,i)=>(v-xs[i])*freq);
const groups = [];
// Major turns cover at least 20° in one direction. Join brief dips below 8°/s.
for (let i=3;i<frames.length-3;i++) {
  const rate=(unwrapped[i+3]-unwrapped[i-3])/.6;
  if (Math.abs(rate)<=8) continue;
  const g=groups.at(-1);
  if (g && i-g.end<8 && g.sign===Math.sign(rate)) g.end=i;
  else groups.push({start:i,end:i,sign:Math.sign(rate)});
}
const turns=groups.filter(g=>Math.abs(unwrapped[g.end]-unwrapped[g.start])>=20);
const active=index=>turns.some(g=>index>=g.start && index<=g.end);
const count=frames.length;
if(frames.some((f,i)=>f.frame!==f.camera_frame || f.video_index!==i || Math.abs(f.sim_time-i/fps)>1e-7 ||
  Math.abs(f.video_time-i/fps)>1e-7)) throw Error('Expected the public 10 Hz exposure join');

function compare(mode) {
  let prior=unwrapped[0],lastDraw=0;
  const old=[],next=[],raw=[];
  for(let k=0;k<=(count-1)*hz/fps;k++) {
    const index=Math.floor(k/(hz/fps)),sub=k%(hz/fps),frac=sub/(hz/fps);
    const native=unwrapped[index], continuous=native+frac*((unwrapped[index+1]??native)-native);
    const target=mode==='movie'?native:continuous;
    // Previous animate() at nominal 60 Hz: every changing exposure/fraction dirties
    // the scene; otherwise settling draws only after 24 ms and until .0006 rad.
    const dirty=sub===0||mode==='sensorOnly';
    if(dirty || (Math.abs(target-prior)>.0006*deg && (k-lastDraw)/hz>=.024)){
      prior+=(target-prior)*(1-Math.exp(-Math.min(.1,(k-lastDraw)/hz)/.32));
      lastDraw=k;
    }
    old.push(prior); next.push(-heading.at(index,frac)*deg); raw.push(continuous);
  }
  const measure=xs=>{
    const jerk=d(d(d(xs,hz),hz),hz);
    const errors=xs.map((x,k)=>x-raw[k]).filter((_,k)=>active(k*fps/hz));
    // Find constant time shift minimizing heading MSE on the turn windows (±0.5 s).
    let best={ms:0,mse:Infinity};
    for(let shift=-hz/2;shift<=hz/2;shift++){
      let square=0,n=0;
      for(let k=hz;k<xs.length-hz;k++) if(active(k*fps/hz)){square+=(xs[k]-raw[k-shift])**2;n++;}
      if(square/n<best.mse)best={ms:round(1000*shift/hz),mse:square/n};
    }
    // Report a second jerk value purely on acquisition boundaries, independent of
    // how many display refreshes are available between exposures.
    const knots=xs.filter((_,k)=>k%(hz/fps)===0);
    return {angular_error_deg:stats(errors),best_fit_lag_ms:best.ms,
      render_jerk_deg_s3:stats(jerk.filter((_,k)=>active((k+3)*fps/hz))),
      acquisition_jerk_deg_s3:stats(d(d(d(knots,fps),fps),fps).filter((_,i)=>active(i+3)))};
  };
  return {previous:measure(old),updated:measure(next)};
}
const info=turns.map(g=>({
  frames:[frames[g.start].frame,frames[g.end].frame],seconds:[frames[g.start].sim_time,frames[g.end].sim_time],
  signed_sweep_deg:round(unwrapped[g.end]-unwrapped[g.start]),
  peak_applied_steer:round(Math.max(...frames.slice(g.start,g.end+1).map(f=>Math.abs(f.applied_control.steer)))),
  peak_commanded_steer:round(Math.max(...frames.slice(g.start,g.end+1).map(f=>Math.abs(f.commanded_control.steer)))),
}));
console.log(JSON.stringify({
  pack_sha256:createHash('sha256').update(input).digest('hex'),exposures:count,
  acquisition_hz:fps,modelled_display_hz:hz,turns:info,
  wrap_frames:frames.filter((_,i)=>i&&Math.abs(yaws[i]-yaws[i-1])>180).map(f=>f.frame),
  max_pose_rounding_deg:round(Math.max(...frames.map((f,i)=>Math.abs(wrap(f.ego_pose.yaw-yaws[i]))))),
  median_chase_minus_body_deg:round(quantile(frames.map(f=>wrap(f.camera_pose.yaw-f.ego_pose.yaw)),.5)),
  visible_movie:compare('movie'),smooth_sensor_only:compare('sensorOnly'),
},null,2));
