"""Check the rendered Follow camera on native High turn windows and capture a short comparison."""
import base64
import gzip
import hashlib
import json
import os
from io import BytesIO
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw
from playwright.sync_api import sync_playwright

from capture_rider_motion import PUBLIC, ROOT, constrain_browser, fonts, lower_priority, serve, source_files


OUTPUT = ROOT / "results/perception/rider-turns"
FIRST, LAST = 592, 650
SIZE = 400, 224


def main():
    mask = lower_priority()
    output = OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    pack = json.loads(gzip.decompress((PUBLIC / "replay-pack.json.gz").read_bytes()))
    frames = pack["replay"]
    if isinstance(frames, dict):
        frames = frames["frames"]
    for begin, end in pack["path_ranges"]:
        for i in range(begin, end):
            frames[i]["observer_map"]["path"] = pack["shared_path"]
    # Include the complete preceding history during the displayed turn, plus both
    # signed-wrap windows and the other seven major turns for geometry checks.
    turns = json.loads((output / "turns.json").read_text())["turns"]
    selected = set(range(0, LAST+11))
    fps = 10
    for t in turns:
        a, b = (round(s*fps) for s in t["seconds"])
        selected.update(range(a-9, b+10))
    for i in (1550,2452):
        selected.update(range(i-12,i+13))
    rows = [frames[i] for i in sorted(selected)]
    del frames, pack
    source = source_files(None, ROOT / "web")
    data = {k: (v, "application/json" if k.endswith(".json") else "text/javascript") for k,v in source.items()}
    data.update({
        "": (b'<!doctype html><html data-display="day"><style>*{margin:0}canvas{width:400px;height:224px;display:block}</style><canvas></canvas>', "text/html"),
        "frames": (json.dumps(rows, separators=(",",":")).encode(), "application/json"),
        "town": ((PUBLIC / "town.json").read_bytes(), "application/json"),
        "vehicle-assets/waymo-ipace.glb": ((ROOT / "assets/waymo-ipace/waymo-ipace.glb").read_bytes(),"model/gltf-binary"),
    })
    chase = subprocess.check_output([
        "ffmpeg","-v","error","-threads","1","-ss",str(FIRST/fps),"-i",str(PUBLIC/"chase.mp4"),
        "-frames:v",str(LAST-FIRST+1),"-filter_threads","1","-vf","scale=400:224","-f","rawvideo","-pix_fmt","rgb24","-",
    ])
    stride = SIZE[0]*SIZE[1]*3
    assert len(chase)==(LAST-FIRST+1)*stride
    target = output / "right-turn.mp4"
    font = fonts()[1]
    with serve(data) as url, sync_playwright() as p:
        chrome = os.environ.get("ADO_CHROME_PATH",r"C:\Program Files\Google\Chrome\Application\chrome.exe")
        browser = p.chromium.launch(executable_path=chrome, args=[
            "--use-gl=angle","--use-angle=swiftshader","--enable-unsafe-swiftshader","--disable-gpu-compositing",
            "--disable-accelerated-video-decode","--renderer-process-limit=1","--num-raster-threads=1",
        ])
        try:
            page = browser.new_page(viewport={"width":400,"height":224},reduced_motion="reduce")
            errors = []
            page.on("pageerror",lambda error: errors.append(str(error)))
            page.on("console",lambda msg: errors.append(msg.text) if "Shader Error" in msg.text else None)
            page.goto(url)
            constrain_browser(browser,mask)
            renderer = page.evaluate(SETUP)
            constrain_browser(browser,mask)
            result = page.evaluate(CHECKS,{"turns":turns})
            assert not result["failures"], result
            command = ["ffmpeg","-v","error","-y","-f","rawvideo","-pix_fmt","rgb24","-s","800x256","-r","10","-i","-",
                       "-filter_threads","1","-c:v","libx264","-threads","1","-pix_fmt","yuv420p","-crf","20","-movflags","+faststart",str(target)]
            with subprocess.Popen(command,stdin=subprocess.PIPE) as movie:
                with movie.stdin as stream:
                    for j, idx in enumerate(range(FIRST,LAST+1)):
                        png = page.evaluate("i=>capture(i)",idx)
                        image = Image.new("RGB",(800,256),"#15202b")
                        draw = ImageDraw.Draw(image)
                        draw.text((8,6),f"Rider Follow   ·   {idx/fps:.1f} s",fill="white",font=font)
                        draw.text((408,6),f"Recorded chase camera   ·   frame {rows[idx]['frame']}",fill="white",font=font)
                        image.paste(Image.open(BytesIO(base64.b64decode(png))).convert("RGB"),(0,32))
                        image.paste(Image.frombytes("RGB",SIZE,chase[j*stride:(j+1)*stride]),(400,32))
                        stream.write(image.tobytes())
                if movie.wait():
                    raise RuntimeError("Encoding failed")
            assert not errors, errors
        finally:
            browser.close()
    probe = json.loads(subprocess.check_output(["ffprobe","-v","error","-count_frames","-show_entries",
        "stream=width,height,avg_frame_rate,nb_read_frames,codec_name","-of","json",str(target)]))["streams"][0]
    assert int(probe["nb_read_frames"])==LAST-FIRST+1 and probe["avg_frame_rate"]=="10/1"
    result.update(renderer=renderer,clip=probe,sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    (output/"render-check.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))


SETUP = """async () => {
  const {PerceptionScene} = await import('/perception-scene.js');
  const {townFeatures} = await import('/signal-heads.js');
  const {bicycleAssociations} = await import('/rider-association.mjs');
  const rows = await fetch('/frames').then(r=>r.json());
  const town = await fetch('/town').then(r=>r.json());
  const scene = new PerceptionScene(document.querySelector('canvas'));
  cancelAnimationFrame(scene.raf);
  const gl = scene.renderer.getContext(), ext = gl.getExtension('WEBGL_debug_renderer_info');
  const renderer = ext && gl.getParameter(ext.UNMASKED_RENDERER_WEBGL);
  if (!/swiftshader/i.test(renderer)) throw Error('Software renderer required');
  await townFeatures(town.producer.map);
  scene.setMap(town);
  await scene.loadCar('/vehicle-assets/waymo-ipace.glb');
  const indices = new Map(rows.map((r,i)=>[r.frame,i]));
  function draw(frame,fraction=0, cameraFraction=fraction) {
    const i = indices.get(frame);
    scene.setRiderAssociations(bicycleAssociations(rows,i));
    scene.updateReplay(rows,i,fraction,cameraFraction);
    scene.sample(performance.now()); scene.placeCamera();
  }
  Object.assign(window,{rows,indices,scene,draw,capture:(i)=>{
    draw(rows[i].frame);scene.renderer.render(scene.scene,scene.camera);
    return scene.canvas.toDataURL('image/png').split(',')[1];
  }});
  return renderer;
}"""


CHECKS = """({turns}) => {
  const failures=[],reference=new Map();
  const invariant=() => JSON.stringify([
    scene.frame.frame,scene.ego.position.toArray(),scene.ego.rotation.y,
    scene.returnGroup.position.toArray(),scene.returnGroup.rotation.y,
    scene.returnGroup.children.map(c=>[c.geometry.attributes.position.version,c.geometry.drawRange.count]),
    scene.map.matrixWorld.elements,scene.path?.matrixWorld.elements,
    scene.preview.geometry.attributes.position.version,scene.previewSpan,scene.path.material.uniforms.uStation.value,
    ...[...scene.tracks.values()].map(v=>[v.id,v.group.position.toArray(),v.group.rotation.y,v.group.scale.toArray(),v.opacity])]);
  const view=()=>[...scene.camera.position.toArray(),...scene.camera.quaternion.toArray()];
  const pose=(a,b)=>Math.max(...a.map((v,i)=>Math.abs(b[i]-v)));
  const angle=(a,b)=>Math.atan2(Math.sin(a-b),Math.cos(a-b));
  let poses=0,locked=0,rewinds=0,trackFrames=0,peakPan=0,peakEgoError=0,peakAcquisitionError=0,peakCenterDrift=0;
  const ids=new Set([2276,2277,2278,3178,3179,3180]);
  for(const {frames:[start,end]} of turns) for(let f=start;f<=end;f++) ids.add(f);
  const ordered=[...ids].sort((a,b)=>a-b);
  for(const f of ordered){
    draw(f); const row=rows[indices.get(f)], ego=row.observer_map.ego, h=row.observer_map.ego_z;
    const error=Math.max(pose([ego[0],h,ego[1]],scene.ego.position.toArray()),Math.abs(angle(-ego[2]*Math.PI/180,scene.ego.rotation.y)));
    const acquisition=Math.max(pose([ego[0],h,ego[1]],scene.returnGroup.position.toArray()),Math.abs(angle(-ego[2]*Math.PI/180,scene.returnGroup.rotation.y)));
    peakEgoError=Math.max(peakEgoError,error);peakAcquisitionError=Math.max(peakAcquisitionError,acquisition);
    if(error>1e-9||acquisition>1e-9) failures.push(['raw pose',f]);
    poses++; if(scene.tracks.size)trackFrames++;
    const geometry=invariant(),camera=view(),before=scene.cameraYaw;
    const center=scene.ego.position.clone().project((scene.camera.updateMatrixWorld(),scene.camera));
    reference.set(f,camera);
    draw(f,0,.75);const pan=Math.abs(angle(scene.cameraYaw,before))*180/Math.PI;
    peakPan=Math.max(pan,peakPan);
    const during=scene.ego.position.clone().project((scene.camera.updateMatrixWorld(),scene.camera));
    peakCenterDrift=Math.max(peakCenterDrift,during.distanceTo(center));
    if(invariant()!==geometry)failures.push(['camera pan moved geometry',f]); locked++;
    draw(f);scene.sample(1e8);scene.placeCamera();
    if(pose(camera,view())>1e-9)failures.push(['paused',f]);
  }
  for(const f of ordered.filter((_,i)=>i%9===0).reverse()){
    draw(f);if(pose(reference.get(f),view())>1e-9)failures.push(['seek',f]);rewinds++;
  }
  scene.setView('overview');draw(1350);const overview=view();draw(1350,0,.75);
  if(pose(overview,view())>1e-9)failures.push(['overview']);
  scene.update(null);scene.update(rows[indices.get(1350)]);scene.animate(performance.now());cancelAnimationFrame(scene.raf);
  if(scene.followHeading||Math.abs(angle(scene.cameraYaw,scene.yaw))>1e-9)failures.push(['reduced motion live reset']);
  scene.setView('follow');
  if(peakPan<1||peakCenterDrift>1e-9)failures.push(['pivot',peakPan,peakCenterDrift]);
  return {poses,locked,rewinds,trackFrames,peakEgoError,peakAcquisitionError,peakPanDeg:peakPan,peakCenterDriftNdc:peakCenterDrift,failures};
}"""


if __name__=="__main__":
    main()
