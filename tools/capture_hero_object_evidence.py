"""Pair a true native exposure with the same acquisition in the production rider scene."""
import argparse
from base64 import b64decode
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright

from capture_rider_motion import ROOT, constrain_browser, lower_priority, serve, source_files


SETUP = """async ({target,source})=>{
  const [{PerceptionScene},{applyBarrierReview},{bicycleAssociations},T,{townFeatures}]=await Promise.all([
    import('/perception-scene.js'),import('/barrier-review.mjs'),import('/rider-association.mjs'),
    import('/vendor/city-view/three-view.js'),import('/signal-heads.js')]);
  const [frames,town,review]=await Promise.all(['/frames','/town','/review'].map(u=>fetch(u).then(r=>r.json())));
  // Replay only the required prefix. The producer's full review is separately
  // hash-checked against the evaluator output before this native frame lookup.
  const data=applyBarrierReview(frames,{...review,last_frame:frames.at(-1).frame});
  const i=data.findIndex(f=>f.frame===target);
  const canvas=document.querySelector('canvas'),scene=new PerceptionScene(canvas);
  cancelAnimationFrame(scene.raf);
  await townFeatures(town.producer.map);
  scene.setMap(town);await scene.loadCar('/vehicle-assets/waymo-ipace.glb');
  scene.setView('follow');scene.setRiderAssociations(bicycleAssociations(data,i));
  scene.updateReplay(data,i,0);scene.sample(1000000+i*100);scene.cameraYaw=scene.followHeading.at(i,0);
  scene.placeCamera();scene.renderer.render(scene.scene,scene.camera);
  const object=[...scene.tracks.values()].find(t=>t.sourceId===source||t.sourceIds?.includes(source));
  if(!object)throw Error('Selected source is absent from the production scene');
  const corners=[];
  for(const x of [-.5,.5])for(const y of [0,1])for(const z of [-.5,.5]){
    const p=object.group.localToWorld(new T.Vector3(x,y,z)).project(scene.camera);
    corners.push([(1+p.x)/2,(1-p.y)/2]);
  }
  const bounds=[Math.min(...corners.map(p=>p[0])),Math.min(...corners.map(p=>p[1])),
    Math.max(...corners.map(p=>p[0])),Math.max(...corners.map(p=>p[1]))];
  const gl=scene.renderer.getContext(),ext=gl.getExtension('WEBGL_debug_renderer_info');
  return {bounds,key:object.id,label:object.label,renderer:gl.getParameter(ext?.UNMASKED_RENDERER_WEBGL??gl.RENDERER),
    png:canvas.toDataURL('image/png').split(',')[1]};
}"""


def native_exposure(stream, index):
    pending = b""
    with stream.open("rb") as src:
        for _ in range(index + 1):
            end = pending.find(b"\xff\xd9")
            while end < 0:
                chunk = src.read(512 * 1024)
                if not chunk:
                    raise ValueError("Missing native JPEG for the selected acquisition")
                pending += chunk
                end = pending.find(b"\xff\xd9")
            image, pending = pending[:end + 2], pending[end + 2:]
            if not image.startswith(b"\xff\xd8"):
                raise ValueError("Malformed native JPEG boundary")
    return image


def recorded_inputs(raw, end):
    fields = ("frame", "sim_time", "run_id", "seed", "scenario", "perception_version", "tracks",
              "detections", "camera_detections", "lidar", "lidar_xyz", "radar", "corner_radar", "signals",
              "brake", "plan_xy", "speed", "video_time")
    payload = BytesIO()
    payload.write(b"[")
    with (raw / "perception.jsonl").open("rb") as source:
        for index, line in enumerate(source):
            frame = json.loads(line)
            row = {key: frame[key] for key in fields if key in frame}
            row["observer_map"] = {key: frame["observer_map"][key] for key in ("ego", "ego_z", "path") if key in frame["observer_map"]}
            if index:
                payload.write(b",")
            payload.write(json.dumps(row, separators=(",", ":")).encode())
            if frame["frame"] == end:
                payload.write(b"]")
                return payload.getvalue()
    raise ValueError("Journal does not contain the requested acquisition")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path)
    parser.add_argument("review", type=Path)
    parser.add_argument("metrics", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--case", choices=("board", "cyclist", "unmatched"), default="board")
    parser.add_argument("--example", type=int, default=0, help="Index in the ranked unmatched examples")
    parser.add_argument("--chrome", default=os.environ.get("ADO_CHROME_PATH", r"C:\Program Files\Google\Chrome\Application\chrome.exe"))
    args = parser.parse_args()
    metrics = json.loads(args.metrics.read_bytes())
    if args.case=="unmatched":
        example=metrics["evidence"]["unmatched_current_examples"][args.example]
        chosen=dict(example["example"],display_id=example["display_id"],camera_label=example["example"]["camera_class"])
        residual=f"nearest registered actor {chosen['nearest']['distance_m']:.2f} m"
    else:
        chosen=metrics["evidence"][args.case]
        residual=f"{chosen['reference_residual_m']:.2f} m from actor origin"
    review_bytes = args.review.read_bytes()
    if sha256(review_bytes).hexdigest() != metrics["source"]["barrier_review_sha256"]:
        raise ValueError("Sidecar differs from the evaluated recording")
    src = source_files(None, ROOT / "web")
    src["barrier-review.mjs"] = (ROOT / "web/barrier-review.mjs").read_bytes()
    data = {name: (body, "application/json" if name.endswith(".json") else "text/javascript") for name, body in src.items()}
    data.update({"": (b"<style>*{margin:0}canvas{display:block;width:800px;height:450px}</style><canvas></canvas>", "text/html"),
                 "frames": (recorded_inputs(args.raw, min(metrics["source"]["last_frame"],chosen["frame"]+1)), "application/json"),
                 "review": (review_bytes, "application/json"), "town": ((args.raw / "town.json").read_bytes(), "application/json"),
                 "vehicle-assets/waymo-ipace.glb": ((ROOT / "assets/waymo-ipace/waymo-ipace.glb").read_bytes(),"model/gltf-binary")})
    mask = lower_priority()
    with serve(data) as url, sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=args.chrome, args=["--use-gl=angle","--use-angle=swiftshader",
            "--enable-unsafe-swiftshader","--renderer-process-limit=1","--num-raster-threads=1"])
        try:
            page=browser.new_page(viewport={"width":800,"height":450}, device_scale_factor=1)
            page.goto(url)
            constrain_browser(browser,mask)
            rendering=page.evaluate(SETUP,{"target":chosen["frame"],"source":chosen["source_id"]})
        finally:
            browser.close()
    if "swiftshader" not in rendering["renderer"].lower() or rendering["key"] != chosen["display_id"]:
        raise ValueError("Renderer or actual display identity differs from the evaluated source")
    raw_jpeg=native_exposure(args.raw/(chosen["camera"]+".mjpeg"),chosen["frame"]-metrics["source"]["first_frame"])
    native=Image.open(BytesIO(raw_jpeg)).convert("RGB").resize((800,450),Image.Resampling.LANCZOS)
    rider=Image.open(BytesIO(b64decode(rendering.pop("png")))).convert("RGB").resize((800,450),Image.Resampling.LANCZOS)
    strip=Image.new("RGB",(1600,516),"#15202c")
    title=ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf",23)
    small=ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf",17)
    row=ImageDraw.Draw(strip)
    for i,(picture,box,heading,sub) in enumerate([
        (native,chosen["box"],f"Native {chosen['camera'].replace('_',' ')} camera",f"Camera class {chosen['camera_label']} · score {chosen['score']:.2f}"),
        (rider,rendering["bounds"],"Rider display",f"Object {rendering['key']} · source {chosen['source_id']} · {residual}")]):
        draw=ImageDraw.Draw(picture)
        bounds=(box[0]*800,box[1]*450,box[2]*800,box[3]*450)
        draw.rectangle(bounds,outline="#ffda50",width=3)
        strip.paste(picture,(i*800,66))
        row.text((i*800+12,4),heading,font=title,fill="white")
        row.text((i*800+12,35),f"Frame {chosen['frame']} · {chosen['sim_time']:.1f} s · {sub}",font=small,fill="#c5d6e6")
    strip.save(args.output,quality=92)
    print(json.dumps({"frame":chosen["frame"],"native_jpeg_sha256":sha256(raw_jpeg).hexdigest(),**rendering}))


if __name__=="__main__":
    main()
