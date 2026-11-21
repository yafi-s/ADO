"""Measure actual rasterized road-user pixels at the default rider camera and laptop viewport."""
import argparse
from base64 import b64decode
from io import BytesIO
import json
import os
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageDraw
from playwright.sync_api import sync_playwright
from capture_hero_object_evidence import recorded_inputs
from capture_rider_motion import ROOT, constrain_browser, lower_priority, serve, source_files


RENDER = """async ({targets})=>{
 const [{PerceptionScene},{applyBarrierReview},{townFeatures}]=await Promise.all([
   import('/perception-scene.js'),import('/barrier-review.mjs'),import('/signal-heads.js')]);
 const [frames,town,review]=await Promise.all(['/frames','/town','/review'].map(u=>fetch(u).then(r=>r.json())));
 const data=applyBarrierReview(frames,review?{...review,last_frame:frames.at(-1).frame}:null);
 const canvas=document.querySelector('#scene'),scene=new PerceptionScene(canvas);
 cancelAnimationFrame(scene.raf);
 await townFeatures(town.producer.map);scene.setMap(town);await scene.loadCar('/vehicle-assets/waymo-ipace.glb');
 scene.setView('follow');
 const answer=[];
 for(const {frame,source} of targets){
   const i=data.findIndex(f=>f.frame===frame);scene.updateReplay(data,i,0);
   scene.sample(1000000+i*100);scene.cameraYaw=scene.followHeading.at(i,0);scene.placeCamera();
   const track=[...scene.tracks.values()].find(t=>t.sourceId===source||t.sourceIds?.includes(source));
   if(!track)throw Error('Missing production track '+source+' at '+frame);
   const snapshot=()=>{scene.renderer.render(scene.scene,scene.camera);return canvas.toDataURL('image/png').split(',')[1];};
   const rendered=snapshot();
   // Keep footprint, trail and surrounding scene identical. Subtract the target mesh and its optical accent alone.
   const body=track.group.children[0]; body.visible=false;
   const accentOn=track.distanceAccent?.visible;
   if(track.distanceAccent)track.distanceAccent.visible=false;
   const without=snapshot();body.visible=true;
   if(track.distanceAccent)track.distanceAccent.visible=accentOn;
   answer.push({rendered,without,body:track.label});
 }
 scene.dispose();return answer;
}"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--targets", nargs="+", required=True, help="Reference IDs with a half-visible display match")
    args = parser.parse_args()
    web = ROOT / "web"
    report = json.loads(args.report.read_bytes())
    selected = []
    for key in args.targets:
        actor = next(a for a in report["encounters"] if str(a["id"]) == key)
        selected.append(dict(actor=key, role=actor["role"], **actor["first"]["display"]))
    assets = source_files(None, web)
    for name in ("ado-home.css", "display.css", "barrier-review.mjs"):
        assets[name] = (web / name).read_bytes()
    bundle = {k: (v, "text/css" if k.endswith(".css") else "application/json" if k.endswith(".json")
                  else "text/javascript") for k, v in assets.items()}
    html = re.sub(r'<script\b.*?</script>', '', (web / "index.html").read_text(encoding="utf8"), flags=re.S)
    bundle.update({"": (html.encode(), "text/html"), "frames": (recorded_inputs(args.raw, max(
        r["frame"] for r in selected)+1), "application/json"), "town": ((args.raw / "town.json").read_bytes(),
        "application/json"), "review": (args.review.read_bytes() if args.review else b"null", "application/json"),
        "vehicle-assets/waymo-ipace.glb": ((ROOT / "assets/waymo-ipace/waymo-ipace.glb").read_bytes(),"model/gltf-binary")})
    cpu = lower_priority()
    with serve(bundle) as url, sync_playwright() as pw:
        browser=pw.chromium.launch(executable_path=os.environ.get("ADO_CHROME_PATH",r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
            args=["--use-gl=angle","--use-angle=swiftshader","--enable-unsafe-swiftshader","--renderer-process-limit=1","--num-raster-threads=1"])
        try:
            constrain_browser(browser,cpu)
            page=browser.new_page(viewport={"width":1366,"height":768},device_scale_factor=1)
            page.goto(url); draws=page.evaluate(RENDER,dict(targets=selected))
        finally:
            browser.close()
    args.output.mkdir(exist_ok=True,parents=True)
    for row,draw in zip(selected, draws, strict=True):
        rendered,hidden=[Image.open(BytesIO(b64decode(draw[k]))).convert("RGB") for k in ("rendered","without")]
        a,b=np.asarray(rendered,dtype=np.int16),np.asarray(hidden,dtype=np.int16)
        delta=np.abs(a-b).max(axis=2)
        # Include only visibly changed pixels. Jagged edges sometimes change by a single 8-bit step.
        y,x=np.where(delta>=12)
        row.update(symbol=draw["body"],scene_size_px=rendered.size,pixels_rgb_delta_ge_12=len(x))
        if len(x):
            extent=[int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)]
            row["extent_px"]=[extent[2]-extent[0],extent[3]-extent[1]]
            row["screen_bounds_px"]=extent
            row["median_pixel_contrast"]=float(np.median(delta[y,x]))
            annotated=rendered.copy()
            d=ImageDraw.Draw(annotated);cx=(extent[0]+extent[2])//2
            d.line((cx,extent[1]-29,cx,extent[1]-5), fill="#ef8d27",width=2)
            annotated.save(args.output / f"frame-{row['frame']}.jpg",quality=92)
    (args.output / "visibility.json").write_text(json.dumps(selected,indent=2)+"\n")


if __name__=="__main__":
    main()
