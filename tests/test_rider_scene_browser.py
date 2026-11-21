"""Small software-rendered integration check for the rider scene. Set ADO_BROWSER_QA=1 to run."""
import os
from pathlib import Path
import unittest

from crosswalk.viewio import Live


@unittest.skipUnless(os.environ.get("ADO_BROWSER_QA") == "1", "opt-in browser rendering check")
class RiderSceneBrowserTest(unittest.TestCase):
    def test_registered_returns_heads_and_movie_clock(self):
        from playwright.sync_api import sync_playwright

        chrome = os.environ.get("ADO_CHROME_PATH")
        if not chrome:
            installed = Path(os.environ.get("ProgramFiles", "")) / "Google/Chrome/Application/chrome.exe"
            chrome = str(installed) if installed.exists() else None
        with Live(0) as server, sync_playwright() as playwright:
            browser = playwright.chromium.launch(executable_path=chrome, args=[
                "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--num-raster-threads=1",
            ])
            try:
                page = browser.new_page(viewport={"width": 520, "height": 380})
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.on("console", lambda msg: errors.append(msg.text) if "Shader Error" in msg.text else None)
                page.route("**/qa-scene", lambda route: route.fulfill(
                    content_type="text/html", body='<canvas id="scene" style="width:500px;height:340px"></canvas>'))
                page.goto(server.url + "qa-scene")
                measured = page.evaluate("""async () => {
                  const {PerceptionScene}=await import('/perception-scene.js');
                  const scene=new PerceptionScene(document.querySelector('canvas'));
                  scene.measuredSignals.setMasts([{id:'same',post:[12,0]}],()=>0);
                  const track={id:31,label:'car',x:-10,y:3,z:.8,length:4,width:1.8,height:1.6,
                    heading:0,speed_mps:1,confidence:.8,sources:['rgb','lidar'],cameras:['rear_right']};
                  const frames=[0,1,2].map(i=>({perception_version:'surround-v1',frame:i,sim_time:i*.1,
                    observer_map:{ego:[i,0,0],ego_z:0,path:[[0,0],[45,0]],others:[{type:'walker',x:2,y:1}]},
                    plan_xy:i<2?[[i,0],[i+5,-.5],[i+18,-3.3],[i+40,-3.5]]:undefined,
                    tracks:[{...track,x:-10-i}],lidar_xyz:[[4,2,.9]],lidar:[[4,2]],
                    corner_radar:{radar_rear_right:[[-5,2,1.4,-3.2]]},
                    signals:[{id:'head/a',map_id:'same',x:12,y:2,z:5,state:'red',source:'camera'},
                             {id:'head/b',map_id:'same',x:12,y:6,z:5,state:'green',source:'map'}]}));
                  scene.updateReplay(frames,0,.5);
                  scene.sample(performance.now());
                  const heads=[...scene.measuredSignals.heads.values()];
                  const result={ego:scene.ego.position.x, acquisition:scene.returnGroup.position.x,
                    surface:scene.lidar.geometry.attributes.position.getY(0),
                    rearRadar:scene.cornerRadar.geometry.attributes.position.getY(0),
                    tracks:[...scene.tracks.values()].map(tr=>tr.label),headCount:heads.length,
                    masts:scene.measuredSignals.masts.size,
                    colors:heads.map(head=>head.material.uniforms.uLamp.value),
                    positions:heads.map(head=>head.position.toArray())};
                  const p=scene.preview.geometry.attributes.position;
                  result.selectedPath=scene.preview.visible&&!scene.path.visible&&
                    Math.abs((p.getZ(p.count-2)+p.getZ(p.count-1))/2+3.5)<.001;
                  for (const view of ['follow','overview']){
                    scene.setView(view);scene.placeCamera();scene.renderer.render(scene.scene,scene.camera);
                  }
                  result.overview=scene.preview.visible&&scene.path.visible&&scene.path.material.uniforms.uPreviewSpan.value>39;
                  scene.setLayer('path',false);result.toggle=!scene.preview.visible&&!scene.path.visible;
                  scene.setLayer('path',true);scene.setView('follow');
                  scene.updateReplay(frames,2,0);
                  result.legacyPath=scene.path.visible&&!scene.preview.visible&&scene.path.material.uniforms.uPreviewSpan.value===0;
                  scene.update(null);
                  result.cleared=!scene.returnGroup.visible&&!scene.measuredSignals.heads.size;
                  scene.dispose();
                  return result;
                }""")
                self.assertFalse(errors, errors)
                self.assertEqual(measured["ego"], .5)
                self.assertEqual(measured["acquisition"], 0)
                self.assertAlmostEqual(measured["surface"], .9, places=5)
                self.assertAlmostEqual(measured["rearRadar"], 1.4, places=5)
                self.assertEqual(measured["tracks"], ["car"])
                self.assertEqual(measured["headCount"], 2)
                self.assertEqual(measured["masts"], 1)
                self.assertEqual(measured["colors"], [0, -1])
                self.assertEqual(measured["positions"], [[12, 5, 2], [12, 5, 6]])
                self.assertTrue(all(measured[key] for key in ("selectedPath","overview","toggle","legacyPath")))
                self.assertTrue(measured["cleared"])
            finally:
                browser.close()
