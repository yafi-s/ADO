"""Capture the same native High frames in a reference and candidate rider display."""
from argparse import ArgumentParser
from base64 import b64decode
from contextlib import contextmanager
import ctypes
from hashlib import sha256
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from itertools import islice
import json
import os
from pathlib import Path
import shutil
import subprocess
from threading import Thread

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "out/journey-captures/long-city-drive-mqt_d1ge/raw"
SNAPSHOT = ROOT / ".local/rider-surround-qa/high-prefix-700/perception.jsonl"
OUTPUT = ROOT / "results/perception/rider-motion"
CACHE = ROOT / ".local/rider-surround-qa/motion-clip"
PUBLIC = ROOT / "web/media/surround-city"
FIRST, LAST, NATIVE_FIRST = 1062, 1152, 727
PREFIX_COUNT = 700
FPS, WIDTH, HEIGHT = 10, 400, 225
CAMERA = {"orbit": 0.21, "pitch": 0.27, "zoom": 0.9}
OFFSET_Y = .14
CAMERAS = ("front", "right", "rear_right")
PAGE = ('<!doctype html><html data-display="day"><meta charset="utf-8">'
        f'<style>body{{margin:0}}canvas{{display:block;width:{WIDTH}px;height:{HEIGHT}px;'
        f'--perception-offset-y:{OFFSET_Y}}}</style><canvas></canvas>')

SETUP = """async ({camera, first, associations}) => {
  const modules = await Promise.all([import('/perception-scene.js'), import('/signal-heads.js')]);
  const riders = associations ? (await import('/rider-association.mjs')).bicycleAssociations : null;
  const [frames, town] = await Promise.all(['/acquisitions', '/captured-town'].map(url => fetch(url).then(r => r.json())));
  const canvas = document.querySelector('canvas');
  const scene = new modules[0].PerceptionScene(canvas);
  cancelAnimationFrame(scene.raf);
  const gl = scene.renderer.getContext();
  const extension = gl.getExtension('WEBGL_debug_renderer_info');
  const renderer = extension ? gl.getParameter(extension.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
  if (!/swiftshader/i.test(renderer)) throw Error(`Expected software WebGL, got ${renderer}`);
  const features = await modules[1].townFeatures(town.producer.map);
  if (features?.map !== town.producer.map) throw Error('Town feature data unavailable');
  scene.setMap(town);
  await scene.loadCar('/vehicle-assets/waymo-ipace.glb');
  scene.setView('follow');
  Object.assign(scene, camera);
  const readGeometry = () => ({
    ego: [...scene.ego.position.toArray(), scene.ego.rotation.y],
    acquisition: [...scene.returnGroup.position.toArray(), scene.returnGroup.rotation.y],
    tracks: Array.from(scene.tracks.values(), track => ({
      key: String(track.id), source: track.sourceId, sourceIds: track.sourceIds, label: track.label,
      position: track.group.position.toArray(), angle: track.group.rotation.y,
      scale: track.group.scale.toArray(), footprint: track.foot.scale.toArray(), alpha: track.opacity,
    })).sort((a,b) => a.key.localeCompare(b.key)),
  });
  const associatedFrames = new Set();
  window.rider = {
    associationFrames: () => [...associatedFrames].sort((a,b) => a-b),
    clear() { scene.update(null); },
    draw(frame, fraction, capture = false) {
      const i = frame - first;
      if (frames[i]?.frame !== frame) throw Error(`Frame ${frame} unavailable`);
      if (riders) {
        const found = riders(frames,i);
        if (found.size) associatedFrames.add(frame);
        scene.setRiderAssociations?.(found);
      }
      scene.updateReplay(frames, i, fraction);
      scene.sample(1000000 + (i + fraction) * 100);
      if (!capture) return readGeometry();
      scene.cameraYaw = scene.yaw;
      scene.placeCamera();
      scene.renderer.render(scene.scene, scene.camera);
      return canvas.toDataURL('image/png').split(',')[1];
    },
    paused() {
      scene.sample(2000000);
      return readGeometry();
    },
  };
  return renderer;
}"""

PROBE = """async ({first, last}) => {
  const {clear, draw, paused} = rider;
  const reference = new Map();
  const results = Object.fromEntries(['paused', 'seek', 'skipped'].map(key => [key, {
    targets: 0, mismatches: [], maxPositionM: 0, maxAngleDeg: 0, maxScale: 0, maxAlpha: 0,
  }]));
  function compare(kind, frame, fraction, current) {
    const target = reference.get(`${frame}:${fraction}`);
    const stats = results[kind];
    stats.targets++;
    const position = (a,b) => Math.hypot(a[0]-b[0], a[1]-b[1], a[2]-b[2]);
    const angle = (a,b) => Math.abs(Math.atan2(Math.sin(a-b),Math.cos(a-b))) * 180 / Math.PI;
    let p = Math.max(position(target.ego,current.ego),position(target.acquisition,current.acquisition));
    let a = Math.max(angle(target.ego[3],current.ego[3]),angle(target.acquisition[3],current.acquisition[3]));
    let s = 0, o = 0;
    const names = tracks => tracks.map(t => [t.key,t.source,t.sourceIds,t.label]);
    const differentTracks = JSON.stringify(names(target.tracks)) !== JSON.stringify(names(current.tracks));
    if (!differentTracks) for (const [i, t] of target.tracks.entries()) {
      const c = current.tracks[i];
      p = Math.max(p,position(t.position,c.position));
      a = Math.max(a,angle(t.angle,c.angle));
      s = Math.max(s,position(t.scale,c.scale),position(t.footprint,c.footprint));
      o = Math.max(o,Math.abs(t.alpha-c.alpha));
    }
    stats.maxPositionM = Math.max(stats.maxPositionM,p);
    stats.maxAngleDeg = Math.max(stats.maxAngleDeg,a);
    stats.maxScale = Math.max(stats.maxScale,s);
    stats.maxAlpha = Math.max(stats.maxAlpha,o);
    if (differentTracks || !Number.isFinite(p+a+s+o) || Math.max(p,a,s,o)>1e-7) {
      stats.mismatches.push({frame,fraction,differentTracks,p,a,s,o});
    }
  }
  for (let f = first; f <= last; f++) for (const fraction of [0, .5]) {
    reference.set(`${f}:${fraction}`,draw(f,fraction));
    compare('paused', f, fraction, paused());
  }
  clear();
  for (const f of [1112,1064,1147,1075,1136,1082,1124,1094,1152,1100,1062]) {
    for (const fraction of [0,.5]) compare('seek',f,fraction,draw(f,fraction));
  }
  clear();
  for (let f=first; f<=last; f+=3) for (const fraction of [0,.5]) {
    compare('skipped',f,fraction,draw(f,fraction));
  }
  clear();
  const bytes = new TextEncoder().encode(JSON.stringify([...reference]));
  const digest = new Uint8Array(await crypto.subtle.digest('SHA-256',bytes));
  return {checks: results, associationFrames: rider.associationFrames(),
    straightSha256: [...digest].map(b => b.toString(16).padStart(2,'0')).join('')};
}"""


def lower_priority():
    if os.name == "nt":
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        kernel.SetPriorityClass.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        kernel.GetProcessAffinityMask.argtypes = (wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t))
        kernel.SetProcessAffinityMask.argtypes = (wintypes.HANDLE, ctypes.c_size_t)
        process = kernel.GetCurrentProcess()
        if not kernel.SetPriorityClass(process, 0x4000):  # BELOW_NORMAL_PRIORITY_CLASS
            raise ctypes.WinError(ctypes.get_last_error())
        allowed, system = ctypes.c_size_t(), ctypes.c_size_t()
        if not kernel.GetProcessAffinityMask(process, ctypes.byref(allowed), ctypes.byref(system)):
            raise ctypes.WinError(ctypes.get_last_error())
        cores = [1 << bit for bit in range(allowed.value.bit_length()) if allowed.value & (1 << bit)]
        mask = sum(cores[-2:])
        if not kernel.SetProcessAffinityMask(process, ctypes.c_size_t(mask)):
            raise ctypes.WinError(ctypes.get_last_error())
        return mask
    else:
        os.nice(10)
        return None


def constrain_browser(browser, mask):
    if mask is None:
        return
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.SetPriorityClass.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel.SetProcessAffinityMask.argtypes = (wintypes.HANDLE, ctypes.c_size_t)
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    session = browser.new_browser_cdp_session()
    try:
        processes = session.send("SystemInfo.getProcessInfo")["processInfo"]
    finally:
        session.detach()
    for process in processes:
        # Chrome can promote the SwiftShader worker or reset its affinity after launch.
        handle = kernel.OpenProcess(0x0200, False, int(process["id"]))  # PROCESS_SET_INFORMATION
        if not handle:
            if ctypes.get_last_error() == 87:  # A short-lived child already exited.
                continue
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            if not kernel.SetPriorityClass(handle, 0x4000) or not kernel.SetProcessAffinityMask(handle, mask):
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            kernel.CloseHandle(handle)


def source_files(revision, folder):
    files = [
        "perception-scene.js", "object-memory.mjs", "road-users.js", "signal-heads.js",
        "sensor-returns.mjs", "lamp-observations.mjs", "vendor/city-view/three-view.js",
        "media/map-features/Town03_Opt.json",
    ]
    optional = "rider-association.mjs"
    if revision:
        exists = subprocess.check_output(["git", "ls-tree", "--name-only", revision, f"web/{optional}"], cwd=ROOT)
        if exists:
            files.append(optional)
        return {name: subprocess.check_output(["git", "show", f"{revision}:web/{name}"], cwd=ROOT) for name in files}
    if (folder / optional).is_file():
        files.append(optional)
    return {name: (folder / name).read_bytes() for name in files}


@contextmanager
def serve(data):
    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            name = self.path.lstrip("/")
            if name not in data:
                self.send_error(404)
                return
            content, mime = data[name]
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def acquisitions(public_dir):
    if public_dir is None:
        payload = SNAPSHOT.read_bytes()
        rows = [json.loads(line) for line in payload.splitlines()]
        with (RAW / "perception.jsonl").open("rb") as native:
            if b"".join(islice(native, len(rows))) != payload:
                raise ValueError("The snapshot is not the prefix of the native camera capture")
    else:
        published = json.loads((public_dir / "replay.json").read_bytes())
        if not isinstance(published, list):
            raise ValueError("Expected a public replay.json containing an array of acquisitions")
        rows = published[:PREFIX_COUNT]
        if any(row.get("video_index") != i or row.get("camera_frame") != row["frame"] or
               abs(row.get("video_time", -1) - i / FPS) >= 1e-7 for i, row in enumerate(rows)):
            raise ValueError("Public replay camera acquisition indices are missing or misaligned")
        payload = b"\n".join(json.dumps(row, separators=(",", ":"), allow_nan=False).encode() for row in rows) + b"\n"
    if len(rows) != PREFIX_COUNT or any(row["frame"] != NATIVE_FIRST + i or
                              abs(row["sim_time"] - i / FPS) >= 1e-7 for i, row in enumerate(rows)):
        raise ValueError("Expected 700 contiguous native acquisitions at 10 Hz starting from frame 727")
    return payload, rows


def render(label, revision, folder, chrome, smoke, mask, probe_only=False, public_dir=None):
    from playwright.sync_api import sync_playwright

    payload, _ = acquisitions(public_dir)
    town = ((public_dir or RAW) / "town.json").read_bytes()
    car = (ROOT / "assets/waymo-ipace/waymo-ipace.glb").read_bytes()
    source = source_files(revision, folder)
    data = {name: (body, "application/json" if name.endswith(".json") else "text/javascript")
            for name, body in source.items()}
    data.update({
        "": (PAGE.encode(), "text/html"),
        "acquisitions": (b"[" + b",".join(payload.splitlines()) + b"]", "application/json"),
        "captured-town": (town, "application/json"),
        "vehicle-assets/waymo-ipace.glb": (car, "model/gltf-binary"),
    })
    cache = CACHE if public_dir is None else CACHE / "public"
    dest = cache / (f"{label}-preview" if smoke else label)
    if not probe_only:
        dest.mkdir(parents=True, exist_ok=True)
    args = [
        "--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
        "--disable-gpu-compositing", "--disable-accelerated-video-decode", "--disable-accelerated-video-encode",
        "--renderer-process-limit=1", "--num-raster-threads=1",
    ]
    with serve(data) as url, sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=chrome, args=args)
        errors = []
        try:
            page = browser.new_page(viewport={"width": WIDTH, "height": HEIGHT}, device_scale_factor=1)
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.on("console", lambda message: errors.append(message.text) if "Shader Error" in message.text else None)
            page.goto(url)
            constrain_browser(browser, mask)
            renderer = page.evaluate(SETUP, {"camera": CAMERA, "first": NATIVE_FIRST, "associations": "rider-association.mjs" in source})
            constrain_browser(browser, mask)
            if not probe_only:
                print(renderer, flush=True)
            checks = None if smoke else page.evaluate(PROBE, {"first": FIRST, "last": LAST})
            selected = [] if probe_only else ([1063, 1095, 1141] if smoke else range(FIRST, LAST + 1))
            start = FIRST
            for frame in selected:
                png = page.evaluate("""({start, frame}) => {
                  for (let n = start; n < frame; n++) rider.draw(n,0);
                  return rider.draw(frame,0,true);
                }""", {"start": start, "frame": frame})
                (dest / f"{frame}.png").write_bytes(b64decode(png))
                start = frame + 1
            if errors:
                raise RuntimeError(errors)
        finally:
            browser.close()
    if checks is not None:
        evidence = {
            "source": revision or "workspace", "renderer": renderer,
            "geometry": checks["checks"], "straight_geometry_sha256": checks["straightSha256"],
            "cycle_association_frames": checks["associationFrames"],
            "inputs_sha256": {"snapshot": sha256(payload).hexdigest(), "town": sha256(town).hexdigest(), "vehicle": sha256(car).hexdigest()},
            "source_sha256": {name: sha256(body).hexdigest() for name, body in source.items()},
        }
        if probe_only:
            return evidence
        (dest / "geometry.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")


def mjpeg_frames(path):
    marker, end = b"\xff\xd8\xff", b"\xff\xd9"
    selected = []
    frame, chunk = 0, b""
    with path.open("rb") as source:
        while frame <= LAST - NATIVE_FIRST:
            block = source.read(256 * 1024)
            if not block:
                raise ValueError(f"Truncated stream at native index {frame}: {path}")
            chunk += block
            while frame <= LAST - NATIVE_FIRST:
                if len(chunk) < len(marker):
                    break
                if not chunk.startswith(marker):
                    raise ValueError(f"Invalid JPEG boundary at native index {frame}: {path}")
                cut = chunk.find(end, len(marker))
                if cut < 0:
                    break
                cut += len(end)
                if frame >= FIRST - NATIVE_FIRST:
                    selected.append(chunk[:cut])
                chunk = chunk[cut:]
                frame += 1
    return selected


def public_camera_frames(path):
    first, last = FIRST - NATIVE_FIRST, LAST - NATIVE_FIRST
    size = (320, 180)
    stride = size[0] * size[1] * 3
    data = subprocess.check_output([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-threads", "1", "-i", str(path),
        "-an", "-filter_threads", "1", "-vf", f"select=between(n\\,{first}\\,{last}),scale={size[0]}:{size[1]}",
        "-fps_mode", "passthrough", "-frames:v", str(last - first + 1), "-threads", "1",
        "-pix_fmt", "rgb24", "-f", "rawvideo", "-",
    ])
    if len(data) != (last - first + 1) * stride:
        raise ValueError(f"The video is missing frames in the comparison interval: {path}")
    frames = []
    for start in range(0, len(data), stride):
        encoded = BytesIO()
        Image.frombytes("RGB", size, data[start:start + stride]).save(encoded, format="PNG")
        frames.append(encoded.getvalue())
    return frames


def fonts():
    font_path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    if (font_path / "segoeuib.ttf").is_file():
        return (
            ImageFont.truetype(str(font_path / "segoeuib.ttf"), 22),
            ImageFont.truetype(str(font_path / "segoeuib.ttf"), 15),
            ImageFont.truetype(str(font_path / "segoeui.ttf"), 14),
            ImageFont.truetype(str(font_path / "segoeuib.ttf"), 11),
        )
    return tuple(ImageFont.load_default(size=size) for size in (22, 15, 14, 11))


def panel(frame, row, camera_images, typefaces, size, get_scene):
    title, heading, detail, small = typefaces
    canvas = Image.new("RGB", size, "#101722")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), "Occluded pedestrian and roadside van", fill="#f5f8ff", font=title)
    draw.text((size[0] - 12, 12), f"HIGH   •   FRAME {frame}   •   {row['sim_time']:.1f} s", fill="#e4f3ff", font=heading, anchor="ra")
    draw.line((12, 40, size[0] - 12, 40), fill="#384559")
    labels = ("Former rider display", "Updated rider display", "Native simulator cameras")
    for index, label in enumerate(labels):
        x = 12 + index * (WIDTH + 8)
        draw.text((x, 47), label, fill="#f5f8ff", font=heading)
        if index < 2:
            canvas.paste(get_scene(index), (x, 70))
            continue
        draw.rectangle((x, 70, x + WIDTH - 1, 294), fill="#080d13")
        for camera_index, (name, jpg) in enumerate(zip(CAMERAS, camera_images)):
            with Image.open(BytesIO(jpg)) as camera:
                camera.thumbnail((198, 111), Image.Resampling.LANCZOS)
                center = x + (200 if camera_index == 0 else 100 + 200 * (camera_index - 1))
                y = 70 if camera_index == 0 else 184
                left = center - camera.width // 2
                canvas.paste(camera.convert("RGB"), (left, y))
                label = name.replace("_", " ").upper()
                draw.rectangle((left + 3, y + 3, left + 11 + int(draw.textlength(label, font=small)), y + 20), fill="#121922")
                draw.text((left + 7, y + 4), label, fill="#fff", font=small)
    draw.text((12, 303), "Town03_Opt, seed 211  |  Same acquired frame on every panel  |  Follow camera, fixed orbit and zoom", fill="#bcccdd", font=detail)
    return canvas


def assemble(smoke, public_dir):
    _, rows = acquisitions(public_dir)
    cache = CACHE if public_dir is None else CACHE / "public"
    output = OUTPUT if public_dir is None else cache / "comparison"
    if public_dir is None:
        cameras = [mjpeg_frames(RAW / f"{name}.mjpeg") for name in CAMERAS]
    else:
        cameras = [public_camera_frames(public_dir / f"{name}.mp4") for name in CAMERAS]
    size = (1240, 330)
    typefaces = fonts()

    def picture(frame):
        def scene(index):
            label = "before" if index == 0 else "after"
            if smoke:
                label += "-preview"
            with Image.open(cache / label / f"{frame}.png") as source:
                return source.convert("RGB")

        return panel(frame, rows[frame - NATIVE_FIRST], [c[frame - FIRST] for c in cameras], typefaces, size, scene)

    if smoke:
        for frame in (1063, 1095, 1141):
            picture(frame).save(cache / f"preview-{frame}.png")
        return
    evidence = {}
    for label in ("before", "after"):
        folder = cache / label
        record = folder / "geometry.json"
        newest_frame = max((folder / f"{frame}.png").stat().st_mtime_ns for frame in range(FIRST, LAST + 1))
        if record.stat().st_mtime_ns < newest_frame:
            raise ValueError(f"The {label} capture is incomplete. Run its capture step again.")
        evidence[label] = json.loads(record.read_text(encoding="utf-8"))
    if evidence["before"]["inputs_sha256"] != evidence["after"]["inputs_sha256"]:
        raise ValueError("The two rider captures do not use the same inputs")
    output.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError("ffmpeg is required to encode the comparison")
    out = output / "high-occluded-pedestrian-and-van.mp4"
    temporary = cache / out.name
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-threads", "1", "-f", "rawvideo",
               "-pixel_format", "rgb24", "-video_size", f"{size[0]}x{size[1]}", "-framerate", str(FPS), "-i", "-",
               "-an", "-filter_threads", "1", "-c:v", "libx264", "-threads", "1", "-pix_fmt", "yuv420p",
               "-preset", "fast", "-crf", "18", "-movflags", "+faststart", str(temporary)]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
        with process.stdin as stream:
            for frame in range(FIRST, LAST + 1):
                stream.write(picture(frame).tobytes())
        if process.wait():
            raise RuntimeError("ffmpeg did not encode the comparison")
    probe = subprocess.check_output([
        "ffprobe", "-v", "error", "-threads", "1", "-count_frames",
        "-show_entries", "stream=codec_name,codec_type,pix_fmt,width,height,avg_frame_rate,nb_read_frames,duration",
        "-of", "json", str(temporary),
    ])
    stream, = json.loads(probe)["streams"]
    count = LAST - FIRST + 1
    actual = (stream["codec_name"], stream["codec_type"], stream["pix_fmt"], stream["width"], stream["height"],
              stream["avg_frame_rate"], int(stream["nb_read_frames"]), float(stream["duration"]))
    if actual != ("h264", "video", "yuv420p", *size, f"{FPS}/1", count, count / FPS):
        raise RuntimeError(f"Encoded video does not match the captured frames: {stream}")
    evidence.update({
        "frame_range": [FIRST, LAST], "native_index_range": [FIRST - NATIVE_FIRST, LAST - NATIVE_FIRST],
        "fps": FPS, "panels": {"size": [WIDTH, HEIGHT], "camera": CAMERA, "offset_y": OFFSET_Y, "palette": "day"},
        "camera_selection_sha256": {name: sha256(b"".join(data)).hexdigest() for name, data in zip(CAMERAS, cameras)},
        "camera_source": "native JPEGs" if public_dir is None else "PNG frames decoded from the public MP4s",
        "video": {"file": out.name, "bytes": temporary.stat().st_size, "sha256": sha256(temporary.read_bytes()).hexdigest(), **stream},
    })
    temporary.replace(out)
    (output / "high-replay-geometry.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")


def verify_working(folder, chrome, mask):
    path = OUTPUT / "high-replay-geometry.json"
    evidence = json.loads(path.read_text(encoding="utf-8"))
    frozen = render("probe", None, folder, chrome, False, mask, probe_only=True)
    if evidence["after"]["source_sha256"] != frozen["source_sha256"]:
        raise ValueError("The frozen source files differ from those used to render the clip")
    working = render("probe", None, ROOT / "web", chrome, False, mask, probe_only=True)
    if frozen["inputs_sha256"] != working["inputs_sha256"] or frozen["straight_geometry_sha256"] != working["straight_geometry_sha256"]:
        raise ValueError("The working renderer differs from the clip's geometry")
    if any(stats["mismatches"] for result in (frozen, working) for stats in result["geometry"].values()):
        raise ValueError("Replay geometry depends on playback order")
    evidence["working_geometry_verification"] = {
        "exact_straight_samples": 2 * (LAST - FIRST + 1),
        "frozen_straight_geometry_sha256": frozen["straight_geometry_sha256"],
        "working": working,
    }
    path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")


def verify_public(public_dir, chrome, mask):
    path = OUTPUT / "high-replay-geometry.json"
    evidence = json.loads(path.read_text(encoding="utf-8"))
    result = render("probe", None, ROOT / "web", chrome, False, mask, probe_only=True, public_dir=public_dir)
    expected = evidence["working_geometry_verification"]["frozen_straight_geometry_sha256"]
    if result["straight_geometry_sha256"] != expected or any(stat["mismatches"] for stat in result["geometry"].values()):
        raise ValueError(f"Published replay geometry differs from the native comparison: {result['straight_geometry_sha256']}")
    print(json.dumps(result, indent=2))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("before", "after", "assemble", "verify-working", "verify-public"))
    parser.add_argument("--revision", default="35b0af102a9d4ebbf9e0c1619df19f0b6b88a1fc")
    parser.add_argument("--candidate-sources", type=Path, help="Directory with the candidate renderer source files")
    parser.add_argument("--public-dir", type=Path, help="Use the delivered replay.json, town.json and MP4s in this directory")
    parser.add_argument("--chrome", default=os.environ.get("ADO_CHROME_PATH", r"C:\Program Files\Google\Chrome\Application\chrome.exe"))
    parser.add_argument("--smoke", action="store_true", help="Preview three frames before a full capture")
    opts = parser.parse_args()
    mask = lower_priority()
    if opts.action == "assemble":
        assemble(opts.smoke, opts.public_dir)
    elif opts.action == "verify-public":
        verify_public(opts.public_dir or PUBLIC, opts.chrome, mask)
    elif opts.action == "verify-working":
        if opts.candidate_sources is None:
            parser.error("verify-working requires --candidate-sources to pin the clip renderer")
        verify_working(opts.candidate_sources, opts.chrome, mask)
    else:
        render(opts.action, opts.revision if opts.action == "before" else None,
               opts.candidate_sources or ROOT / "web", opts.chrome, opts.smoke, mask, public_dir=opts.public_dir)


if __name__ == "__main__":
    main()
