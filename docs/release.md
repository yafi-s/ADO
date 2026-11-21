# ADO for Windows

Download the ZIP and `SHA256SUMS.txt` from the same release. With both files in the same folder, run the following command in PowerShell and compare the full hash with the value in `SHA256SUMS.txt`.

```powershell
Get-FileHash .\ADO-v0.4.0-Windows.zip -Algorithm SHA256
Get-FileHash .\ADO-Urban-Pace-1080p.mp4 -Algorithm SHA256
```

Extract the entire ZIP to a writable folder and double-click `ADO\Start-ADO.cmd`. The first launch installs the official CARLA 1.0 Windows server, Town01_Opt and Town03_Opt map content, the matching Python client, and the project's Python dependencies. The two simulator downloads total about 20.8 GB. Allow around 50 GB of free space on the extraction drive for the downloads and unpacked simulator. If a download is interrupted, run `Start-ADO.cmd` again to resume. After setup, use `Start-ADO.cmd` or `Watch-Live.cmd` to open the original Town01 live driving console. Press Ctrl+C in the launcher window to stop it. To open the Town03 console, follow the [city driving instructions](city-driving.md#run-it-locally).

Use 64-bit Windows, 64-bit Python 3.12, and a GPU with Direct3D 12 support and current drivers. The project tested its five-camera profile on an RTX 3070 with 8 GB of VRAM. A compatible NVIDIA card can use DLSS on auxiliary camera views. An internet connection is required for first-time setup. If you already have the official CARLA 1.0 Windows distribution, set `CARLA_ROOT` to its extracted folder before launching from the same terminal. Setup can add Town01_Opt to that installation.

The ZIP includes trained checkpoints, source code, Blender vehicle assets, synchronized recordings, and evaluation logs. It also includes the NanoDet detector and the YOLOX-S surround observer weights with their [Apache 2.0 license](../vendor/yolox/LICENSE) and [source attribution](../vendor/yolox/README.md). Windows setup installs ONNX Runtime for CPU inference on the optional surround observer. Open the MP4 files in `ADO\web\media` to watch without CARLA. To use the 3D replay viewer from the extracted ZIP, run the command below from the `ADO` folder and open `http://127.0.0.1:8765/web/` in your browser. Python's built-in web server serves the local recordings and replay data without installing project dependencies.

```powershell
py -3.12 -m http.server 8765 --bind 127.0.0.1
```

In v0.4, `Watch-City.cmd` opens the Town03 console with the urban pace checkpoint and barrier sensing. The featured Market drive includes six synchronized camera tracks, the frame inspector, and the barrier appearance review. The Windows ZIP includes the original replay data and the exact camera-exposure evidence behind that review. The [v0.4 release](https://github.com/yafi-s/ADO/releases/tag/v0.4.0) also includes the native 1920 × 1080 chase film, `ADO-Urban-Pace-1080p.mp4`. The [public replay](https://yafi-s.github.io/ADO/?run=urban-pace) keeps the camera tracks and compressed replay data on Pages. Large original JSON downloads can link to the release commit to keep the site within its size limit.

The [v0.3 release](https://github.com/yafi-s/ADO/releases/tag/v0.3.0) includes the city cruise checkpoint, the tactical keep, yield, and pass studies, and `ADO-Surround-City-1080p.mp4`. The [v0.2 release](https://github.com/yafi-s/ADO/releases/tag/v0.2.0) includes the original Town03 film and the Behavior v2 training notebook, checkpoint, and matched Town01 comparison. The current ZIP includes the [closed-loop study with its archive provenance](../results/behavior-v2/README.md). The separate Waymo research source and pilot carry the original licenses and attribution. [Waymo research terms](../waymo_training/NOTICE.md)

## Build the release ZIP

Install Git and Python 3.12 or later. Put `ffprobe` from FFmpeg on PATH for the video checks. Commit the evaluated manifest, summaries, compressed traces, recordings, and Builds & logs index with the code and models. Build from the exact release commit or from the tag after it has been created. Use a separate output folder for each version so its checksum file stays with the matching archive.

```powershell
py -3.12 tools/build_windows_release.py --ref v0.4.0 --tag v0.4.0 --output .local/release/v0.4.0 --film-master "C:\Path\To\Capture\master\chase.mp4"
py -3.12 tools/build_windows_release.py --ref v0.3.0 --tag v0.3.0 --output .local/release/v0.3.0 --film-master "C:\Path\To\Capture\master\chase.mp4"
py -3.12 tools/build_windows_release.py --ref v0.2.0 --tag v0.2.0 --output .local/release/v0.2.0
py -3.12 tools/build_windows_release.py --ref v0.1.0 --tag v0.1.0 --output .local/release/v0.1.0
```

The optional `--film-master` argument supplies the chase master produced by the capture. The build checks its size, hash, frame count and encoding against the committed manifest, stages it alongside the ZIP, and includes both files in `SHA256SUMS.txt`. Upload the versioned ZIP, checksum file and staged chase master to the matching GitHub release. The local `release-metadata.json` records the archived commit, ZIP size, and SHA-256. The build reads the selected Git tree so the archive uses the committed source, media and evidence. The external CARLA runtime, virtual environments, local experiments and private Waymo records stay outside the ZIP.
