# ADO for Windows

Download the ZIP and `SHA256SUMS.txt` from the same release. With both files in the same folder, run the following command in PowerShell and compare the full hash with the value in `SHA256SUMS.txt`.

```powershell
Get-FileHash .\ADO-v0.7.0-Windows.zip -Algorithm SHA256
Get-FileHash .\ADO-City-Maneuvers-1080p.mp4 -Algorithm SHA256
```

Extract the entire ZIP to a writable folder and double-click `ADO\Start-ADO.cmd`. The first launch installs the official CARLA 1.0 Windows server, Town01_Opt and Town03_Opt map content, the matching Python client, and the project's Python dependencies. The two simulator downloads total about 20.8 GB. Allow around 50 GB of free space on the extraction drive for the downloads and unpacked simulator. If a download is interrupted, run `Start-ADO.cmd` again to resume. After setup, use `Start-ADO.cmd` or `Watch-Live.cmd` to open the original Town01 live driving console. Press Ctrl+C in the launcher window to stop it. To open the Town03 console, follow the [city driving instructions](city-driving.md#run-it-locally).

Use 64-bit Windows, 64-bit Python 3.12, and a GPU with Direct3D 12 support and current drivers. The project tested its five-camera profile on an RTX 3070 with 8 GB of VRAM. A compatible NVIDIA card can use DLSS on auxiliary camera views. An internet connection is required for first-time setup. If you already have the official CARLA 1.0 Windows distribution, set `CARLA_ROOT` to its extracted folder before launching from the same terminal. Setup can add Town01_Opt to that installation.

The ZIP includes trained checkpoints, source code, Blender vehicle assets, synchronized recordings, and evaluation logs. It also includes the NanoDet detector and the YOLOX-S surround observer weights with their [Apache 2.0 license](../vendor/yolox/LICENSE) and [source attribution](../vendor/yolox/README.md). Windows setup installs ONNX Runtime for CPU inference on the optional surround observer. Open the MP4 files in `ADO\web\media` to watch without CARLA. To use the 3D replay viewer from the extracted ZIP, run the command below from the `ADO` folder and open `http://127.0.0.1:8765/web/` in your browser. Python's built-in web server serves the local recordings and replay data without installing project dependencies.

```powershell
py -3.12 -m http.server 8765 --bind 127.0.0.1
```

The v0.7 Windows ZIP includes City Maneuvers, its driving checkpoint, six synchronized cameras, and the evidence for its passing and pace tests. `ADO-City-Maneuvers-1080p.mp4` is the native 1920 × 1080 chase master. City Maneuvers is the default drive on Pages. Pages hosts the six City Maneuvers camera tracks and the lossless replay packs. It keeps 1080p chase playback for City Progress, City Life, and Urban Pace. If older cameras or original replay and telemetry downloads exceed the site's 1 GB budget, Pages links them to the exact checkout commit after checking that the archived copy supports browser seeking. [v0.7 release](https://github.com/yafi-s/ADO/releases/tag/v0.7.0) · [City Maneuvers replay](https://yafi-s.github.io/ADO/?run=city-maneuvers)

To run City Maneuvers live, open PowerShell in the extracted `ADO` folder and run this command.

```powershell
.\Watch-City.cmd --model models/city-maneuvers/candidate.npz --population-profile city-maneuvers
```

The v0.6 Windows ZIP includes City Progress, its driving checkpoint, six synchronized camera tracks, and the frame-aligned population reference and execution sources. `ADO-City-Progress-1080p.mp4` is its native chase master. [v0.6 release](https://github.com/yafi-s/ADO/releases/tag/v0.6.0) · [City Progress replay](https://yafi-s.github.io/ADO/?run=city-progress)

The v0.5 Windows ZIP includes City Life with six synchronized cameras, the roadside-clearance checkpoint, and the frame-aligned population reference and execution sources. `ADO-City-Life-1080p.mp4` is its native chase master. The standalone range study remains available from the same release. [v0.5 release](https://github.com/yafi-s/ADO/releases/tag/v0.5.0) · [City Life replay](https://yafi-s.github.io/ADO/?run=city-life)

The v0.4 Market drive includes six synchronized camera tracks, the frame inspector, and the barrier appearance review. The current Windows ZIP keeps the original replay and camera-exposure evidence and includes the cone review from the same capture. The v0.4 release includes the original chase master, `ADO-Urban-Pace-1080p.mp4`. [v0.4 release](https://github.com/yafi-s/ADO/releases/tag/v0.4.0) · [Urban Pace replay](https://yafi-s.github.io/ADO/?run=urban-pace)

The [v0.3 release](https://github.com/yafi-s/ADO/releases/tag/v0.3.0) includes the city cruise checkpoint, the tactical keep, yield, and pass studies, and `ADO-Surround-City-1080p.mp4`. The [v0.2 release](https://github.com/yafi-s/ADO/releases/tag/v0.2.0) includes the original Town03 film and the Behavior v2 training notebook, checkpoint, and matched Town01 comparison. The current ZIP includes the [closed-loop study with its archive provenance](../results/behavior-v2/README.md). The separate Waymo research source and pilot carry the original licenses and attribution. [Waymo research terms](../waymo_training/NOTICE.md)

## Build the release ZIP

Install Git and Python 3.12 or later. Put `ffprobe` from FFmpeg on PATH for the video checks. Commit the evaluated manifest, summaries, compressed traces, recordings, and Builds & logs index with the code and models. Build from the exact release commit or from the tag after it has been created. Use a separate output folder for each version so its checksum file stays with the matching archive.

```powershell
py -3.12 tools/build_windows_release.py --ref v0.7.0 --tag v0.7.0 --output "$env:LOCALAPPDATA\ADO\city-maneuvers\release\v0.7.0" --film-master "C:\Path\To\Capture\master\chase.mp4"
py -3.12 tools/build_windows_release.py --ref v0.6.0 --tag v0.6.0 --output .local/release/v0.6.0 --film-master "C:\Path\To\Capture\master\chase.mp4"
py -3.12 tools/build_windows_release.py --ref v0.5.0 --tag v0.5.0 --output .local/release/v0.5.0 --film-master "C:\Path\To\Capture\master\chase.mp4"
py -3.12 tools/build_windows_release.py --ref v0.4.0 --tag v0.4.0 --output .local/release/v0.4.0 --film-master "C:\Path\To\Capture\master\chase.mp4"
py -3.12 tools/build_windows_release.py --ref v0.3.0 --tag v0.3.0 --output .local/release/v0.3.0 --film-master "C:\Path\To\Capture\master\chase.mp4"
py -3.12 tools/build_windows_release.py --ref v0.2.0 --tag v0.2.0 --output .local/release/v0.2.0
py -3.12 tools/build_windows_release.py --ref v0.1.0 --tag v0.1.0 --output .local/release/v0.1.0
```

`--film-master` supplies the chase master from the featured capture and is required from v0.5 onward. The build checks its size, hash, frame count and encoding against the committed manifest. It also checks the populated drives' safety and density totals against the native frame data and verifies the historical Urban Pace review. For v0.7 it verifies the matched trial captures and checks the two passes, the occupied-gap wait, and the return to lane against the full drive's native evidence. Keep each release asset below 2 GiB. Upload the versioned ZIP, `SHA256SUMS.txt` and staged chase master to the matching GitHub release. `release-metadata.json` records the archived commit and the sizes and SHA-256 hashes of the ZIP and chase master. The ZIP uses the selected Git tree for the committed source, media and evidence. The external CARLA runtime, virtual environments, local experiments and private Waymo records stay outside the ZIP.

Build v0.7 from a full Git checkout so the verifier can compare the trial source archives to their recorded evaluator commits. The registered amendment covers the matched `slow` and `blocked` reruns. The other trials use the original evaluator archive.

If optional City Progress raw sensor sweeps would put the v0.6 Windows ZIP over GitHub’s limit, the build adds `ADO/results/city-progress/EXTERNAL-SWEEPS.md`. That index lists the affected raw sweep archives with their download URLs pinned to the release commit, byte lengths, and SHA-256 hashes. The Windows ZIP includes the films, controller logs, and synchronized replay needed to review the drives locally.

The v0.7 Windows ZIP keeps the historical showcase films, controller logs, and synchronized replays. If the measured ZIP would exceed 2 GiB, the build moves optional v0.6 raw sensor sweeps and then exploratory trial clips into the external research index at `ADO/results/city-progress/EXTERNAL-RESEARCH.md`. Each record gives a download URL pinned to the published v0.6 commit, the byte length, and a SHA-256 hash. Building the v0.7 ZIP requires that published commit in the local Git history. [Archived City Progress trials](https://github.com/yafi-s/ADO/tree/51b1cbb25837eb43693b775ccb14e68bffc4084b/results/city-progress)

## Reproduce the City Life range study

`ADO-City-Life-Range-Sweeps.zip` is a separate download for reproducing the offline perception study. Its `raw` directory contains the original perception journal, road geometry and scored telemetry. `sweeps` contains one native lidar and radar packet per recorded frame, plus the original front-horizon crop on acquisition frames. `manifest.json` lists frame coverage and the byte length and SHA-256 of every evidence file. `reference/object-memory-before.mjs` preserves the baseline renderer from its pinned Git commit. [City Life range study download](https://github.com/yafi-s/ADO/releases/download/v0.5.0/ADO-City-Life-Range-Sweeps.zip)

From the extracted v0.5 Windows package or the tagged ADO source, use Python 3.12 with the packages in `requirements.txt` and Node.js for the display score. These commands verify the archive and reproduce the full-range tracker, the 60 m single-scale comparison and a sampled check of the native front-horizon boxes. The offline study uses the recorded sensor packets and can run without CARLA. [Scoring method and complete results](../results/distant-awareness/README.md)

```powershell
$asset = 'C:\Downloads\ADO-City-Life-Range-Sweeps.zip'
$study = 'C:\ADO-City-Life-Range-Sweeps'
$work = 'C:\ADO-City-Life-Range-Results'
Get-FileHash $asset -Algorithm SHA256
py -3.12 tools/package_city_range_study.py verify $asset
Expand-Archive -LiteralPath $asset -DestinationPath $study
New-Item -ItemType Directory -Path $work
$env:OPENBLAS_NUM_THREADS = '1'
py -3.12 tools/ablate_distant_range.py "$study/raw" "$study/sweeps" "$work/native-verification" --verification-only
py -3.12 tools/ablate_distant_range.py "$study/raw" "$study/sweeps" "$work/sixty-metre-single-scale"
py -3.12 tools/project_distant_reference.py "$study/raw" "$work/registration.json"
node tools/score_distant_awareness.mjs "$work/sixty-metre-single-scale" "$work/before.json" --memory-file "$study/reference/object-memory-before.mjs" --projection "$work/registration.json"
node tools/score_distant_awareness.mjs "$study/raw" "$work/after.json" --projection "$work/registration.json"
py -3.12 tools/replay_distant_optics.py "$study/raw" "$work/crop-check.json" --native-sweeps "$study/sweeps" --stride 128
```

To build the optional asset, pass the sealed capture directory, its matching sweep directory and a new output path. After the Windows ZIP and chase master have passed their own checks, attach the range asset to add its checksum to `SHA256SUMS.txt`. The attach step checks that both ZIPs contain evidence from the same City Life recording.

```powershell
$release = '.local/release/v0.5.0'
$studyZip = "$release/ADO-City-Life-Range-Sweeps.zip"
py -3.12 tools/package_city_range_study.py build 'C:\Path\To\SealedCapture' 'C:\Path\To\NativeSweeps' $studyZip
py -3.12 tools/package_city_range_study.py attach $studyZip $release
```
