# ADO for Windows

Download the ZIP and `SHA256SUMS.txt` from the same release. With both files in the same folder, run the following command in PowerShell and compare the full hash with the value in `SHA256SUMS.txt`.

```powershell
Get-FileHash .\ADO-v0.1.0-Windows.zip -Algorithm SHA256
```

Extract the entire ZIP to a writable folder and double-click `ADO\Start-ADO.cmd`. The first launch installs the official CARLA 1.0 Windows server, Town01_Opt map content, the matching Python client, and the project's Python dependencies. The two simulator downloads total about 20.8 GB. Allow around 50 GB of free space on the extraction drive for the downloads and unpacked simulator. If a download is interrupted, run `Start-ADO.cmd` again to resume. After setup, use `Start-ADO.cmd` or `Watch-Live.cmd` to open the live driving console. Press Ctrl+C in the launcher window to stop it.

Use 64-bit Windows, 64-bit Python 3.12, and a GPU with Direct3D 12 support and current drivers. The project tested its five-camera profile on an RTX 3070 with 8 GB of VRAM. A compatible NVIDIA card can use DLSS on auxiliary camera views. An internet connection is required for first-time setup. If you already have the official CARLA 1.0 Windows distribution, set `CARLA_ROOT` to its extracted folder before launching from the same terminal. Setup can add Town01_Opt to that installation.

The ZIP includes trained checkpoints, the NanoDet detector, source code, Blender vehicle assets, synchronized recordings, and evaluation logs. Open the MP4 files in `ADO\web\media` to watch without CARLA, or use the 3D replay viewer on the [project site](https://yafi-s.github.io/ADO/).

## Build the release ZIP

Install Git and Python 3.12 or later. Put `ffprobe` from FFmpeg on PATH for the video checks. Commit the evaluated manifest, summaries, compressed traces, recordings, and generated Builds & logs index with the code and models. Build from the exact release commit or from the tag after it has been created.

```powershell
py -3.12 tools/build_windows_release.py --ref v0.1.0 --tag v0.1.0 --output .local/release
```

Upload `ADO-v0.1.0-Windows.zip` and `SHA256SUMS.txt` to the GitHub release for `v0.1.0`. The local `release-metadata.json` records the archived commit, ZIP size, and SHA-256. The build reads the selected Git tree, so only committed project files in the allowed folders are eligible for packaging. The external CARLA runtime, virtual environments, and local experiment workspaces stay outside the ZIP.
