# Experimental lane-change comfort profile

The offline profile rejected the first cone pass in the continuous Market section. Leave it disabled. The six original invitations and every sampled later source-lane start failed the slower-path rear-clearance check. A paired simulator run was not performed.

The candidate measures the curvature of a quintic blend between the two native lane centerlines, checks the ego's recent yaw and traveled distance, and previews the return behind measured traffic. It uses the checkpoint's 1.729 m/s² lateral budget, 2.591 m/s² recovery coefficient and fitted stopping curve. The frozen controller retains authority over stopping, signaling, steering and rejoin approval. The candidate can veto a new invitation and lower a tactical speed ceiling. A failed forecast after commitment disables optional slowing.

## Recorded observations and projected margins

- The release committed at frame 13911. It forecast a 6.125 s occupation and 17.13 m of spare rear clearance. The first native body-vacated flag arrived 12.8 s later, at frame 14039.
- With the recorded ego motion and immediate hypothetical rear acceleration of 3.6 m/s² up to 13 m/s, the release's spacing margin first fell below zero at frame 13988. The minimum was −54.10 m. In the native recording the rear first exceeded 0.5 m/s four seconds after commitment. Its peak half-second speed was 11.43 m/s and the minimum center-to-center station gap was 31.52 m.
- The reference tapes showed about 3.92 m/s² peak rear acceleration from overlapping half-second speed estimates. The candidate therefore uses a 4.25 m/s² stress acceleration and preserves an initially higher measured speed. For the recorded invitation it predicts 13.25 s to clear, with a −73.18 m margin even if the rear begins at rest. These are conditional stress results. The assumed acceleration and speed limits are not universal physical bounds.
- On the frozen pass trajectory at frames 13924–13937, the candidate would suggest ceilings of 15.04–21.90 km/h versus released requests of 31.21–39.29 km/h. The entry veto prevents interpreting those numbers as measured comfort or progress.
- The optimistic delayed-start sweep kept the 31.729 m rear witness stationary until each potential start and assumed a free target and 60 m of source visibility. The tested station grid ran from the original commit to 2.6 m before the first cone surface and tested 0–11 m/s. The best margin was −9.53 m. The native fixture activates the rear at road station 81, after the static cones at 70–73, so waiting behind the cones cannot cause the rear to depart on its schedule. The finite grid is documented in the JSON.

## Reproduction

From the project root, run the offline audit with the local CARLA Python wheel. It parses the native OpenDRIVE map without connecting to a server.

```powershell
.\.venv-city\Scripts\python.exe -m tools.maneuver_comfort_candidate audit
```

The optional paired native command is retained for reproduction. It requires the matched section runner and a separately coordinated server on port 3718. The clean worktree contains the pinned release source. The candidate command snapshots the override alongside that source before simulator entry.

```powershell
Set-Location "$env:LOCALAPPDATA\ADO\city-maneuvers\pin-9bd6dafa"
& 'A:\Waymo FSD Omniverse\.venv-city\Scripts\python.exe' 'A:\Waymo FSD Omniverse\tools\maneuver_comfort_candidate.py' native comfort-section01
```

The section uses seed 212, market destination, ClearNoon, crossing delay 0.5 s, and the continuous fixture from 278 to 625 m. Its match to the existing integrated capture is the common prefix. The integrated capture continued to 906 m. The native runner and archive receipt were checked up to the simulator boundary with a substituted simulator entry. No native comfort trial was started.
