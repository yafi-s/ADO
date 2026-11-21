# Native bicycle and rider range

At a second Town03 site, the updated tracker placed a bicycle within 1.5 m of the moving cyclist on **19 of 60 frames**, up from **5 of 60** with the v0.3 tracker. Starting the 5 Hz camera one 10 Hz sweep later gave **20 of 60**, up from **8 of 60**. The two camera phases replay the same drive geometry and range captures. They are sensitivity checks on the short sequences. [Even-phase results and measured points](phase-even.json) · [Odd-phase results and measured points](phase-odd.json)

## Measurements

The capture contains 180 synchronized CARLA 1.0 front-camera, full 32-channel lidar and front-radar exposures. Each of two sites has 15 frames each of a bicycle seen from the rear, at an oblique angle and from the side, a pedestrian alone, a moving bicycle beside a separate moving walker, and a stationary mounted bicycle. The camera approaches the actors on a scripted trajectory. The moving targets also follow scripted poses. Doppler comes from the native radar and is assigned by the existing unique geometric association. These controlled poses do not validate physical speed estimates. [Complete native range sweeps, detector outputs, six original images and scoring poses](native-range.zip)

The detector is the existing OpenCV Zoo YOLOX-S. Camera boxes must identify one plausible bicycle and rider. An upper-body lidar cluster and lower returns must occupy a compact depth interval. The image identities must persist across three observations and their ego-motion-corrected displacement must agree. A confirmed pair supplies one track and can retire an earlier nearby bicycle track with the exact same bicycle image identity. Each camera box carries the same track ID and its range source.

- **Shared rider/bicycle lidar** places the mounted road user at the measured body. The wheels can be entirely inside the rider box.
- **Bicycle-exclusive lidar** requires at least three lower returns outside the rider box plus a one-pixel margin, with physical spread and agreement with earlier bicycle-only returns after rider motion compensation. The output retains the actual supporting XYZ coordinates. A camera gap records evidence age and zero fresh points.
- A low horizontal strip from one laser ring cannot independently establish a bicycle. This removed 12 falsely located bicycle frames on the development site's standard phase, and eight across both sites on the alternate phase. Some came from a rising road surface well behind the image detection.

Across the four moving-bicycle configurations per site, the development site's correctly located bicycle count rose from **21 to 29 of 60** on even frames and **14 to 28 of 60** on odd frames. Duplicate drawings at the cyclist fell from **21 to 4** and **14 to 2**. At the second site they fell from **5 to 4** and **8 to 0**. These counts include coasting. The separate pedestrian beside the cyclist retained the same detection counts in both versions and camera phases. The rear view remained difficult. At the second site it supplied zero bicycle tracks on the standard phase and two on the alternate phase.

On the standard phase, 23 confirmed associations had fresh range and radar Doppler. Four exposures on the development site had 3–6 measured bicycle-exclusive returns. One exposure on the alternate phase had four. The second site supported shared range on its newly associated tracks. Every confirmed association across both phases had a center error below 0.30 m in these controlled poses.

## Parked bicycle and pedestrian controls

CARLA's drivable crossbike includes a seated rider. In the stationary native bicycle sequences, ego motion yielded **zero moving-rider associations**. Native walker-only sequences also produced zero mounted associations. A riderless control uses the original bicycle boxes and only the low native lidar returns outside the stationary seated person's padded box. Another control combines those samples with separately recorded native walker boxes and lidar at the same ego poses. These are counterfactual sensor combinations. The tracker receives neither actor labels nor scoring poses beyond its own ego pose.

Across 60 riderless control frames, eight camera exposures put a walker in plausible alignment with a bicycle. The tracker produced **zero false mounted associations**. Parked bicycles remained difficult to range when almost all their measured surface had been removed. With the nearby walker present at the second site, the standalone bicycle remained undetected and the pedestrian was measured in 13 of 15 frames. [Counterfactual control results](parked.json)

## Original High city film

Across 3,173 frames, the original tracker produced zero bicycle tracks and the saved film contained 22 explicit bicycle boxes. Reprojecting the published thinned lidar plus recorded track samples found one low return outside the padded rider box on aligned exposures. It occurred at frame 3449. Three exposures retained sufficient shared upper/lower support. The complete sweeps for this encounter were not retained, so the historical film cannot establish separately ranged bicycle structure. [Archived-film projection](archived-city.json)

## Reproduction

Extract the native archive into an empty directory. It includes all 180 unchanged full-precision range packets, original YOLOX detections for all 180 exposures, scorer-only actor reference positions and ego poses. The full native capture retained locally also contains every RGB exposure. The archive's SHA-256 is `1654e8bc114a86ca293ca4b315270fbb877420ce873822eabf041cb3957b50ea`.

From the repository root, use the CARLA city Python environment. The comparison loads the original tracker from commit `ae31697ea43aa9f773c7e73a5bcd3114b3547175`. The study defaults to the same alternating front-camera schedule as the live observer.

```text
python -m crosswalk.cycle_study EXTRACTED_DIRECTORY phase-even.json
python -m crosswalk.cycle_study EXTRACTED_DIRECTORY phase-odd.json --phase 1
python -m crosswalk.cycle_study EXTRACTED_DIRECTORY parked.json --parked-controls
python -m unittest tests.test_cycle_association
```

To repeat the actual acquisition, give the collector sole access to an idle synchronous Town03 server with a 0.1 s step. Use `python -m crosswalk.cycle_collect --port PORT --output NEW_DIRECTORY`. Running the study on a fresh capture performs YOLOX inference and caches the original boxes before scoring.
