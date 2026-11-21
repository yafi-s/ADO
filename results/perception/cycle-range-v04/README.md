# Native bicycle and rider range

At a second Town03 site, the updated tracker placed a bicycle within 1.5 m of the moving cyclist on **19 of 60 frames**, up from **5 of 60** with the v0.3 tracker. Starting the 5 Hz camera one 10 Hz sweep later gave **20 of 60**, up from **8 of 60**. The two camera phases replay the same drive geometry and range captures. They are sensitivity checks on the short sequences. [Even-phase results and measured points](phase-even.json) · [Odd-phase results and measured points](phase-odd.json)

## Completed High city drive, seed 212

The cyclist was within 35 m on 239 of the 3,354 recorded simulator frames. The tracker located the cyclist under one ID for 170 consecutive frames. It reported a bicycle on 32, a person on 138, and neither on 69. All 32 bicycle labels coincided with the moving actor. Sixteen synchronized front-camera exposures had fresh shared lidar range, with 11–25 body returns and front or front-corner radar support. The other 16 labels carried that association across the next sweep. **Zero exposures established bicycle-exclusive range on this trip.** Median center error was 0.081 m on fresh shared measurements, with a maximum of 0.256 m. [Completed-drive score](high212.json) · [Per-frame reference, range and rider evidence](high212-frames.json.gz)

The final rider display drew the cyclist as a bicycle on **78 of 239 nearby frames**, including **78 of 231 moving frames**. The native tracker supplied 32 of those class labels. The display retained the classification on the same source ID for 46 additional frames. Median displayed bicycle center error was 0.157 m and the maximum was 0.354 m. The display still showed only a person on 99 nearby frames, and neither a person nor a bicycle on 62. The cyclist's displayed identity remained continuous from frame 4998 through 5174.

On the identical journal, the earlier display drew both a person and a bicycle with source ID 172 on 71 frames. Its closest drawing switched identity ten times and maximum bicycle position error reached 1.671 m. The corrected display used a single ID, with zero duplicate drawings and zero identity switches through the cyclist encounter. [Initial display score](high212-initial-display.json) · [Initial display frame evidence](high212-initial-display-frames.json.gz)

The first crossing walker produced a separate false ordinary bicycle track on frames 2342–2345, from one rear-left camera measurement followed by three coasted frames. The rider display suppressed it. No bicycle drawing fell within 3 m of a reference pedestrian on the completed drive. Actor references entered only during scoring. Positions are compared in the ego frame with a 3 m match radius and a 35 m cyclist reference range. Moving frames require over 0.3 m/s displacement-derived reference speed. The scorer checks the complete journal's hash and frame bounds against the capture manifest. [Whole-drive rider evaluation](../../urban-pace/perception.json) · [Encounter scorer](../../../tools/score_cycle_encounter.py)

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
