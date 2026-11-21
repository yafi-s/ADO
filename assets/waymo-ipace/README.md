# Waymo fifth-generation Jaguar I-PACE

An independently modeled exterior of the electric Jaguar I-PACE with the fifth-generation Waymo Driver. The original geometry includes the pressed body, glasshouse, hatch spoiler, flush handles, mirrors, wheel arches, forked alloy spokes, brakes, lamps and roof carrier. Six reference frames identify visible sensor locations. The cabin has simple seat and instrument silhouettes.

![Three-quarter studio render](studio.png)

## Files

- **waymo-ipace.blend** contains the individual editable parts, wheel skeleton, photographic lighting and camera.
- **waymo-ipace.glb** contains five rigidly skinned meshes, the wheel skeleton, six sensor anchors and self-contained materials. Use it for the complete model in a glTF viewer or DCC.
- **waymo-ipace.fbx** contains the chassis, four wheels and five-bone skeleton for Unreal/CARLA vehicle asset authoring.
- **studio.png** is the CPU Cycles render of the native model.
- **build.py** regenerates the model and exchanges in Blender 5.2. It uses the repository's mesh, material, interpolation and studio helpers in [build_vehicle.py](../robotaxi/vehicle/build_vehicle.py). The body, closures, glass, wheels, lighting and sensor forms are defined here. **verify.py** reopens all three model files, checks bounds, skin weights and wheel centers and rotates every imported wheel independently.

## Scale and frame

The origin is on the ground midway between the axles. Native Blender coordinates use metres, +X forward, +Y to the vehicle's left, and +Z up. A CARLA vehicle-space point with the same origin uses (x, −y, z) in metres. A native Blender point (x, y, z) maps to (x, z, −y) in glTF's Y-up convention. Importers such as Blender apply that last conversion automatically.

Jaguar lists a body length of **4.682 m**, a **2.990 m** wheelbase, a width of **2.139 m** with mirrors extended, and a stock height with antenna of **1.566 m**. The model's tires have a rolling radius of **0.365 m**. The modeled sensor-equipped roof reaches **2.039 m**. The independently reimported vehicle measures **4.7455 × 2.1390 × 2.0390 m** including the central front grille instrument and rear plate. The GLB contains 245,706 triangles across five meshes and 25 materials. Track and wheelbase follow Jaguar's technical data. Sensor and trim dimensions were estimated from public photos.

| Bone | Blender wheel centre in m, X / Y / Z |
| --- | --- |
| Vehicle_Base | 0 / 0 / 0 |
| Wheel_Front_Left | 1.495 / 0.8205 / 0.365 |
| Wheel_Front_Right | 1.495 / −0.8205 / 0.365 |
| Wheel_Back_Left | −1.495 / 0.8305 / 0.365 |
| Wheel_Back_Right | −1.495 / −0.8305 / 0.365 |

The body, sensor package and brake calipers follow Vehicle_Base. Each tire, rim, rotor and hub follows its wheel. Reading the wheel mesh after rigid skin deformation preserves the axle pivot even though the five runtime meshes share the vehicle origin.

| Native/GLB empty | Native X / Y / Z in m | Forward direction |
| --- | --- | --- |
| sensor_origin_roof_lidar | −0.176 / 0 / 1.958 | +X datum for 360° head |
| sensor_origin_roof_camera_front | 0.494 / 0 / 1.656 | +X |
| sensor_origin_perimeter_front_left | 1.670 / 1.006 / 0.974 | +Y |
| sensor_origin_perimeter_front_right | 1.670 / −1.006 / 0.974 | −Y |
| sensor_origin_perimeter_rear_left | −2.017 / 1.006 / 0.826 | +Y |
| sensor_origin_perimeter_rear_right | −2.017 / −1.006 / 0.826 | −Y |

## Use in CARLA

Import the FBX as a skeletal mesh in the Unreal Editor used for the target CARLA release. Check the 299 cm wheelbase and forward direction in the importer. Assign the four named wheel bones to the target vehicle's wheel/animation setup, set wheel radius to 36.5 cm, author the chassis collision and physics configuration, and register/cook the vehicle blueprint for that CARLA build. Import alone does not create a spawnable actor. The six empty frames are visual approximations for placing simulator sensors; this package does not supply sensor calibration or vehicle dynamics.

## References and licensing

- Waymo's [media resources](https://waymo.com/media-resources/) include the fifth-generation I-PACE in side, head-on and front-left views and a roof-sensor detail. Those views informed the long black carrier band, pearl fairing, camera crown, illuminated upper display, blue collar, four perimeter pods, dark split-spoke wheels and door lettering.
- Waymo describes the [fifth-generation Driver](https://waymo.com/blog/2020/03/introducing-5th-generation-waymo-driver/) and its [industrial design](https://waymo.com/blog/2020/03/designing-5th-generation-waymo-driver/), including the roof-mounted 360° lidar and four perimeter lidars.
- Jaguar's [I-PACE dimensions sheet](https://www.jaguar.com/content/dam/jdx/pdfs/uk/wltp/Jaguar-I-PACE-TD-Insert-1X5902400000GBEN01P.pdf) gives exterior size, track and wheelbase.
- The independently authored geometry, materials and build scripts are provided under the repository's [MIT license](../../LICENSE). The package contains original meshes and procedural materials. Waymo, Jaguar and I-PACE are trademarks of their respective owners. Manufacturer reference photography is linked and is not distributed with the asset. This is an unofficial visual reconstruction.

## Regeneration

From the repository root with Blender 5.2 on PATH, run:

```powershell
blender -b -t 2 --python-exit-code 1 --python assets/waymo-ipace/build.py -- --render
blender -b -t 2 --python-exit-code 1 --python assets/waymo-ipace/verify.py
```

