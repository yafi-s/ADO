# City driving in ADO

The Town03 Market route covers roughly 1.42 km and passes through 11 junctions. CARLA advances the car from its measured steering, throttle and brake inputs. Its cameras, roof lidar and forward radar observe the resulting world on the next simulation tick. A website recording can therefore be checked against what the sensors saw and what the driver commanded.

## The city route

- A pedestrian crosses the street, a second crossing starts late, and a third pedestrian emerges from behind a parked van. The occlusion case records the blocked sightline and the moment the pedestrian becomes visible.
- A stopped vehicle departs, a moving lead brakes, and a queued car waits at an actual red signal. The scorer records the lead's deceleration, release of the queue and the signal state when the ego crosses the line.
- The navigation route requests a lane change around a stationary Mitsubishi Fuso bus. The car has to steer across a broken line into the adjacent same-direction lane. Its actual lane IDs, lateral displacement and clearance alongside the bus determine the score. The experiment supplies the route choice in advance. It tests the learned driver's execution of the pass.
- A cyclist rides ahead and turns away. A six-cone work zone narrows the available space. The run log measures physical exposure and passage for each.
- An unprotected left turn crosses the path of an oncoming vehicle. Junction headings, route progress and time on open road are also recorded.

For each encounter, the log distinguishes the planned location, actual exposure, clearance, passage and any violation. An event beyond a drive's stopping point earns no credit. CARLA actor positions and signal states are available to the scenario scorer. The evaluated driver receives measured sensor features, ego motion and navigation geometry. Scripted NPC movement determines what traffic the car encounters.

## Run it locally

Run the project's Windows setup to install CARLA 1.0, its matching Python client and the AdditionalMaps pack. From the project root, this starts the learned driver and opens a live view of its cameras and measurements.

```powershell
.\.venv-city\Scripts\python.exe -m ado city-drive --serve --open-browser --seed 211
```

Each attempt writes its own summary and compressed frame log under `out/city-drives`. The lower-cost live profile is the default. To record a simultaneous 1080p chase camera and forward camera at High quality, run the film command with a fresh output path. This captures one ego car over one continuous simulation.

```powershell
.\.venv-city\Scripts\python.exe -m ado city-film --model models/behavior-v2/candidate.npz --map Town03_Opt --destination market --seed 211 --output out/my-city-film
```

## Training and evaluation

The Town01-trained controller uses 17,236 examples from 80 CARLA driving demonstrations. The public Colab recipe reproduces the saved model and training report byte-for-byte. Town03 is a separate city integration test. The model also has a controlled Town01 comparison using ten encounter types. The real Waymo research track has its own Colab training, validation and simulator transfer results. [CARLA training and Colab notebook](../data/behavior-v2/README.md) · [Matched comparison protocol](behavior-v2-protocol.md) · [Real Waymo model and transfer trial](../waymo_training/MOTION_TRANSFER.md)

CARLA supplies the roads, traffic actors, signals and vehicle physics. The experiment separates training streets, closed-loop evaluation and attributed failure modes following the concerns examined in the NoCrash study. The Waymo Open Motion Dataset supplies logged multi-agent trajectories for the separate forecasting experiment. [Dosovitskiy et al., CARLA](https://proceedings.mlr.press/v78/dosovitskiy17a.html) · [Codevilla et al., Exploring the Limitations of Behavior Cloning for Autonomous Driving](https://openaccess.thecvf.com/content_ICCV_2019/html/Codevilla_Exploring_the_Limitations_of_Behavior_Cloning_for_Autonomous_Driving_ICCV_2019_paper.html) · [Ettinger et al., Large Scale Interactive Motion Forecasting for Autonomous Driving](https://arxiv.org/abs/2104.10133) · [Software, sensor and asset credits](research.md)
