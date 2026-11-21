# Everyday city driving study

The unit of evaluation is one closed-loop drive. The policy receives RGB images from three directions, range and velocity observations, ego speed and a mapped route. The stop line, signal state, actor positions and scenario phase belong to the simulator and evaluator. The learned driver has no API for consulting them.

## Locked evaluation

Seeds 101 through 110 each run ten multisensor drives. The suite includes a pedestrian crossing, the same crossing delayed by one second, a stopped vehicle, a braking lead vehicle, a pedestrian emerging behind a parked van, a red-to-green signal, an unprotected left turn with opposing traffic, an ordinary intersection turn, a clear street, and a two-intersection destination trip with a pedestrian encounter. The city trips alternate between Market and Station. The suite intentionally overrepresents interactions. The counts do not estimate how often these events occur in real traffic.

Two of those seeds repeat all ten drives with a camera-only learned policy. A supplemental run delays the first Market trip's pedestrian by one second. The primary denominator is 100, the paired ablation denominator is 20, and the supplemental trip is reported separately. Repeated seeds on the same path measure variation in encounters and driving conditions. They do not each represent a new street.

Training routes and evaluation routes use different Town01_Opt road and junction IDs. Training uses seeds 11–16. The training archive stores each demonstration’s mapped centerline in world metres with the corresponding seed. Each policy runs nine development cases on each of seeds 31 and 32 in the training district, for 36 development drives. Model revisions can use these outcomes. Abrupt lead-vehicle braking first appears in the locked evaluation. Save the model digest and test schedule before starting the held-out drives. Preserve the entirety of a test round, including failed drives and stalled cars.

## What to count

A drive ends when the car reaches the destination tolerance, makes contact, departs from the mapped route, crosses the controlled red stop line, or reaches the time limit. Record event exposure, completion, yield violations, contacts by object type, minimum clearance to the scripted hazard, distance and elapsed simulation time. Where a hazard remains out of the lane, the clearance field uses a 60 m sentinel. It is a scenario-specific measurement. Compare the two policies on matching encounters and seeds only.

Publish representative runs selected by case and seed before evaluation, together with telemetry and downloadable inputs. A viewer can change the destination, seed or pedestrian entry delay locally. The public console plays recorded evaluations, and the local console shows the currently running simulation.
