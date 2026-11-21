# Signal control development

The tested fusion model completed all five development drives with zero collisions and zero red-light violations. CARLA ran at 0.1 simulated seconds per tick. Each red-to-green drive and its green-from-start control used the same seed, route and weather in the training district.

- **Seed 31** stopped for 27 red frames, held at least 2.88 m of front-bumper clearance, then resumed and completed 63.51 m. Its matched green-from-start drive completed 63.79 m and registered zero stopped frames after the first metre.
- **Seed 32** stopped for 12 red frames, held at least 2.54 m of clearance, then resumed and completed 63.62 m. Its green-from-start drive completed 64.00 m and registered zero stopped frames after the first metre.
- **Clear road, seed 31** completed 64.00 m. The expert engaged on zero frames, and every applied control matched the base model.

`results.json` records the model hashes, weather, route, sensor configuration, frame counts and scoring for each trial. The five compressed traces record every selected command and corresponding base command. The stopped-red count requires speed at or below 0.35 m/s after the first metre. Both red-cycle runs spent four frames at or below that speed immediately after the signal turned green.

## Recognition and scope

The classifier recognized 16/16 red and 15/16 green images from the two physical heads excluded from fitting. Its Brier score was 0.022. The additional seed captures repeated those same poles. One route was shorter than two requested parking distances, which produced two pairs at the same distance. The original seeds 31 and 32 informed a revision to the image features. These measurements describe development performance.

All accepted crops came from the front camera. Side-camera classification, full occlusion, yellow lamps and other maps remain unverified. A geometrically visible housing can still be occluded in the image. The separately scheduled final district evaluation has its own manifest and results.

## Reproduce the closed-loop checks

`development-base-fusion.npz` preserves the exact base model used in this diagnostic. Run the following from the project root using the CARLA 1.0 environment and a fresh output directory.

```sh
python -m crosswalk.lamp_study validate --base results/signal-development/development-base-fusion.npz --expert data/signals/lamp.npz --output out/signal-drive-reproduction
```

The command starts a rendering-capable server and closes it on completion. To use an idle graphical Town01_Opt server, append `--attach --port 2468`. The static image and actuator fitting inputs and offline steps are in the [signal lesson package](../../data/signals/README.md).
