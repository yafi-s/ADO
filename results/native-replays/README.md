# Native CARLA replays

The two `.rec` files come from CARLA 1.0's built-in additional-data recorder. A second client subscribed to the existing evaluation server and requested recording. The scoring client retained control of ticks and vehicle actuators.

- **103-trip-fusion** contains the held-out Market trip. The scored interval spans original server frames 59,369–59,726.
- **104-pedestrian-fusion** contains the next held-out pedestrian crossing. The scored interval spans original server frames 59,736–59,933.

The companion JSON files identify the server and map, the observation interval around each drive, the binary size, and its SHA-256 digest. Each recording includes setup or teardown ticks outside its scored interval. The corresponding entries in the [evaluation traces](../traces.jsonl.gz) contain the original per-tick control commands. A native recording can be opened with CARLA's `client.replay_file` API on Town01_Opt. Sensor images are generated again by the playback renderer. [CARLA recorder reference](https://github.com/carla-simulator/carla/blob/1.0/Docs/core_recorder.md)
