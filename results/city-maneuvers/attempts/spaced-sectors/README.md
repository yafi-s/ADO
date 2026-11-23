# Signal-led High sectors with the original sedan moved downstream

These two registered runs used the same 1,416 m Market geometry up to their test window and kept the original held sedan on road 67 at native s275 instead of s238. It stayed in its lane with its original departure rules. The control checkpoint and passing planner matched the original High recording. [Registration](../../qualification/spaced-registration.json)

| Traffic schedule | End station | Moving car speed alongside | Minimum body gap to moving lead | Minimum moving passing-lane clearance | Gap to original stationary sedan |
|---|---:|---:|---:|---:|---:|
| [continuous](continuous/summary.json) | 581.841 m | 1.575 m/s | 1.707 m | 21.547 m | 26.899 m |
| [later-gap](later-gap/summary.json) | 582.523 m | 1.904 m/s | 1.710 m | 20.934 m | 26.216 m |

Each remained in the passing lane between the cone group and slow sedan. The ego subsequently returned and stopped in its source lane too far from the held sedan to trigger its original departure. Neither run completed the registered separate first return, second yield and downstream-sedan sequence. The later-gap run also failed the pre-existing roadside-board lane/clearance grade while returning. The boards remained fixed and their recorded minimum horizontal clearance was 2.311 m. [Nominal forward film](continuous/front.mp4) · [Later-gap forward film](later-gap/front.mp4)

Each folder includes its trace, native participant reference, observations, checkpoint and source archive.
