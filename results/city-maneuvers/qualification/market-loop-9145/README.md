# Market Loop native qualification

These separate captures test the same 1,719.890 m route with the moving rear vehicle physically driving from road 69/+2, through connectors 1960 and 101, onto road 67/+2. The ego starts the tail tests at route station 1,210 m. Both tails finished with a real moving-source pass, legal lane choice, complete body return and zero ego collisions, red-light violations or missed yields. [Registered feeder](../market-loop-upstream-feeder-registration.json)

| Schedule | Occupied gap | Applied yield | Minimum ego speed during occupancy | Slow-car passing clearance | Closest moving passing-lane vehicle | Whole-body return |
|---|---:|---:|---:|---:|---:|---:|
| [tail-continuous film](tail-continuous/front.mp4) | 2.5 s | 2.5 s | 0.944 m/s | 1.662 m | 1.537 m | 1695.791 m |
| [tail-later-gap film](tail-later-gap/front.mp4) | 2.4 s | 2.3 s | 0.000 m/s | 1.662 m | 1.538 m | 1702.609 m |

The frozen driver completed the first in-lane cone pass and returned at 104.9 s in the separate full-origin prefix. The original stationary Tesla was physically exposed at zero speed, restarted at 124.3 s and left the route at 139.1 s. The ego stayed in the adjacent lane and stopped at 641.502 m with only 7.435 m remaining in that surveyed corridor. The prefix therefore failed. The same parallel background car that previously occupied the left boulevard lane traveled 281.590 m after the real signal turned green and stopped 3.877 m before the red side-street line. [Prefix film](prefix-continuous/front.mp4) · [Prefix grade](prefix-continuous/grade.json) · [Parallel traffic](prefix-continuous/parallel-flow.json)

The first native arrival snapshot found the three eventual tail vehicles 195.315–305.262 m from the stationary warm-up ego, with all 90 site-to-camera rays blocked by buildings and a minimum 1.534 m body gap between nearby actors. [Native commissioning](commission-continuous/commission.json) · [Native tail source](tail-continuous/protocol.json)
