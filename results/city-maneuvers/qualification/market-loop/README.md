# Return-carriageway traffic commissioning

The Market Loop preserves the original 1,416.438 m route through the Market endpoint and then traverses three more junctions to reach road 67 in the opposite direction. The return section uses native lane +3 and the legal adjacent +2 passing lane. [Registered route and source](../market-loop-registration.json)

The original pair on road 69 physically continued into the later boulevard, travelling 308.789 m and 290.711 m. Three incoming vehicles entered only after the original convoy had cleared the sites, at 190.425–240.788 m from the ego. All 90 building-occlusion ray checks passed. The closest body separation during arrival was 1.534 m. [Native commissioning and geometry](commission/commission.json)

With the first, shorter feeder, both tails overtook the moving sedan and returned to the original source lane. In the nominal cut the rear car passed the ego center on connector 101 at 52.0 s. It was already 35.279 m ahead when the whole ego car entered road 67/+3. ADO made a clear-lane overtake with 1.662 m closest lead clearance, but this run did not expose the required occupied gap during a legal lane choice. [Nominal film](tail-continuous/front.mp4) · [Nominal grade](tail-continuous/grade.json)

The registered later arrival produced 12 actual yield frames while the passing gap was occupied. The ego kept rolling at a minimum of 1.168 m/s, overtook the sedan while it was moving at 1.572 m/s, and fully rejoined at 1,696.779 m. The closest moving passing-lane vehicle stayed 1.538 m away. [Later-gap film](tail-later-gap/front.mp4) · [Later-gap grade](tail-later-gap/grade.json)

Both tests started at route station 1,300 m after a stationary-ego physical warm-up for the original traffic. Their native camera, lidar, radar, source, and model records are available in the individual case directories. The additional moving-rear feeder and signal-released parallel-flow candidate has its own registration. [Upstream feeder](../market-loop-upstream-feeder-registration.json) · [Independent parallel flow](../market-loop-parallel-flow-registration.json)
