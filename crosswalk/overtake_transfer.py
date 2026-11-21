"""Evaluate the frozen v0.4 pass and hold on a separate legal Town03 boulevard."""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import mkdtemp

import numpy as np

from .overtake import MODEL, PassingCorridor
from .overtake_trials import evaluate
from .runtime import producer_info, simulator
from .urban_map import _left_lane


SOURCE = (67, -2)
EGO_STATION = 210.
MODEL_PATH = Path(__file__).resolve().parents[1] / 'models/city-pace/candidate.npz'


def second_corridor(road_map):
    stations = range(180, 301)
    source = [road_map.get_waypoint_xodr(*SOURCE, float(s)) for s in stations]
    if any(wp is None or (wp.road_id, wp.lane_id) != SOURCE for wp in source):
        raise ValueError('Missing road 67 source lane sample')
    target = [_left_lane(wp) for wp in source]
    if any(wp.lane_id != -1 for wp in target):
        raise ValueError('Road 67 adjacent lane changed across the pass')
    corridor = PassingCorridor(source, target)
    arc, lateral = corridor.coordinates(corridor.source[:, :2])
    _, target_lateral = corridor.coordinates(corridor.target[:, :2])
    residual = max(float(np.max(abs(lateral))), float(np.max(abs(target_lateral + corridor.width))))
    if residual > .05 or np.any(np.diff(arc) < .95) or arc[-1] < 119.:
        raise ValueError('Transfer requires straight, parallel, continuous lane centers')
    geometry = dict(road_id=SOURCE[0], source_lane=SOURCE[1], target_lane=-1,
                    station_interval_m=[stations[0], stations[-1]], ego_station_m=EGO_STATION,
                    native_sample_count=len(source), center_separation_m=corridor.width,
                    minimum_lane_width_m=min(wp.lane_width for wp in [*source, *target]),
                    maximum_lateral_residual_m=residual, length_m=float(arc[-1]),
                    elevation_change_m=float(corridor.source[-1, 2] - corridor.source[0, 2]))
    return corridor, geometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=3268)
    parser.add_argument('--output', type=Path, default=Path('results/tactical-pass/v04-road67'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    folder = Path(mkdtemp(prefix='paired-', dir=args.output))
    print(folder, flush=True)
    with simulator(args.port, quality='Low', map_name='Town03_Opt') as world:
        road_map = world.get_map()
        corridor, geometry = second_corridor(road_map)
        sources = [Path(__file__).with_name(n) for n in ('overtake.py', 'overtake_trials.py', 'overtake_transfer.py', 'city_pace.py')]
        manifest = dict(producer=producer_info(world), quality='Low', geometry=geometry,
                        opendrive_sha256=hashlib.sha256(road_map.to_opendrive().encode()).hexdigest(),
                        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                        model_sha256=hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
                        tactical_sha256=hashlib.sha256(MODEL.read_bytes()).hexdigest(), cases=[])
        for case in ('clear', 'occupied', 'no-lead'):
            result = evaluate(world, case, MODEL_PATH, folder, record=True, source_lane=SOURCE,
                              ego_s=EGO_STATION, corridor=corridor)
            manifest['cases'].append(result)
            (folder/'summary.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf8')
    return all(r['passed'] for r in manifest['cases'])


if __name__ == '__main__':
    raise SystemExit(0 if main() else 1)
