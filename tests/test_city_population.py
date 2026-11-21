"""Verify native lane direction, sidewalk occupancy bounds and per-frame movement scoring."""
import gzip
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import carla

from crosswalk.city_population import CARS, WALKS, lane_segment, sidewalk_segment
from crosswalk.population_score import score_population


class CityPopulationTests(unittest.TestCase):
    def test_native_town03_population_geometry(self):
        root = Path(__file__).resolve().parents[1] / ".local" / "carla-1.0"
        configured = os.environ.get("CARLA_TOWN03_XODR")
        path = (Path(configured) if configured else
                next(root.glob("*/Packs/AdditionalMaps/Content/Maps/OpenDrive/Town03_Opt.xodr"), None))
        if path is None or not path.is_file():
            self.skipTest("Requires the local native Town03 OpenDRIVE installation")
        road_map = carla.Map("Town03_Opt", path.read_text())
        for road, lane, start, end, *_ in CARS:
            with self.subTest(road=road, lane=lane, start=start):
                path = lane_segment(road_map, road, lane, start, end, carla.LaneType.Driving)
                self.assertGreater(len(path), 8)
                for p, q in zip(path, path[1:]):
                    travel = q.transform.location - p.transform.location
                    self.assertGreater(p.transform.get_forward_vector().dot(travel), .5)
        for road, side, start, end in WALKS:
            with self.subTest(road=road, side=side, start=start):
                path = sidewalk_segment(road_map, road, side, start, end)
                self.assertGreater(len(path), 8)
                self.assertGreaterEqual(min(p.lane_width for p in path), 1.85)

    def test_counts_simultaneous_motion_excluding_discontinuities(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            records, ref = root / "frames.jsonl", root / "reference.jsonl.gz"
            with records.open("w", encoding="utf-8") as output:
                for f in range(8):
                    actors = [dict(type=kind, role="background", blueprint=kind, actor_id=i,
                                   x=8+i+(f*.2 if i < 2 else f*.1), y=2)
                              for i, kind in enumerate(("car", "car", "pedestrian", "pedestrian"))]
                    if f == 6:
                        actors[2]["x"] += 10  # teleport cannot qualify on this or the next frame
                    item = dict(frame=f+10, sim_time=f/10, progress_m=0, observer_map=dict(ego=[0,0,0], others=actors))
                    output.write(json.dumps(item)+"\n")
            score = score_population(records, ref)
            data = score["radii_m"]["50"]
            self.assertEqual((data["scored_frames"], data["both_frames"], data["two_of_each_frames"]), (3,3,1))
            self.assertEqual(data["peak"]["simultaneous_frame"], 15)
            with gzip.open(ref, "rt", encoding="utf-8") as stream:
                self.assertEqual([json.loads(row)["frame"] for row in stream], list(range(10,18)))


if __name__ == "__main__":
    unittest.main()
