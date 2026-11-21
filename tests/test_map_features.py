"""The committed town features are what the road description says, and fit the town export."""
import json
import os
from pathlib import Path
import unittest

try:
    from crosswalk import map_features
except ImportError:     # the simulator client is not installed
    map_features = None

ROOT = Path(__file__).resolve().parents[1]
TOWN01_MAP = Path(os.environ.get("CARLA_TOWN01_XODR", ".local/Town01_Opt-1.0.xodr"))
COMMITTED = ROOT / "web/media/map-features/Town01_Opt.json"


@unittest.skipUnless(map_features and TOWN01_MAP.is_file(), "needs the CARLA client and the Town01 road description")
class MapFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.features = map_features.features("Town01_Opt", TOWN01_MAP.read_text(encoding="utf-8"))

    def test_committed_file_is_what_the_road_description_gives(self):
        self.assertEqual(json.loads(COMMITTED.read_text(encoding="utf-8")), json.loads(json.dumps(self.features)))

    def test_every_exported_lane_is_described(self):
        town = json.loads((ROOT / "web/media/town.json").read_text(encoding="utf-8"))
        described = {(tuple(lane["a"]), tuple(lane["b"])) for lane in self.features["lanes"]}
        missing = [lane for lane in town["roads"] if (tuple(lane[0][:2]), tuple(lane[-1][:2])) not in described]
        self.assertEqual(missing, [])
        self.assertEqual(len(town["roads"]), len(self.features["lanes"]))

    def test_lane_direction_and_centre_line(self):
        # The first road runs along y = 0. Traffic keeps to the right of the yellow centre line,
        # and CARLA's y axis points to the right of its x axis.
        west, east = self.features["lanes"][:2]
        self.assertTrue(west["forward"] and west["a"][0] > west["b"][0] and west["a"][1] < 0)
        self.assertTrue(not east["forward"] and east["a"][1] > 0)
        yellow = {"line": "broken", "color": "yellow"}
        self.assertEqual((west["minus"], west["plus"]), (yellow, None))
        self.assertEqual((east["plus"], east["minus"]), (yellow, None))

    def test_signals_stand_beside_the_traffic_they_govern(self):
        signals = self.features["signals"]
        self.assertEqual(len(signals), 36)
        self.assertEqual(len({signal["id"] for signal in signals}), 36)
        by_id = {signal["id"]: signal for signal in signals}
        # Westbound traffic on the first road meets signal 362 on its own side of the road.
        self.assertEqual(abs(by_id[362]["governs_yaw"]), 180.)
        self.assertLess(by_id[362]["post"][1], -2.)


if __name__ == "__main__":
    unittest.main()
