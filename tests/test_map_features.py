"""The committed town features are what each road description says, and fit the town exports."""
import json
import os
from pathlib import Path
import unittest

try:
    from crosswalk import map_features
except ImportError:     # the simulator client is not installed
    map_features = None

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "web/media/map-features"
# Each town's road description, and a town export recorded in it.
TOWNS = {
    "Town01_Opt": (Path(os.environ.get("CARLA_TOWN01_XODR", ".local/Town01_Opt-1.0.xodr")), ROOT / "web/media/town.json"),
    "Town03_Opt": (Path(os.environ.get("CARLA_TOWN03_XODR", ".local/Town03_Opt-1.0.xodr")),
                   ROOT / "web/media/long-city-drive/town.json"),
}


def described(town):
    xodr, export = TOWNS[town]
    if map_features is None or not xodr.is_file():
        raise unittest.SkipTest(f"needs the CARLA client and the {town} road description")
    return map_features.features(town, xodr.read_text(encoding="utf-8")), json.loads(export.read_text(encoding="utf-8"))


class MapFeatureTests(unittest.TestCase):
    def test_committed_files_are_what_the_road_descriptions_give(self):
        for town in TOWNS:
            with self.subTest(town):
                features, _ = described(town)
                committed = json.loads((FEATURES / f"{town}.json").read_text(encoding="utf-8"))
                self.assertEqual(committed, json.loads(json.dumps(features)))

    def test_every_exported_lane_is_described(self):
        for town in TOWNS:
            with self.subTest(town):
                features, export = described(town)
                self.assertEqual(export["producer"]["map"], town)
                lanes = {(tuple(lane["a"]), tuple(lane["b"])) for lane in features["lanes"]}
                missing = [lane for lane in export["roads"] if (tuple(lane[0][:2]), tuple(lane[-1][:2])) not in lanes]
                self.assertEqual(missing, [])
                self.assertEqual(len(export["roads"]), len(features["lanes"]))

    def test_lane_direction_and_centre_line(self):
        # Town01's first road runs along y = 0. Traffic keeps to the right of the yellow centre
        # line, and CARLA's y axis points to the right of its x axis.
        features, _ = described("Town01_Opt")
        west, east = features["lanes"][:2]
        self.assertTrue(west["forward"] and west["a"][0] > west["b"][0] and west["a"][1] < 0)
        self.assertTrue(not east["forward"] and east["a"][1] > 0)
        yellow = {"line": "broken", "color": "yellow"}
        self.assertEqual((west["minus"], west["plus"]), (yellow, None))
        self.assertEqual((east["plus"], east["minus"]), (yellow, None))

    def test_signals_face_the_traffic_they_govern(self):
        features, _ = described("Town01_Opt")
        signals = {signal["id"]: signal for signal in features["signals"]}
        self.assertEqual(len(signals), 36)
        # Westbound traffic on the first road meets signal 362 on its own side of the road,
        # with the governed lane's centre a little under 3 m to the left of the post.
        self.assertEqual(abs(signals[362]["governs_yaw"]), 180.)
        self.assertLess(signals[362]["post"][1], -2.)
        self.assertTrue(all(2. < signal["reach"] < 3.2 for signal in signals.values()))

        # Town03's posts stand across wide junctions from the lanes they govern. Signal 2024
        # governs eastbound traffic from a mast that reaches 11 m over the road.
        features, _ = described("Town03_Opt")
        signals = {signal["id"]: signal for signal in features["signals"]}
        self.assertEqual(len(signals), 38)
        self.assertEqual(signals[2024]["governs_yaw"], 0.)
        self.assertAlmostEqual(signals[2024]["reach"], 11.2, delta=.3)
        self.assertTrue(all(signal["reach"] > 4. for signal in signals.values()))


if __name__ == "__main__":
    unittest.main()
