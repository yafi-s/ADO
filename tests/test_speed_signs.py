import gzip
from importlib.util import find_spec
import json
import os
from pathlib import Path
import unittest
from types import SimpleNamespace

from crosswalk.speed_signs import SignedRouteLimits

ROOT = Path(__file__).resolve().parents[1]
CARLA_ROOT = Path(os.environ.get("CARLA_ROOT") or ROOT/".local/carla-1.0")
MAP_FILE = CARLA_ROOT/"CarlaUnreal/Packs/AdditionalMaps/Content/Maps/OpenDrive/Town03_Opt.xodr"


class SignMap:
    def to_opendrive(self):
        return """<OpenDRIVE><road id="8"><objects>
          <object name="Speed_90" s="12" t="-5" orientation="-"/>
          <object name="Speed_30" s="89" t="-5" orientation="-"/>
          <object name="Speed_20" s="77" t="6" orientation="+"/>
        </objects></road><road id="17"><objects>
          <object name="Speed_10" s="25" t="-4" orientation="-"/>
        </objects></road></OpenDRIVE>"""

    def get_waypoint(self, location):
        return location


class SignedLimitTests(unittest.TestCase):
    def test_lane_and_direction_match(self):
        points = [SimpleNamespace(road_id=8, lane_id=-2, s=s) for s in (0.,50.,100.)]
        limits = SignedRouteLimits.from_map(SignMap(), points, (0.,50.,100.), 2.3)
        self.assertEqual(limits.zones, ((12.,90.), (89.,30.)))
        opposite = [SimpleNamespace(road_id=8, lane_id=1, s=s) for s in (100.,50.,0.)]
        self.assertEqual(SignedRouteLimits.from_map(SignMap(), opposite, (0.,50.,100.), 2.3).zones, ((23.,20.),))
        self.assertEqual(SignedRouteLimits.from_map(SignMap(), list(reversed(points)), (0.,50.,100.), 2.3).zones, ())

    def test_lower_zone_only_constrains_its_approach(self):
        limits = SignedRouteLimits([(100.,30.)], 2.4)
        self.assertIsNone(limits.approach(40., 11., 90.))
        self.assertIsNone(limits.approach(90., 8., 30.))
        early = limits.approach(76., 10.5, 60.)
        late = limits.approach(95., 9., 60.)
        self.assertEqual(early[1:], (30.,24.))
        self.assertLess(late[0], early[0])
        self.assertLess(late[0]*3.6,30.)
        self.assertIsNone(limits.approach(105.,9.,60.))

    @unittest.skipUnless(MAP_FILE.is_file() and find_spec("carla"), "Native CARLA client and Town03 map required")
    def test_signed_preview_preserves_archived_211_requests(self):
        import carla
        import numpy as np
        from crosswalk.city_pace import CityPaceModel
        from crosswalk.control import Navigation
        from crosswalk.urban_map import urban_route
        native = carla.Map("Town03_Opt", MAP_FILE.read_text())
        route = [w.transform.location for w in urban_route(native, 211, "market").points]
        zones = SignedRouteLimits.from_map(native, route, Navigation(route).arc, 2.5)
        self.assertEqual([entry[1] for entry in zones.zones], [90.,60.,30.])
        policy = CityPaceModel.load(ROOT/"models/city-pace/candidate.npz")
        policy.new_observer()
        directory = ROOT/"results/city-pace/seed-211"
        with np.load(directory/"observations.npz") as data:
            with gzip.open(directory/"trace.jsonl.gz","rt") as recorded:
                for x, expected, line in zip(data["features"], data["speed_request_mps"], recorded, strict=True):
                    row = json.loads(line)
                    policy.posted_limit_kmh = row["posted_limit_kmh"]
                    approach = zones.approach(row["progress_m"], x[6], policy.posted_limit_kmh)
                    policy.signed_speed_limit = approach[0] if approach else None
                    policy.predict(x)
                    self.assertAlmostEqual(policy.last_speed_request, expected, places=5)


if __name__ == "__main__":
    unittest.main()
