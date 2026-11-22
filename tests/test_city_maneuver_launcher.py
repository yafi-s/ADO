from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from crosswalk import city_drive


class CityManeuverLauncherTests(unittest.TestCase):
    def test_maneuver_checkpoint_enables_native_rig_and_accepts_an_urban_ceiling(self):
        class ReachedDriver(Exception):
            pass

        model = Path(__file__).resolve().parents[1] / "models/city-maneuvers/candidate.npz"
        cases = [
            ([], "city-maneuvers", True, "market-loop"),
            (["--destination", "market"], "city-maneuvers", True, "market"),
            (["--urban-cap", "--no-tactical", "--no-surround", "--population-profile", "standard"], "standard", False, "market"),
        ]
        with TemporaryDirectory() as output:
            for flags, profile, enabled, destination in cases:
                argv = ["city-drive", "--model", str(model), "--output", output, *flags]
                with (
                    self.subTest(flags=flags),
                    patch("sys.argv", argv),
                    patch.object(city_drive, "simulator", return_value=nullcontext(object())),
                    patch.object(city_drive, "producer_info", return_value={}),
                    patch.object(city_drive, "episode", side_effect=ReachedDriver) as episode,
                ):
                    with self.assertRaises(ReachedDriver):
                        city_drive.main()
                    args, kwargs = episode.call_args
                    self.assertEqual(args[3], "ado-city-maneuvers-v1")
                    self.assertEqual(args[4].policy.allow_high_speed, enabled)
                    self.assertEqual(bool(kwargs.get("surround")), enabled)
                    self.assertEqual(bool(kwargs.get("tactical")), enabled)
                    self.assertEqual(kwargs["scene_factory"].keywords["population_profile"], profile)
                    self.assertEqual(kwargs["destination"], destination)
