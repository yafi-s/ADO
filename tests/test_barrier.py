"""Native exposure regression and independent roadside-passage grading."""
from types import SimpleNamespace
import io
import unittest
from zipfile import ZipFile

import carla
import numpy as np

from crosswalk.barrier import detect
from crosswalk.barrier_scene import barrier_gap
from crosswalk.barrier_study import ARCHIVE, fit, read_rgb
from crosswalk.journey import Encounter, required_event_kinds
from crosswalk.tracking import Tracker


class BarrierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.archive = ZipFile(ARCHIVE)
        cls.addClassCleanup(cls.archive.close)

    def sample(self, name):
        rgb = read_rgb(self.archive.read(name + ".jpg"))
        with np.load(io.BytesIO(self.archive.read(name + ".npz"))) as scan:
            return rgb, scan["lidar"].astype(np.float32)

    def test_actual_positive_requires_both_rgb_pattern_and_compatible_depth(self):
        rgb, cloud = self.sample("holdout_2_barricade_13")
        empty = np.empty((0, 4))
        detections = detect(rgb, cloud, "front")
        self.assertEqual(len(detections), 1)
        tracker = Tracker(surround=True)
        track, = tracker.update(cloud, empty, {"front": detections}, (0, 0, 0), 0.)
        self.assertEqual(track["label"], "barrier")
        self.assertEqual(track["sources"], ["rgb", "lidar"])
        self.assertGreaterEqual(track["lidar_hits"], 3)
        self.assertLess(np.hypot(track["x"] - 13., track["y"] - 2.1), .65)
        self.assertEqual(detect(rgb, empty, "front"), [])
        self.assertEqual(detect(np.zeros_like(rgb), cloud, "front"), [])
        self.assertEqual(detect(rgb, cloud * 2, "front"), [])
        track_again, = tracker.update(cloud, empty, {"front": detections}, (0, 0, 0), .2)
        self.assertEqual(track["id"], track_again["id"])

    def test_board_breadth_needs_horizontal_returns(self):
        from crosswalk.barrier_posts import board_grade
        self.assertFalse(board_grade([[5.,1.,.4],[5.08,1.08,1.5]])["confirmed"])
        grade=board_grade([[5.,1.,.4],[5.,1.4,.4],[5.,1.3,.6]])
        self.assertTrue(grade["confirmed"])
        self.assertAlmostEqual(grade["measured_diameter_m"],.4)

    def test_connected_chain_publication_needs_decisive_or_repeated_exclusive_ranges(self):
        from crosswalk.barrier_posts import ChainHistory
        history=ChainHistory()
        sparse={"posts":[2,3],"span_hits":1}
        strong={"posts":[4,3],"span_hits":2}
        raw=dict(chain_ends_xy=[[5.,1.],[5.,2.44]], chain_support=sparse, range_supported=True, track_id=7)
        appearance=dict(confirmed=True,post_scores=[.8,.75])
        self.assertFalse(history.qualify(raw,appearance,(0,0,0),0.,sparse)["confirmed"])
        self.assertFalse(history.qualify(raw,appearance,(0,0,0),0.,sparse)["confirmed"])
        # A change in vehicle pose should leave the two stationary post sites aligned.
        later=dict(raw,chain_ends_xy=[[4.8,2.44],[4.8,1.]])
        self.assertTrue(history.qualify(later,appearance,(.2,0,0),.2,sparse)["repeat_supported"])
        self.assertFalse(history.qualify(raw,appearance,(0,0,0),1.,sparse)["confirmed"])
        self.assertFalse(ChainHistory().qualify(dict(raw,chain_support=strong,range_supported=False),
                                              appearance,(0,0,0),0.,strong)["confirmed"])
        self.assertTrue(ChainHistory().qualify(dict(raw,chain_support=strong),
                                              appearance,(0,0,0),0.,strong)["confirmed"])

    def test_actual_parked_car_cone_curb_and_planter_remain_distinct(self):
        for negative in ("clear", "cone", "parked_car", "planter"):
            with self.subTest(negative=negative):
                rgb, cloud = self.sample(f"holdout_2_{negative}_7")
                self.assertEqual(detect(rgb, cloud, "front"), [])

    def test_chain_requires_two_measured_posts_and_an_interior_return(self):
        from crosswalk.barrier_study import ROOT
        source = ROOT / "data/barrier-recognition/town03-chains.zip"
        with ZipFile(source) as archive:
            def observe(case):
                name = f"fit_0_{case}_5"
                rgb = read_rgb(archive.read(name + ".jpg"))
                with np.load(io.BytesIO(archive.read(name + ".npz"))) as packet:
                    cloud = packet["lidar"].astype(np.float32)
                return rgb, cloud, detect(rgb, cloud, "front")
            rgb, cloud, boxes = observe("chain")
            box, = boxes
            self.assertEqual(box["barrier_subtype"], "chain")
            self.assertTrue(box["post_appearance"]["confirmed"])
            self.assertGreaterEqual(min(box["post_appearance"]["post_scores"]), .34)
            ids = box["chain_return_ids"][1]
            self.assertTrue(ids)
            self.assertFalse(detect(rgb, np.delete(cloud, ids, axis=0), "front"))
            track, = Tracker(surround=True).update(cloud, np.empty((0,4)), {"front":boxes}, (0,0,0), 0.)
            self.assertEqual((track["label"], track["barrier_subtype"]), ("barrier", "chain"))
            self.assertGreaterEqual(track["chain_support"]["span_hits"], 1)
            self.assertGreaterEqual(min(track["chain_support"]["posts"]), 2)
            self.assertEqual(track["subtype_confidence"], box["score"])
            for case in ("unlinked", "warning", "cone", "raw_chain", "lone_post"):
                self.assertFalse(observe(case)[2], case)

    def test_fit_uses_two_annotated_training_exposures(self):
        from crosswalk.barrier import MODEL
        import json
        self.assertEqual(fit(self.archive), json.loads(MODEL.read_text()))

    def test_optional_course_requires_physically_passing_every_board(self):
        self.assertEqual(len(required_event_kinds("Town03_Opt")), 10)
        self.assertEqual(len(required_event_kinds("Town03_Opt", barriers=True)), 11)
        origin = carla.Transform()
        board = SimpleNamespace(is_alive=True, get_location=lambda: carla.Location(x=0., y=2.6),
                                bounding_box=carla.BoundingBox(carla.Location(), carla.Vector3D(.2, .6, .55)))
        ego = SimpleNamespace(pose=carla.Transform(carla.Location(x=-13.)),
                              bounding_box=carla.BoundingBox(carla.Location(), carla.Vector3D(2., 1., .8)))
        ego.get_transform = lambda: ego.pose
        lane = SimpleNamespace(road_id=67, lane_id=-2)
        scene = SimpleNamespace(_time=1., ego=ego, _map=SimpleNamespace(get_waypoint=lambda *a, **kw: lane))
        event = Encounter("barrier", 0, origin, 0., pass_station=5.,
                          barriers=[board], barrier_origins=[board.get_location()])
        barrier_gap(scene, event, -13.)
        self.assertIn("exposed_s", event.observed)
        barrier_gap(scene, event, 5.)
        self.assertNotIn("cleared_s", event.observed)
        ego.pose = carla.Transform()
        barrier_gap(scene, event, 0.)
        self.assertAlmostEqual(event.observed["minimum_barrier_clearance_m"], 1.)
        ego.pose = carla.Transform(carla.Location(x=6.))
        barrier_gap(scene, event, 6.)
        event.mark("passed_s", 2.)
        self.assertTrue(event.record()["handled"])
        # A fresh unsafe pass beside the same board must fail the independent score.
        ego.pose = carla.Transform(carla.Location(y=.9))
        barrier_gap(scene, event, 0.)
        self.assertTrue(event.violation)
        self.assertFalse(event.record()["handled"])
