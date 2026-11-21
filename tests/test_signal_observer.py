"""Traffic-signal pixels, survey placement and bounded observation age."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import carla
import numpy as np

from crosswalk.signal_observer import Head, SignalObserver, housing_features, visible_bulb


class SignalObserverTests(unittest.TestCase):
    def setUp(self):
        observer=object.__new__(SignalObserver)
        observer.mounts={"front":np.eye(4)}
        observer.recent={}
        observer.candidates={}
        center=np.array([18.,0.,2.8])
        verts=np.array([[x,y,z] for x in (-.1,.1) for y in (-.25,.25) for z in (-.5,.5)])+center
        observer.heads=[Head("signal","7",verts,center,0.)]
        self.observer = observer
        self.images = {"front": np.full((360,640,3),130,np.uint8)}

    def measure(self, sim_time, evidence=("red", .97), images=None):
        self.observer.classifier = SimpleNamespace(evidence=lambda crop: evidence, threshold=.85)
        return self.observer.read(carla.Transform(), self.images if images is None else images, sim_time)[0]

    def test_temporal_hold_requires_continuous_current_support_and_expires(self):
        self.assertEqual(self.measure(2.9)["state"], "unknown")
        row = self.measure(3.)
        self.assertEqual((row["source"],row["view"],row["state"]),("camera","front","red"))
        self.assertEqual(row["pixel_size"],[640,360])
        row = self.measure(3.1, ("red", .60))
        self.assertEqual((row["state"],row["age_s"]),("red",.1))
        self.assertNotIn("pixel_box",row)
        row = self.measure(3.2, ("red", .60))
        self.assertEqual((row["state"],row["source"]),("unknown","map"))

    def test_unsupported_frame_clears_color_and_prevents_gap_confirmation(self):
        self.measure(0.)
        self.assertEqual(self.measure(.1)["state"], "red")
        self.assertEqual(self.measure(.2, ("off", .97))["state"], "unknown")
        self.assertEqual(self.measure(.3)["state"], "unknown")
        self.assertEqual(self.measure(.4, ("red", .60))["state"], "unknown")
        self.assertEqual(self.measure(.5, ("green", .97))["state"], "unknown")
        self.assertEqual(self.measure(.6, ("green", .97))["state"], "green")
        self.assertEqual(self.measure(.7, images={})["state"], "unknown")

    def test_conflicting_weak_color_invalidates_previous_color(self):
        self.measure(0.)
        self.assertEqual(self.measure(.1)["state"], "red")
        self.assertEqual(self.measure(.2, ("green", .70))["state"], "unknown")
        self.assertEqual(self.measure(.3, ("red", .60))["state"], "unknown")

    @patch("crosswalk.signal_observer.visible_bulb", return_value=True)
    def test_small_housing_needs_two_consecutive_votes_and_does_not_coast(self, _visible):
        head = self.observer.heads[0]
        offset = np.array([37., 0., 0.])
        self.observer.heads = [Head(head.key, head.map_id, head.vertices + offset, head.center + offset, head.yaw)]
        self.measure(0.)
        self.assertEqual(self.measure(.1, ("off", .95))["state"], "unknown")
        self.assertEqual(self.measure(.2, ("red", .65))["state"], "unknown")
        self.assertEqual(self.measure(.3)["state"], "unknown")
        self.assertEqual(self.measure(.4)["state"], "red")
        self.assertEqual(self.measure(.5, ("red", .65))["state"], "unknown")

    def test_resolved_housing_bridges_one_uncertain_exposure(self):
        self.measure(0.)
        self.assertEqual(self.measure(.1)["state"], "red")
        self.assertEqual(self.measure(.2, ("red", .40))["age_s"], .1)
        self.assertEqual(self.measure(.3)["age_s"], 0.)

    def test_small_bulb_needs_enough_pixels_and_local_color_contrast(self):
        image = np.full((12, 8, 3), 100, np.uint8)
        self.assertFalse(visible_bulb(image, "red"))
        image[3:5, 3:5] = [125, 95, 95]
        self.assertTrue(visible_bulb(image, "red"))
        self.assertFalse(visible_bulb(image, "green"))
        self.assertFalse(visible_bulb(image[:9], "red"))

    def test_brightness_alone_cannot_turn_on_a_uniform_head(self):
        white=np.full((20,12,3),230,np.uint8)
        black=np.zeros_like(white)
        np.testing.assert_array_equal(housing_features(white),housing_features(black))
        white[4:7,4:8]=[255,40,10]
        self.assertFalse(np.allclose(housing_features(white),housing_features(black)))
