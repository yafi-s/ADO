"""Traffic-signal pixels, survey placement and bounded observation age."""
import unittest
from types import SimpleNamespace

import carla
import numpy as np

from crosswalk.signal_observer import Head, SignalObserver, housing_features


class SignalObserverTests(unittest.TestCase):
    def test_temporal_hold_requires_real_view_and_expires(self):
        observer=object.__new__(SignalObserver)
        observer.mounts={"front":np.eye(4)}
        observer.recent={}
        center=np.array([18.,0.,2.8])
        verts=np.array([[x,y,z] for x in (-.1,.1) for y in (-.25,.25) for z in (-.5,.5)])+center
        observer.heads=[Head("signal","7",verts,center,0.)]
        sample=np.full((360,640,3),130,np.uint8)
        observer.classifier=SimpleNamespace(read=lambda crop:("red", .97))
        row=observer.read(carla.Transform(),{"front":sample},3.)[0]
        self.assertEqual((row["source"],row["view"],row["state"]),("camera","front","red"))
        self.assertEqual(row["pixel_size"],[640,360])
        observer.classifier=SimpleNamespace(read=lambda crop:None)
        row=observer.read(carla.Transform(),{"front":sample},3.1)[0]
        self.assertEqual((row["state"],row["age_s"]),("red",.1))
        self.assertNotIn("pixel_box",row)
        row=observer.read(carla.Transform(),{"front":sample},3.4)[0]
        self.assertEqual((row["state"],row["source"]),("unknown","map"))

    def test_brightness_alone_cannot_turn_on_a_uniform_head(self):
        white=np.full((20,12,3),230,np.uint8)
        black=np.zeros_like(white)
        np.testing.assert_array_equal(housing_features(white),housing_features(black))
        white[4:7,4:8]=[255,40,10]
        self.assertFalse(np.allclose(housing_features(white),housing_features(black)))
