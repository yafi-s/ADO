"""Current geometric box eligibility and nine-exposure legacy attribution."""
from collections import deque
import gzip
import inspect
import json
import unittest

import numpy as np

from crosswalk.camera_geometry import CAMERAS
from crosswalk.city_maneuvers import camera_horizon
from crosswalk.vision import summarize
from tests.test_city_maneuvers_pace import PATH,returns,road
from tools.maneuver_camera_association_candidate import CameraAssociation,GeometricSource,flank,parent


def measured_box(u):
    camera=CAMERAS["front"]
    distance=16.
    center=(u-.5)*(distance-camera.x)/camera.fx
    x,y,z=np.meshgrid(np.linspace(distance,distance+.3,4),np.linspace(center-.28,center+.28,10),
                      np.linspace(.08,1.70,10))
    laser=np.vstack((road()[0],returns(x,y,z)))
    px=.5+camera.fx*y/(x-camera.x)
    py=.5-camera.fy*(z-camera.z)/(x-camera.x)
    detection=dict(label="person",score=.91,box=[float(px.min()),float(py.min()),float(px.max()),float(py.max())])
    horizon,count=camera_horizon([detection],laser,PATH)
    return summarize([detection])[1],horizon,count


class CameraAttribution(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        folder=parent.ROOT/"results/city-maneuvers/pace/native-v07-boulevard/boulevard"
        with gzip.open(folder/"trace.jsonl.gz","rt") as stream:
            cls.a=next(r for r in map(json.loads,stream) if r["frame"]==15354)["tactical_pass"]
        with np.load(folder/"observations.npz") as tape:
            cls.fields=dict(zip(tape["feature_names"],tape["features"][tape["frame"]==15354][0]))

    def setUp(self):
        self.m=CameraAssociation.load(parent.MODEL)
        self.m.new_observer()
        self.m.tactical_speed_limit=0.
        self.history=deque(maxlen=9)

    def frame(self,fr,height=0.,horizon=7.,count=1,updates=None):
        self.history.append(height)
        x=dict(self.fields,camera_stop_horizon_m=horizon,off_path_camera_boxes=count,rgb_person_height=height,
               rgb_person_height_recent=float(np.mean(self.history)),confirmed_scans=0.,geometry_scans=3.)
        x.update(updates or {})
        self.m.record_camera(fr,x)
        return self.m.allowance(dict(self.a),x),x

    def test_frozen_rule_diff_and_real_box_association(self):
        before=inspect.getsource(parent.FreeSource.allowance)
        after=inspect.getsource(GeometricSource.allowance)
        old='(x["off_path_camera_boxes"]>=1. and self.standoff+1.<camera<59.\n                          and max(x["rgb_person_height"],x["rgb_person_height_recent"])>0.)'
        self.assertEqual(before.replace(old,'x["off_path_camera_boxes"]>=1. and self.standoff+1.<camera<59.'),after)
        peripheral=measured_box(.79)
        central=measured_box(.50)
        intermediate=measured_box(.718)
        self.assertEqual(peripheral[0],0.)
        self.assertGreater(peripheral[1],self.m.standoff+1.)
        self.assertGreater(central[0],.055)
        self.assertEqual(central[1],0.)
        self.assertGreater(intermediate[0],.055)
        self.assertGreater(intermediate[1],self.m.standoff+1.)
        for fr in range(9):
            result,x=self.frame(fr,*peripheral)
        self.assertGreater(result[0],0.)
        self.assertEqual(x["rgb_person_height"],0.)
        self.assertEqual(parent.FreeSource.allowance(self.m,self.a,x)[0],0.)
        self.assertEqual(self.frame(9,*central)[0][0],0.)
        self.assertEqual(self.frame(10,*peripheral)[0][0],0.)
        for fr in range(11,18):
            self.assertEqual(self.frame(fr,*peripheral)[0][0],0.)
        self.assertGreater(self.frame(18,*peripheral)[0][0],0.)

    def test_attributed_recent_and_gaps(self):
        peripheral=measured_box(.79)
        resolved=measured_box(.718)
        for fr in range(8):
            self.frame(fr,*peripheral)
        self.assertGreater(self.frame(8,*resolved)[0][0],0.)
        result,x=self.frame(9,*peripheral)
        self.assertGreater(x["rgb_person_height_recent"],0.)
        self.assertGreater(result[0],0.)
        self.assertEqual(self.frame(10,.04,7.,1)[0][0],0.)
        # The small central detection did not meet camera_horizon's box filter.
        self.assertEqual(self.frame(11,*peripheral)[0][0],0.)
        self.assertEqual(self.frame(17,*peripheral)[0][0],0.)
        for fr in range(18,25):
            self.assertEqual(self.frame(fr,*peripheral)[0][0],0.)
        self.assertGreater(self.frame(25,*peripheral)[0][0],0.)
        # No past calibrated box accounts for the independently reported recent exposure.
        self.assertEqual(self.frame(26,*peripheral,updates={"rgb_person_height_recent":.03})[0][0],0.)

    def test_positive_camera_never_overrules_other_constraints(self):
        for f in range(9):
            _,x=self.frame(f)
        for change in ({"crossing_stop_gap_m":4.},{"rgb_lamp_red_probability":.9},
                       {"camera_stop_horizon_m":0.},{"camera_stop_horizon_m":60.},
                       {"off_path_camera_boxes":0.},{"path_body_radar":3.},
                       {"geometry_scans":2.},{"oncoming_gap_recent":12.}):
            self.assertEqual(self.m.allowance(dict(self.a),x|change)[0],0.)
        self.assertEqual(self.m.allowance(self.a|{"source_lateral_m":-.9},x)[0],0.)
        self.assertEqual(self.m.allowance(self.a|{"return_swept_intrusion":True},x)[0],0.)
        self.assertIsNone(self.m.allowance(self.a|{"committed":True},x)[0])
        self.assertTrue(isinstance(self.m,flank.ParallelFlank))


if __name__=="__main__":
    unittest.main()
