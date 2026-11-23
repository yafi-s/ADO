"""Counterfactuals on source-only motion beside a positively measured parallel body."""
import gzip
import json
import unittest

import numpy as np

from tools.maneuver_parallel_flank_candidate import ParallelFlank,flank_scan,parent,LIDAR_Z


class ParallelFlankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path=np.column_stack(([0.,3.,7.,13.,18.,25.,32.,40.],np.zeros(8)))
        board=parent.ROOT/"results/city-maneuvers/pace/native-v07-boulevard/boulevard"
        with gzip.open(board/"trace.jsonl.gz","rt") as f:
            cls.a=next(r for r in map(json.loads,f) if r["frame"]==15354)["tactical_pass"]
        with np.load(board/"observations.npz") as obs:
            cls.x=dict(zip(obs["feature_names"],obs["features"][obs["frame"]==15354][0]))

    def setUp(self):
        self.m=ParallelFlank.load(parent.MODEL)
        self.m.new_observer()
        self.m.tactical_speed_limit=0.
        self.a=dict(self.a,target_front_m=0.,target_rear_m=0.,target_front_speed_mps=0.,
                    target_rear_closing_mps=0.)
        self.x=dict(self.x,observed_conflict_m=1.,verified_cross_sight_m=1.)

    def cloud(self,edge=2.65,face=True,margin=True,shift=0.):
        x,y=np.meshgrid(np.arange(-10.,29.,.22),np.arange(-9.5,9.6,.20))
        ground=np.c_[x.ravel(),y.ravel(),np.full(x.size,-LIDAR_Z),np.ones(x.size)]
        if not margin:
            ground=ground[(ground[:,1]< -2.05)|(ground[:,1]>-.90)]
        if not face:
            return ground
        x,z=np.meshgrid(np.arange(-2.2,3.3,.11)+shift,np.arange(.45,1.7,.18))
        side=np.c_[x.ravel(),np.full(x.size,-edge),z.ravel()-LIDAR_Z,np.ones(x.size)]
        return np.vstack((ground,side))

    def see(self,cloud,frame):
        self.m.flank_witness(frame,self.a["source_lane"],flank_scan(cloud,self.path,self.a))

    def prime(self):
        cloud=self.cloud()
        for frame in range(5):
            self.see(cloud,frame)

    def test_measured_parked_neighbor_at_same_station(self):
        self.prime()
        self.assertEqual(parent.FreeSource.allowance(self.m,self.a,self.x)[0],0.)
        cap,why=self.m.allowance(self.a,self.x)
        self.assertGreater(cap,0.)
        self.assertLessEqual(cap,1.5)
        self.assertEqual(why,"measured_parallel_flank")
        self.assertGreater(self.m.confirmed_flank["margin_m"],1.)
        self.assertEqual(self.a["target_front_m"],0.)
        self.assertEqual(self.a["target_rear_m"],0.)
        self.assertFalse(self.a["committed"])
        self.assertEqual(self.a["turn_signal"],"off")

    def test_intrusion_unknown_ground_heading_and_closing_car(self):
        for cloud in (self.cloud(edge=1.78),self.cloud(face=False),self.cloud(margin=False)):
            self.prime()
            self.see(cloud,5)
            self.assertIsNone(self.m.confirmed_flank)
            self.assertEqual(self.m.allowance(self.a,self.x)[0],0.)
        self.prime()
        for measured in ({"return_swept_intrusion":True},{"target_rear_closing_mps":4.},
                         {"target_front_speed_mps":3.},{"source_lateral_m":.4}):
            self.assertEqual(self.m.allowance(self.a|measured,self.x)[0],0.)
        self.assertEqual(self.m.allowance(self.a|{"source_lateral_m":.24},self.x|{"route_7m_y_m":.45})[0],0.)
        for i in range(5):
            self.see(self.cloud(shift=-.22*i),10+i)
        self.assertIsNone(self.m.confirmed_flank)
        for i in range(5):
            self.see(self.cloud(edge=2.68-.04*i),20+i)
        self.assertIsNone(self.m.confirmed_flank)

    def test_other_side_and_emerging_walker_own_the_stop(self):
        self.prime()
        for challenges in ({"crossing_stop_gap_m":4.},{"camera_stop_horizon_m":0.},
                           {"rgb_person_height":.11,"camera_stop_horizon_m":60.},
                           {"rgb_person_height":.08,"off_path_camera_boxes":1.,"camera_stop_horizon_m":4.9},
                           {"mapped_lamp_gap_m":15.},{"oncoming_gap":20.},{"path_body_lidar":3.}):
            self.assertEqual(self.m.allowance(self.a,self.x|challenges)[0],0.)
        cloud=self.cloud()
        for lateral in (-2.3,2.3):
            # Distinct nearby upright on either side of the projected source,
            # outside the stationary face's measured longitudinal interval.
            post=np.array([[4.4+dx,lateral,.95-LIDAR_Z,1.] for dx in np.arange(0.,.12,.01)])
            for i in range(5):
                self.see(np.vstack((cloud,post)),100+i)
            self.assertEqual(self.m.allowance(self.a,self.x)[0],0.)


if __name__=="__main__":
    unittest.main()
