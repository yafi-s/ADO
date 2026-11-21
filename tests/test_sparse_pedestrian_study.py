import math
import unittest

import numpy as np

from crosswalk.camera_geometry import CAMERAS
from crosswalk.sparse_pedestrian_study import SparsePedestrianStudy


def sample(distance=32, velocities=(-2.,-2.1,-2.05), radar_offset=0.):
    c=CAMERAS["front"]
    u=.5+c.fx*1.5/(distance-c.x)
    top=.5-c.fy*(1.8-c.z)/(distance-c.x)
    bottom=.5-c.fy*(.1-c.z)/(distance-c.x)
    half_width=c.fx*.25/(distance-c.x)
    box=dict(label="person",score=.75,image_id=4,range_supported=False,
             box=[u-half_width,top,u+half_width,bottom])
    scan=np.array([[distance,1.5,1.1-2.3,1.]])
    radar=[]
    for i,speed in enumerate(velocities):
        x,y,z=distance-2+radar_offset,1.5+.015*i,1.1-.9
        radar.append([speed,math.atan2(y,x),math.atan2(z,math.hypot(x,y)),math.sqrt(x*x+y*y+z*z)])
    return [box],scan,np.asarray(radar).reshape(-1,4)


class SparsePedestrianTests(unittest.TestCase):
    def test_two_exposures_with_independent_laser_and_radar(self):
        obs=sample()
        study=SparsePedestrianStudy()
        self.assertFalse(study.observe(*obs,np.zeros(3),0.))
        row,=study.observe(*obs,np.zeros(3),.2)
        self.assertEqual(row["lidar_indices"],[0])
        self.assertEqual(row["radar_hits"],3)
        self.assertAlmostEqual(row["xy"][0],32.18,places=2)

    def test_inconsistent_and_ambiguous_depth_and_missing_evidence(self):
        boxes,scan,radar=sample()
        for values in (sample(radar_offset=2.),sample(velocities=(-2.,-4.,-6.)),sample(velocities=(-2.,)),
                       sample(distance=9.),([*boxes,dict(boxes[0],label="car")],scan,radar),
                       (boxes,scan[:0],radar),([dict(boxes[0],range_supported=True)],scan,radar)):
            observer=SparsePedestrianStudy()
            for t in (0.,.2,.4):
                self.assertFalse(observer.observe(*values,np.zeros(3),t))

    def test_disconnected_exposures_do_not_confirm(self):
        obs=sample()
        for pose,t in ((np.zeros(3),.7),(np.array([5.,0.,0.]),.2)):
            observer=SparsePedestrianStudy()
            observer.observe(*obs,np.zeros(3),0.)
            self.assertFalse(observer.observe(*obs,pose,t))
