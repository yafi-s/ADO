"""Sensor-only motor vehicle association, including causal timing and source-lane negatives."""
import unittest

import numpy as np

from crosswalk.camera_geometry import CAMERAS
from crosswalk.policy import LIDAR_Z,RADAR_X,RADAR_Z
from tools.maneuver_source_vehicle import source_surface_vehicle


class StraightSource:
    def project_scan(self,world,station):
        n=len(world)
        return world[:,0]-station,world[:,1],world[:,2],np.full(n,3.5),np.ones(n,dtype=bool)


def surface(y=0.,half_width=.91,height=1.50,velocity=0.,reported_label="car"):
    yy,zz=np.meshgrid(np.linspace(y-half_width,y+half_width,11),np.linspace(.30,height,8))
    points=np.c_[np.full(yy.size,30.),yy.ravel(),zz.ravel()]
    laser=np.c_[points[:,0],points[:,1],points[:,2]-LIDAR_Z,np.ones(len(points))]
    px,py=np.meshgrid(np.linspace(2.,27.,31),np.linspace(-4.,4.,17))
    ground=np.c_[px.ravel(),py.ravel(),np.full(px.size,-LIDAR_Z),np.ones(px.size)]
    delta=points-[RADAR_X,0.,RADAR_Z]
    dist=np.linalg.norm(delta,axis=1)
    az=np.arctan2(delta[:,1],delta[:,0])
    alt=np.arcsin(delta[:,2]/dist)
    radar=np.c_[(velocity-2.)*np.cos(az)*np.cos(alt),az,alt,dist]
    cam=CAMERAS["front"]
    # Deliberately give even the cone counterfactual a car-sized camera box.
    # Its near-lane range returns must resolve the ambiguity independently.
    d=30.-cam.x
    box=[.5+cam.fx*(y-1.05)/d,.5-cam.fy*(1.65-cam.z)/d,
         .5+cam.fx*(y+1.05)/d,.5-cam.fy*(.03-cam.z)/d]
    return dict(detections=[dict(label=reported_label,score=.70,box=box)],
                lidar=np.vstack((laser,ground)),radar=radar,ego_pose=np.eye(4),camera_pose=np.eye(4),
                corridor=StraightSource(),station=0.,source_range=30.,speed=2.,frame=110,box_frame=110)


class SourceVehicle(unittest.TestCase):
    def test_class_and_motion_are_separate_measurements(self):
        for label in ("car","truck","bus"):
            args=surface(reported_label=label)
            static=source_surface_vehicle(**args)
            self.assertEqual(static["label"],label)
            self.assertTrue(static["stationary_doppler"])
            self.assertAlmostEqual(static["radar_range_m"],30.)
        moving=source_surface_vehicle(**surface(velocity=1.5))
        self.assertFalse(moving["stationary_doppler"])
        self.assertAlmostEqual(moving["radial_ground_speed_mps"],1.5,places=2)

    def test_decline_unsupported_source_or_box(self):
        for args in (surface(y=-3.5),surface(half_width=.15,height=.55),
                     surface(half_width=.80,height=.61),surface(reported_label="person")):
            self.assertIsNone(source_surface_vehicle(**args))
        args=surface()
        for update in ({"source_range":22.},{"box_frame":108},{"box_frame":111},
                       {"radar":args["radar"][:2]},{"lidar":args["lidar"][:1]}):
            self.assertIsNone(source_surface_vehicle(**(args|update)))

    def test_previous_exposure_uses_its_own_pose(self):
        args=surface()
        old=np.eye(4)
        old[0,3]=-.20
        found=source_surface_vehicle(**(args|{"camera_pose":old,"box_frame":109}))
        self.assertEqual((found["frame"],found["box_frame"]),(110,109))
        old[1,3]=3.50
        self.assertIsNone(source_surface_vehicle(**(args|{"camera_pose":old,"box_frame":109})))


if __name__=="__main__":
    unittest.main()
