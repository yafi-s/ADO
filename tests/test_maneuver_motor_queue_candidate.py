"""Decision controls for the isolated stopped-vehicle follow experiment."""
import unittest
import carla
import numpy as np
from unittest.mock import Mock
from tools.maneuver_motor_queue_candidate import MotorQueue, isolated_episode
from crosswalk import drive


def vehicle(f, v=0.):
    return dict(box_frame=f,label="car",radial_ground_speed_mps=v,stationary_doppler=abs(v)<.55)


class MotorQueueTests(unittest.TestCase):
    def test_new_motor_vehicle_keeps_source_lane_and_reconsiders_after_real_dwell(self):
        q=MotorQueue()
        for f in range(40):
            q.update(f,100.+f*.2,140.-(100.+f*.2),140.-(100.+f*.2),0.,0,2.,vehicle(f) if f%2==0 else None)
        self.assertEqual(q.choose("pass"),"follow")
        self.assertEqual(q.choose("yield"),"follow")
        self.assertEqual(q.wait,0)
        self.assertEqual(q.choose("pass",required_merge=True),"pass")
        for f in range(40,119):
            q.update(f,124.,16.,16.,0.,0,0.,vehicle(f) if f%2==0 else None)
        self.assertEqual(q.choose("pass"),"follow")
        q.update(119,124.,16.,16.,0.,0,0.,None)
        self.assertEqual(q.choose("pass"),"pass")
        self.assertEqual(q.wait,80)

    def test_one_camera_exposure_or_unclassified_object_cannot_block_native_choice(self):
        q=MotorQueue()
        for f in range(11):
            q.update(f,100.,25.,25.,0.,0,2.,vehicle(0) if f<2 else None)
        self.assertIsNone(q.anchor)
        self.assertEqual(q.choose("pass"),"pass")
        self.assertEqual(q.choose("yield"),"yield")

    def test_moving_and_restarting_cases_are_independent(self):
        q=MotorQueue()
        for f in range(20):
            q.update(f,10.,20.+.16*f,20.+.16*f,1.6,0,3.,vehicle(f,1.6))
        self.assertEqual(q.choose("pass"),"pass")
        for f in range(20,42):
            q.update(f,10.,23.2,23.2,0.,0,0.,vehicle(f))
        self.assertEqual(q.choose("pass"),"follow")
        x=23.2; speed=0.
        for f in range(42,58):
            speed+=.32;x+=.1*speed;q.update(f,10.,x,x,speed,0,2.,None)
            self.assertEqual(q.choose("pass"),"follow")
        for f in range(58,85):
            x+=.16;q.update(f,10.,x,x,1.6,0,2.,None)
        self.assertEqual(q.choose("pass"),"pass")

    def test_sparse_limits_do_not_transfer_dwell_to_a_different_surface(self):
        q=MotorQueue()
        for f in range(81):
            q.update(f,20.,10.,10.,0.,0,0.,vehicle(f))
        q.update(82,20.,27.,27.,0.,0,0.,None)
        self.assertIsNone(q.anchor)
        self.assertFalse(q.released)

    def test_fitted_preference_is_independent_of_the_engineering_dwell_comparator(self):
        q=MotorQueue()
        for f in range(90):
            q.update(f,20.,14.,14.,0.,0,0.,vehicle(f) if f%2==0 else None)
        self.assertGreaterEqual(q.wait,80)
        head=Mock()
        feat=q.preference_features(89,14.,0.,0.,18.)
        head.predict.return_value=("follow",.9)
        q.learned_preference(head,feat)
        self.assertEqual(q.choose("pass"),"follow")
        q.wait=0
        head.predict.return_value=("defer_to_existing_choice",.2)
        q.learned_preference(head,q.preference_features(90,14.,0.,0.,18.))
        self.assertEqual(q.choose("pass"),"pass")
        self.assertEqual(q.choose("yield"),"yield")
        self.assertEqual(q.choose("keep"),"keep")

    def test_fitted_planner_from_native_map_keeps_semantic_queue_on_source(self):
        from crosswalk.stationary_intent import StationaryIntent
        from tools.maneuver_motor_queue_candidate import candidate_planner
        from tests.test_maneuver_source_vehicle import surface
        from tests.test_urban_map import MAP
        if not MAP.is_file():
            self.skipTest("Native map unavailable")
        road=carla.Map("Town03_Opt",MAP.read_text())
        route=[road.get_waypoint_xodr(67,-2,k).transform.location for k in range(173,286)]
        pose=road.get_waypoint_xodr(67,-2,205.).transform
        ego=Mock();ego.get_transform.return_value=pose
        ego.bounding_box=carla.BoundingBox(carla.Location(z=.76),carla.Vector3D(2.45,.92,.76))
        cls=candidate_planner(StationaryIntent())
        data=surface()
        target_ground=np.asarray([(v,-3.5,-2.30,.5) for v in np.arange(-37.,40.,.30)])
        sweep=np.vstack((data["lidar"],target_ground))
        for boxes, expected in ((data["detections"],"follow"),([],"pass")):
            planner=cls.from_route(road,route)
            for i in range(22):
                planner.preview(ego,sweep,np.zeros(14),2.,camera_inputs=dict(
                    frame=121+i,boxes=boxes,box_frame=121+i-i%2,box_pose=pose,radar=data["radar"]))
            self.assertEqual(planner.info["choice"],expected)
            self.assertEqual(planner.info["committed"],not bool(boxes))
            self.assertEqual(planner.info["fitted_choice"],"pass")
            if boxes:
                self.assertIsNone(planner.speed_limit)
                self.assertEqual(planner.info["source_motor_vehicle"]["preference"],"follow")

    def test_existing_episode_can_bind_sensor_timing(self):
        cls,method=isolated_episode(drive)
        self.assertTrue(callable(method))
        self.assertTrue(issubclass(cls,__import__('crosswalk.route_maneuvers',fromlist=['RouteManeuvers']).RouteManeuvers))


if __name__=="__main__":unittest.main()
