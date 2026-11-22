"""Regression checks on swept-body, range fallback, signing and measured stopping."""
import unittest
from pathlib import Path

import numpy as np

from crosswalk.city_drive import load_driver
from crosswalk.city_maneuvers import camera_horizon, opposing_surface, shoulder_scan
from crosswalk.city_open_road import PavementHistory, long_sight
from crosswalk.policy import LIDAR_Z
from crosswalk.speed_signs import SignedRouteLimits

ROOT=Path(__file__).resolve().parents[1]
PATH=np.column_stack(([0.,3.,7.,13.,18.,25.,32.,40.,50.,60.,72.],np.zeros(11)))


def returns(x,y,z):
    x,y,z=np.broadcast_arrays(x,y,z)
    return np.column_stack((x.ravel(),y.ravel(),z.ravel()-LIDAR_Z,np.ones(x.size))).astype(np.float32)


def road():
    x,y=np.meshgrid(np.arange(.8,74.,.4),np.arange(-8.,8.1,.3))
    lidar=returns(x,y,0.)
    x=np.arange(15.,60.)
    depth=np.hypot(x-2.,.9)
    radar=np.column_stack((-10.*(x-2.)/depth,np.zeros(len(x)),np.arctan2(-.9,x-2.),depth)).astype(np.float32)
    return lidar,radar


class ManeuverPaceTests(unittest.TestCase):
    def test_moving_low_prop_into_lane_removes_both_shoulder_and_high_speed_permission(self):
        l,r=road()
        x,y,z=np.meshgrid(np.arange(11.,11.5,.1),np.arange(1.95,2.45,.1),np.arange(.3,1.15,.15))
        edge=np.vstack((l,returns(x,y,z)))
        inside=np.vstack((l,returns(x,y-2.,z)))
        self.assertGreater(long_sight(l,r,PATH,10.)[0],60.)
        self.assertGreater(shoulder_scan(edge,PATH)[1],11.)
        self.assertGreater(len(shoulder_scan(edge,PATH)[0]),0)
        self.assertEqual(shoulder_scan(inside,PATH)[1],0.)
        sight,block=long_sight(inside,r,PATH,10.)
        self.assertLess(sight,9.)
        self.assertLess(block,9.)
        self.assertEqual(long_sight(l,np.empty((0,4)),PATH,10.),(0.,0.))

    def test_registered_pavement_cannot_hide_new_obstruction_or_extend_empty_scan(self):
        road_laser,radar=road()
        history=PavementHistory()
        for i in range(6):
            sight,_=long_sight(road_laser,radar,PATH,10.,history,[i,0.,0.])
            self.assertGreater(sight,60.)
        x,y,z=np.meshgrid(np.arange(17.,17.4,.1),np.arange(-.6,.6,.2),np.arange(.4,.85,.1))
        obstacle=np.vstack((road_laser,returns(x,y,z)))
        sight,gap=long_sight(obstacle,radar,PATH,10.,history,[6.,0.,0.])
        self.assertLessEqual(sight,15.)
        self.assertLessEqual(gap,15.)
        # Old road points cannot become the only far witness on the next scan.
        near=road_laser[road_laser[:,0]<13.]
        self.assertLess(long_sight(near,radar,PATH,10.,history,[7.,0.,0.])[0],13.)

    def test_remote_upright_needs_height_stationarity_and_pavement_beyond_it(self):
        floor,radar=road()
        def measured(origin, height=2.6, pavement=True, moving=False):
            history=PavementHistory()
            for i in range(4):
                x,y,z=np.meshgrid(np.linspace(9.,9.45,3)+(i*.45 if moving else 0.),
                                   np.linspace(origin,origin+.35,3),np.linspace(.45,height,6))
                base=floor if pavement else floor[floor[:,1]<origin+.35]
                scan=np.vstack((base,returns(x,y,z)))
                long_sight(scan,radar,PATH,10.,history,[0.,0.,0.])
            return history.post_sight
        self.assertGreater(measured(6.7),16.)
        self.assertEqual(measured(6.7,height=1.8),0.)
        self.assertEqual(measured(6.7,pavement=False),0.)
        self.assertEqual(measured(6.7,moving=True),0.)
        self.assertEqual(measured(3.8),0.)

    def test_central_person_and_unresolved_peripheral_box_abstain(self):
        l,_=road()
        for name,box in (("person",[.44,.38,.52,.66]),("bicycle",[.83,.35,.98,.65]),
                         ("person",[.90,.10,.96,.31])):
            self.assertEqual(camera_horizon([dict(label=name,box=box)],l,PATH)[0],0.)
        self.assertEqual(camera_horizon([dict(label="person",box=[.83,.35,.98,.64])],np.zeros((0,4)),PATH)[0],0.)

    def test_opposing_car_on_bend_reenters_swept_path_and_cannot_release_speed(self):
        laser,_=road()
        rx=np.linspace(24.,25.,14)
        ry=np.linspace(-8.8,-9.4,14)
        dx,dy,dz=rx-2.,ry,np.full(len(rx),.35)
        depth=np.sqrt(dx*dx+dy*dy+dz*dz)
        radar=np.column_stack((-4.-6.*dx/depth,np.arctan2(dy,dx),np.arcsin(dz/depth),depth))
        self.assertGreater(opposing_surface(laser,radar,PATH,6.)[0],16.)
        bend=PATH.copy()
        bend[:,1]=-.014*bend[:,0]**2
        self.assertEqual(opposing_surface(laser,radar,bend,6.)[0],0.)

    def test_turn_toward_a_near_peripheral_person_prevents_camera_release(self):
        laser,_=road()
        # A measured person at 9m depth on the right of the forward image.
        # The second navigation polyline sweeps toward the person.
        x,y,z=np.meshgrid(np.linspace(10.,10.3,3),np.linspace(4.9,5.45,8),np.linspace(.1,1.8,9))
        laser=np.vstack((laser,returns(x,y,z)))
        box=dict(label="person",box=[.765,.484,.824,.659])
        self.assertGreater(camera_horizon([box],laser,PATH)[0],0.)
        bend=PATH.copy()
        bend[:,1]=.045*np.minimum(bend[:,0],15.)**2
        self.assertEqual(camera_horizon([box],laser,bend)[0],0.)

    def test_signed_preview_accounts_for_the_native_tested_operating_range(self):
        limits=SignedRouteLimits([(100.,30.)],2.25)
        self.assertIsNone(limits.approach(44.,14.,90.))
        self.assertGreater(limits.approach(44.,14.,90.,cruise_ceiling_mps=60./3.6)[0],40./3.6)
        self.assertLess(limits.approach(80.,14.,90.,cruise_ceiling_mps=60./3.6)[0],30./3.6)

    def test_validated_speed_requires_three_visible_sweeps_and_respects_conflicts(self):
        model,names,_=load_driver(ROOT/"models/city-maneuvers/candidate.npz")
        def feature(**change):
            x=np.zeros(len(names),dtype=float)
            for k in ("path_core_lidar","path_body_lidar","path_body_radar","path_body_radar_recent",
                      "path_body_lidar_recent","path_core_lidar_recent","right_edge_lidar","left_edge_lidar",
                      "right_edge_lidar_recent","oncoming_gap","oncoming_gap_recent","oncoming_near_gap_recent"):
                x[names.index(k)]=36.
            for k in ("junction_gap","crossing_stop_gap_m","mapped_lamp_gap_m","camera_stop_horizon_m","long_sight_m","long_body_gap_m"):
                x[names.index(k)]=60.
            for k in ("observed_corridor_m","observed_conflict_m","verified_forward_m","verified_cross_sight_m"):
                x[names.index(k)]=24.
            for k,v in dict(geometry_scans=3,confirmed_scans=3,ego_speed_mps=13.,
                            route_3m_x_m=3.,route_7m_x_m=7.,route_13m_x_m=13.).items():
                x[names.index(k)]=v
            for k,v in change.items():
                x[names.index(k)]=v
            return x
        model.new_observer()
        model.posted_limit_kmh=90.
        model.predict(feature())
        self.assertGreater(model.last_speed_request,50./3.6)
        self.assertEqual(model.progress_evidence["speed_regime"],"verified_open_road")
        for kwargs in (dict(geometry_scans=2),dict(camera_stop_horizon_m=0.),
                       dict(oncoming_gap=12.),dict(mapped_lamp_gap_m=20.),
                       dict(junction_gap=20.,bend_13_25=-.78)):
            model.new_observer()
            model.predict(feature(**kwargs))
            self.assertLessEqual(model.last_speed_request,11.)
        model.signed_speed_limit=7.8
        model.predict(feature())
        self.assertLessEqual(model.last_speed_request,7.8)
        action=model.predict(feature(crossing_stop_gap_m=3.,long_body_gap_m=4.))
        self.assertLess(model.last_speed_request,1.)
        self.assertGreaterEqual(action[2],.45)
        self.assertEqual(action[1],0.)
