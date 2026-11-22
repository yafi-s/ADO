"""Associate an actual front-camera motor vehicle to the measured native source-lane surface."""
import numpy as np

from crosswalk.behavior_v2 import _ground_plane
from crosswalk.camera_geometry import CAMERAS
from crosswalk.policy import LIDAR_Z
from crosswalk.radar_geometry import ego_returns


def _matrix(pose):
    return np.asarray(pose.get_matrix() if hasattr(pose,"get_matrix") else pose,dtype=float)


def source_surface_vehicle(detections,lidar,radar,ego_pose,camera_pose,corridor,station,source_range,
                           speed,frame,box_frame):
    """Return measured support for one motor vehicle, or None when identity is unresolved.

    Inputs are the unmodified control NanoDet xyxy boxes, current roof sweep
    and forward radar (native velocity/azimuth/altitude/depth), actual ego
    transforms at the two exposures, and the planner's source-lane surface.
    Source range and returned longitudinal positions are ego-origin metres.
    This function does not track elapsed dwell or make any lane/speed choice.
    """
    if (not isinstance(frame,(int,np.integer)) or not isinstance(box_frame,(int,np.integer)) or
            frame-box_frame not in (0,1) or not 3.<source_range<59. or not np.isfinite(speed)):
        return None
    boxes=[d for d in detections if d["label"] in ("car","truck","bus") and d["score"]>=.35]
    if not boxes or corridor is None:
        return None
    ego,old=_matrix(ego_pose),_matrix(camera_pose)
    if ego.shape!=(4,4) or old.shape!=(4,4) or not np.isfinite(ego).all() or not np.isfinite(old).all():
        return None
    raw=np.asarray(radar).reshape(-1,4)
    raw=raw[np.isfinite(raw).all(axis=1)&(raw[:,3]>0.)]
    if len(raw)<3:
        return None
    rays=ego_returns(raw,"radar")
    doppler=rays[:,3]+speed*np.cos(raw[:,1])*np.cos(raw[:,2])
    laser=np.asarray(lidar).reshape(-1,4)[:,:3].copy()
    laser=laser[np.isfinite(laser).all(axis=1)]
    if len(laser)<3:
        return None
    laser[:,2]+=LIDAR_Z
    cam=CAMERAS["front"]
    points=np.vstack((laser,rays[:,:3]))
    world=points@ego[:3,:3].T+ego[:3,3]
    rel,lateral,height,_,inside=corridor.project_scan(world,station)
    cam_points=(world-old[:3,3])@old[:3,:3]
    # Reproject the current returns to the *actual box exposure*; replay of a
    # held 0.1 s box on the current pose can assign the wrong traffic lane.
    depth=cam_points[:,0]-cam.x
    uv=np.column_stack((.5+cam.fx*(cam_points[:,1]-cam.y)/np.maximum(depth,.001),
                        .5-cam.fy*(cam_points[:,2]-cam.z)/np.maximum(depth,.001)))
    lane=inside&(abs(lateral)<1.22)&(height>.25)&(height<3.2)&(depth>0.)&(rel>2.)&(rel<59.)
    ga,gb,gc=_ground_plane(*cam_points[:len(laser)].T)
    matches=[]
    for detection in boxes:
        left,top,right,bottom=detection["box"]
        if not (0.<=left<right<=1. and 0.<=top<bottom<=1.):
            continue
        ray_side=((left+right)/2.-.5)/cam.fx
        ray_down=(bottom-.5)/cam.fy
        denom=ray_down+ga+gb*ray_side
        if denom<=0.:
            continue
        foot_depth=(cam.z-ga*cam.x-gb*cam.y-gc)/denom
        if not 1.<foot_depth<85.:
            continue
        foot=np.array([[cam.x+foot_depth,cam.y+foot_depth*ray_side,cam.z-foot_depth*ray_down]])
        foot_s,foot_d,_,_,on_lane=corridor.project_scan(foot@old[:3,:3].T+old[:3,3],station)
        if (not on_lane[0] or abs(foot_d[0])>1.25 or
                abs(foot_s[0]-source_range)>max(3.,.19*source_range)):
            continue
        hits=lane&(left<=uv[:,0])&(uv[:,0]<=right)&(top<=uv[:,1])&(uv[:,1]<=bottom)
        if np.count_nonzero(hits&(rel<source_range-2.1))>=3:
            continue
        close=hits&(abs(rel-source_range)<1.65)
        laser_hits,radar_hits=close[:len(laser)],close[len(laser):]
        nl,nr=int(np.sum(laser_hits)),int(np.sum(radar_hits))
        if nl<2 or nr<3:
            continue
        laser_s=float(np.median(rel[:len(laser)][laser_hits]))
        radar_s=float(np.median(rel[len(laser):][radar_hits]))
        laser_d=float(np.median(lateral[:len(laser)][laser_hits]))
        radar_d=float(np.median(lateral[len(laser):][radar_hits]))
        if abs(laser_s-radar_s)>1.3 or abs(laser_d-radar_d)>1.:
            continue
        # A few closely co-located beams on a cone can lie within the image
        # of a distant vehicle. Require a vehicle-sized measured face as well
        # as a vehicle-sized image. Percentiles ignore isolated high outliers.
        face_width=float(np.ptp(np.quantile(lateral[close],[.10,.90])))
        face_high=float(np.quantile(height[close],.90))
        if face_width<.65 or face_high<.76:
            continue
        z=float(np.median(depth[close]))
        width,height_m=(right-left)*z/cam.fx,(bottom-top)*z/cam.fy
        if not (1.05<width<4.5 and .75<height_m<4.8):
            continue
        velocity=doppler[radar_hits]
        v=float(np.median(velocity));spread=float(np.quantile(abs(velocity),.85))
        matches.append(dict(label=detection["label"],confidence=float(detection["score"]),box=list(detection["box"]),
            frame=int(frame),box_frame=int(box_frame),lidar_hits=nl,radar_hits=nr,
            source_range_m=float(source_range),lidar_range_m=laser_s,radar_range_m=radar_s,
            lidar_lateral_m=laser_d,radar_lateral_m=radar_d,ground_ray_range_m=float(foot_s[0]),
            ground_ray_lateral_m=float(foot_d[0]),projected_width_m=float(width),projected_height_m=float(height_m),
            measured_face_width_m=face_width,measured_face_high_m=face_high,
            radial_ground_speed_mps=v,radial_abs_p85_mps=spread,
            stationary_doppler=bool(abs(v)<.55 and spread<.85)))
    return max(matches,key=lambda m:m["confidence"],default=None)
