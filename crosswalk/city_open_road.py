"""Longitudinal support from observed pavement and swept-body clearance at higher speeds."""
from collections import deque
import numpy as np

from . import behavior_v2 as v2
from .city_clearance import BODY
from .city_progress import _pavement_horizon
from .policy import _returns, LIDAR_Z, RADAR_X, RADAR_Z


class PavementHistory:
    """Register six preceding positive pavement sweeps in the current ego frame."""
    def __init__(self):
        self.scans=deque(maxlen=6)
        self.posts=deque(maxlen=3)
        self.post_sight=0.

    def clear(self):
        self.scans.clear()
        self.posts.clear()
        self.post_sight=0.

    def far_shoulder(self,x,y,h,station,lateral,pose):
        """Resolve tall narrow stationary surfaces with visible pavement behind them."""
        self.post_sight=0.
        pose=np.asarray(pose,dtype=float)
        if pose.shape!=(3,) or not np.isfinite(pose).all():
            self.posts.clear()
            return
        yaw=np.deg2rad(pose[2])
        rotation=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
        ground=abs(h)<.28
        solid=(h>.28)&(h<3.2)
        horizon=24.
        anchors=[]
        confirmed=0
        for sign in (-1.,1.):
            side=sign*lateral
            floor=ground&(side>4.)&(side<9.)&(station>=0.)&(station<24.)
            horizon=min(horizon,_pavement_horizon(station,floor))
            flank=solid&(side>BODY)&(side<9.)&(station> -1.8)&(station<24.)
            ix=np.flatnonzero(flank)
            if not len(ix):
                continue
            ix=ix[np.argsort(station[ix])]
            for group in np.split(ix,np.flatnonzero(np.diff(station[ix])>.7)+1):
                s,d,heights=station[group],side[group],h[group]
                if max(s)<0. or min(s)>=horizon:
                    continue
                # Ignore a surface only when its entire measured width is well
                # outside the approach shoulder, its upright extent exceeds a
                # pedestrian and two lower rays independently see past it.
                behind=ground&(side>max(d)+.2)&(side<max(d)+2.8)&(station>min(s)-1.)&(station<max(s)+1.)
                qualifying=(len(group)>=4 and min(d)>5.7 and np.ptp(s)<.95 and np.ptp(d)<.95
                            and max(heights)>2.15 and min(heights)<1.25 and behind.sum()>=2)
                if qualifying:
                    center=np.array([np.median(x[group]),np.median(y[group])])@rotation.T+np.asarray(pose[:2])
                    anchors.append(center)
                    if len(self.posts)==3 and all(any(np.linalg.norm(center-old)<.6 for old in scan)
                                                   for scan in (self.posts[0],self.posts[-1])):
                        confirmed+=1
                        continue
                # Keep unclassified nearby surfaces even when a separate post
                # already has temporal support in this frame.
                horizon=min(horizon,max(0.,min(s)))
        self.posts.append(anchors)
        if confirmed:
            self.post_sight=float(horizon)

    def previous(self,xy,pose):
        pose=np.asarray(pose,dtype=float)
        if pose.shape!=(3,) or not np.isfinite(pose).all():
            self.clear()
            return np.empty((0,2))
        angle=np.deg2rad(pose[2])
        c,s=np.cos(angle),np.sin(angle)
        rotation=np.array([[c,-s],[s,c]])
        origin=pose[:2]
        previous=(np.concatenate(self.scans)-origin)@rotation if self.scans else np.empty((0,2))
        self.scans.append(xy@rotation.T+origin)
        return previous


def long_sight(lidar, radar, path, speed, history=None, pose=None):
    """Return pavement extent and first obstruction after removing the measured ground plane."""
    laser, doppler = _returns(lidar), _returns(radar)
    if len(laser)<100 or len(doppler)<20 or len(path)<8:
        if history is not None:
            history.clear()
        return 0.,0.
    x,y,z = laser[:,0],laser[:,1],laser[:,2]+LIDAR_Z
    a,b,c = v2._ground_plane(x,y,z)
    # This clearance head is for level, measured pavement. A changing grade
    # can shorten the valid witness or put a surface above the ground plane.
    if abs(a)>.075 or abs(b)>.085:
        if history is not None:
            history.clear()
        return 0.,0.
    roi=(x>0.) & (x<77.) & (abs(y)<38.)
    x,y,h=x[roi],y[roi],(z-a*x-b*y-c)[roi]
    s,d=v2._project(x,y,path)
    end=float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum())-2.25
    valid=(s>=0.) & (s<min(70.,end-.2))
    ground=valid & (abs(h)<.22)
    # A stationary sign, low cone, person or moving vehicle is a genuine
    # obstruction until outside the swept path and its approach shoulders.
    solid=valid & (h>.20) & (h<3.2)
    # The legacy 36m horizon cannot measure the high-speed extension.
    def first(stations, mask):
        return float(min(stations[mask])) if mask.any() else 75.
    body=first(s,solid & (abs(d)<BODY+.4))
    shoulder=first(s,solid & (abs(d)<4.5))
    velocity,az,alt,depth=doppler.T
    flat=depth*np.cos(alt)
    rx,ry,rz=flat*np.cos(az)+RADAR_X,flat*np.sin(az),depth*np.sin(alt)+RADAR_Z
    st,side=v2._project(rx,ry,path)
    raised=rz-a*rx-b*ry-c
    rv=velocity+speed*np.cos(az)*np.cos(alt)
    targets=(st>=0.) & (st<end-.2) & (rx>0.) & (raised>.18) & (raised<3.2)
    body=min(body,first(st,targets & (abs(side)<BODY+.4)))
    shoulder=min(shoulder,first(st,targets & (abs(side)<4.5)),
                 first(st,targets & (abs(side)<7.0) & (abs(rv)>2.)))
    # Each range section needs positive ground in the lane and on both sides.
    # Beyond 15m, successive rays of a 32-channel spinning sensor land further
    # apart. Limit the gap between actual returns in each strip. A bounded gap
    # moves the horizon only once the next surface has really been observed.
    ps,pd=np.empty(0),np.empty(0)
    if history is not None:
        pavement=ground & (s>10.) & (abs(d)<5.5)
        previous=history.previous(np.column_stack((x[pavement],y[pavement])),pose)
        history.far_shoulder(x,y,h,s,d,pose)
        if len(previous):
            ps,pd=v2._project(previous[:,0],previous[:,1],path)
    strips=[]
    for lo,hi in ((-4.8,-1.5),(-1.2,1.2),(1.5,4.8)):
        stations=np.sort(s[ground & (d>lo) & (d<hi)])
        if len(stations)<5:
            return 0.,body
        if len(ps):
            # Recorded ground can bridge the gap between laser rings only
            # between currently observed returns. In particular it cannot
            # extend beyond today's last true witness in any strip.
            old=ps[(pd>lo)&(pd<hi)&(ps>10.)&(ps<stations[-1])]
            stations=np.sort(np.r_[stations,old])
        delta=np.diff(np.r_[0.,stations])
        # Sampling geometry for level pavement at the actual 32-channel
        # vertical spacing. At long range cap allowed inter-ring spacing.
        allowed=np.clip(1.+.024*stations**2/2.3,3.,12.)
        gaps=np.flatnonzero(delta>allowed)
        count=int(gaps[0]) if len(gaps) else len(stations)
        strips.append(float(stations[count-1]) if count else 0.)
    return max(0.,min(*strips,body,shoulder)),body
