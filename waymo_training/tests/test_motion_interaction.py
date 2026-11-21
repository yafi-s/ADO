"""Causality, frame conversion, portable weights and staged TFRecord integrity."""
import hashlib
import json
import math
import struct

import numpy as np
import pytest

from waymo_training.interaction import ActorTrack, FEATURE_NAMES, MotionSpeedModel, _basis, make_features
from waymo_training.motion_extract import SCENARIO, decode, masked_crc, read_staged, staged_objects


def scenario():
    msg = SCENARIO(scenario_id="static-fixture", current_time_index=10, sdc_track_index=0)
    msg.timestamps_seconds.extend(np.arange(41)/10)
    for typ in (1,1,2,3):
        msg.tracks.add(object_type=typ)
    # Ego faces world north. Its left is world west. The near pedestrian at
    # 13 m should score ahead of the stationary cyclist 40 m away.
    for j in range(41):
        sec=(j-10)/10
        for track,x,y,vx,vy in zip(msg.tracks, (0,0,-4,-2),
                                    (5*sec,20+3*sec,13,40),(0,0,1,0),(5,3,0,0), strict=True):
            track.states.add(center_x=x+vx*sec,center_y=y,heading=math.pi/2,
                             velocity_x=vx,velocity_y=vy,valid=True)
    feat=msg.map_features.add()
    feat.lane.type=2
    for y in range(-5,26):
        feat.lane.polyline.add(x=0,y=y)
    return msg


def test_official_current_frame_and_timestamps():
    msg=scenario()
    ident,features,target=decode(msg.SerializeToString())
    assert ident=="static-fixture"
    observed=[ActorTrack("vehicle",20,0,3,0),ActorTrack("pedestrian",13,4,0,-1),
              ActorTrack("cyclist",40,2,0,0)]
    expected=make_features([5,5,5],observed,0,True)
    np.testing.assert_allclose(features,expected,atol=1e-5)
    np.testing.assert_allclose(target,[5,5,5])
    assert features[13]==1
    msg.timestamps_seconds[30]+=.05
    assert decode(msg.SerializeToString()) is None


def test_future_changes_only_labels():
    msg=scenario()
    _,before,y=decode(msg.SerializeToString())
    for idx,track in enumerate(msg.tracks):
        for t in range(msg.current_time_index+1,len(track.states)):
            state=track.states[t]
            state.center_x=1000*idx+t
            state.center_y=-1000*idx-t
            state.velocity_x=70.
            state.velocity_y=-30.
            state.heading=-.25
    _,after,other_y=decode(msg.SerializeToString())
    np.testing.assert_array_equal(before,after)
    assert np.all(other_y!=y)


def test_missing_actor_masks_and_vehicle_selection():
    msg=scenario()
    msg.ClearField("map_features")
    for track in msg.tracks[1:]:
        track.states[10].valid=False
    _,row,_=decode(msg.SerializeToString())
    assert not row[3:].any()
    obs=[ActorTrack("vehicle",1,4,0,0),ActorTrack("vehicle",7,0,2,0),
         ActorTrack("vehicle",9,1,0,0),ActorTrack("other",1,0,0,0)]
    a=make_features([2,2.5,3],obs)
    np.testing.assert_array_equal(a[:7],[3,1,1,1,7,0,-1])
    unsupported=make_features([2,2,2],[ActorTrack("vehicle",np.nan,0,0,0)],
                              lane_bend=np.nan,lane_valid=True)
    assert not unsupported[3:].any()


def test_earlier_track_preferred_over_current_velocity():
    msg=scenario()
    msg.tracks[1].states[10].velocity_y=50
    assert decode(msg.SerializeToString())[1][6]==pytest.approx(-2)
    msg.tracks[1].states[5].valid=False
    assert decode(msg.SerializeToString())[1][6]==pytest.approx(45)


def test_portable_batch_and_scalar_predict(tmp_path):
    x=make_features([1,2,3],[ActorTrack("vehicle",8,0,0,0)])
    weight=np.zeros((_basis(x,"interaction").shape[1],3))
    weight[0]=[-4,0,5]
    model=MotionSpeedModel("interaction",weight)
    path=tmp_path/"motion-speed.npz"
    model.save(path)
    loaded=MotionSpeedModel.load(path)
    np.testing.assert_array_equal(loaded.predict(x),[0,3,8])
    np.testing.assert_array_equal(loaded.predict(np.stack([x,x])),[[0,3,8],[0,3,8]])
    with pytest.raises(ValueError,match="invalid motion features"):
        loaded.predict(np.zeros(len(FEATURE_NAMES)-1))


def test_record_and_digest_integrity(tmp_path):
    payload=scenario().SerializeToString()
    n=struct.pack("<Q",len(payload))
    content=n+struct.pack("<I",masked_crc(n))+payload+struct.pack("<I",masked_crc(payload))
    path=tmp_path/"source.tfrecord"
    path.write_bytes(content)
    job=dict(size=len(content),sha256=hashlib.sha256(content).hexdigest())
    assert list(read_staged(path,job))==[payload]
    with pytest.raises(ValueError,match="differs"):
        list(read_staged(path,{**job,"sha256":"0"*64}))
    path.write_bytes(content[:-1]+bytes([content[-1]^1]))
    with pytest.raises(ValueError,match="checksum"):
        list(read_staged(path,job))


def test_staged_split_plan_rejects_overlap_and_size(tmp_path):
    objs=[dict(name=f"uncompressed/scenario/{s}/{s}.tfrecord-00000-of-00150",size=300,
               generation="12345",sha256="0"*64) for s in ("training","validation")]
    manifest=tmp_path/"manifest.json"
    manifest.write_text(json.dumps(dict(objects=objs)))
    assert len(staged_objects(tmp_path))==2
    manifest.write_text(json.dumps(dict(objects=[objs[0],objs[0]])))
    with pytest.raises(ValueError,match="duplicate"):
        staged_objects(tmp_path)
    manifest.write_text(json.dumps(dict(objects=[objs[0],{**objs[1],"size":2**31}])))
    with pytest.raises(ValueError,match="budget"):
        staged_objects(tmp_path)
