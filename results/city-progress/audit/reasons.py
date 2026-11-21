"""Localize the candidate's recorded speed-limiting conditions to the fixed audit windows."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


def reasons(stage, score):
    result=json.loads(score.read_bytes());cap=json.loads((stage/"capture.json").read_bytes())
    trace=json.loads((stage/"raw/telemetry.json").read_bytes())
    pack=stage/"policy-observations.npz"
    with pack.open("rb") as stream:
        digest=hashlib.file_digest(stream,"sha256").hexdigest()
    if digest!=cap["policy_observations_sha256"]:raise ValueError("Unsealed feature tape")
    with np.load(pack,allow_pickle=False) as tape:
        np.testing.assert_array_equal(tape["frame"],[r["frame"] for r in trace])
        np.testing.assert_allclose(tape["speed_request_mps"],[r["speed_request_mps"] for r in trace],atol=.0001)
        names=tape["feature_names"].tolist()
        X=tape["features"]
    rows=[]
    for i,r in enumerate(trace):
        d=r["progress_evidence"]
        rows.append(dict(binding=d["binding"],station=r["progress_m"],time=r["sim_time"],frame=r["frame"],
                         ego_kmh=r["speed_mps"]*3.6,request_kmh=r["speed_request_mps"]*3.6,
                         inherited_kmh=d["base_request_mps"]*3.6,proposal_kmh=d["proposal_mps"]*3.6,
                         sight_m=d["sight_m"],corridor_m=d["corridor_m"],cross_sight_m=d["cross_sight_m"],
                         blind_end_m=d["blind_end_m"],margin_m=d["adjacent_margin_m"],
                         governing_gap_m=float(X[i,names.index("mapped_lamp_gap_m")]),
                         ego_camera_red=float(X[i,names.index("rgb_lamp_red_probability")])))
    def stats(seq):
        if not seq:return {"frames":0}
        sign=[r for r in seq if r["governing_gap_m"]<59]
        # The candidate always preserves the original when its proposed release fails to exceed it.
        inherited=sum(r["request_kmh"]<=r["inherited_kmh"]+.05 for r in seq)
        return dict(frames=len(seq),seconds=round(.1*len(seq),1),
                    median={k:round(float(np.median([r[k] for r in seq])),3)
                            for k in ("ego_kmh","request_kmh","inherited_kmh","sight_m","corridor_m","cross_sight_m","blind_end_m")},
                    binding_counts=dict(Counter(r["binding"] for r in seq)),inherited_target_frames=inherited,
                    modeled_governing_lamp_frames=len(sign),
                    sensor_red_range=[round(min(r["ego_camera_red"] for r in sign),3),round(max(r["ego_camera_red"] for r in sign),3)] if sign else None,
                    first=seq[0]["frame"],last=seq[-1]["frame"])
    ans={}
    for name in ("junctions","sections","station_bins","under_15_spells"):
        ans[name]=[]
        for loc in result[name]:
            if name=="under_15_spells": selection=[r for r in rows if loc["from_s"]<=r["time"]<=loc["to_s"]]
            else: selection=[r for r in rows if loc["from_m"]<=r["station"]<loc["to_m"]]
            ans[name].append(dict(from_m=loc["from_m"],to_m=loc["to_m"],**({"id":loc["id"]} if "id" in loc else {}),
                                  all=stats(selection),slow=stats([r for r in selection if r["ego_kmh"]<15])))
    return dict(source=cap["capture_id"],feature_sha256=digest,
                note="Binding labels describe the candidate extension's evaluation. When its proposal is below the inherited target it returns the inherited target. Governing gap <59 m is the driver stopline feature, distinct from similarly oriented green lamps in the visual display.",
                whole=stats(rows),**ans)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("stage",type=Path)
    p.add_argument("score",type=Path);p.add_argument("output",type=Path);a=p.parse_args()
    a.output.write_text(json.dumps(reasons(a.stage,a.score),indent=2)+"\n",encoding="utf8")
