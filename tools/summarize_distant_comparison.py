"""Compare exact actor/frame pairs in the two stages of a matched forward-awareness experiment."""
import argparse
import json
from pathlib import Path


def pairs(actor, stage):
    step=1 if stage=="display" else 2
    return {n for start,end in actor.get("match_runs",{}).get(stage,[]) for n in range(start,end+1,step)}


def compare(baseline, candidate, paired_jpeg=False):
    for key in ("frames","first_frame","last_frame","front_acquisitions","projection","registered_parked_sha256"):
        if baseline.get(key)!=candidate.get(key):
            raise ValueError(f"Exposure or reference mismatch in {key}")
    before={r["id"]:r for r in baseline["encounters"]}
    after={r["id"]:r for r in candidate["encounters"]}
    if before.keys()!=after.keys():
        raise ValueError("Different reference actors in paired streams")
    optical_before="same_jpeg_full" if paired_jpeg else "native_full_only"
    optical_after="same_jpeg_multiscale" if paired_jpeg else "native_optical"
    result={"schema":1,"frames":candidate["frames"],"front_acquisitions":candidate["front_acquisitions"],
            "candidate_journal_sha256":candidate["journal_sha256"],
            "baseline_journal_sha256":baseline["journal_sha256"],"projection":candidate["projection"],
            "stages":{},"encounters":[]}
    for label,lstage,rstage,left in [("optical",optical_before,optical_after,candidate),
                                     ("display","display","display",baseline)]:
        l=left["bands"][lstage];r=candidate["bands"][rstage]
        if {k:v["opportunities"] for k,v in l.items()}!={k:v["opportunities"] for k,v in r.items()}:
            raise ValueError(f"Different {label} denominators")
        result["stages"][label]={"bands":{k:dict(opportunities=r[k]["opportunities"],
            before=l[k]["matched"],after=r[k]["matched"]) for k in l},
            "unmatched_before":left["unmatched"].get(lstage,{}).get("total",0),
            "unmatched_after":candidate["unmatched"].get(rstage,{}).get("total",0),
            "gained_actor_frames":0,"lost_actor_frames":0}
    for key,a in after.items():
        b=before[key]
        row=dict(id=key,label=a["label"],role=a["role"])
        if not paired_jpeg and pairs(b,"native_optical")!=pairs(a,"native_full_only"):
            raise ValueError("Ablation changed the optical baseline")
        for name,old,new,la,ra in [("optical",optical_before,optical_after,a,a),("display","display","display",b,a)]:
            prev,current=pairs(la,old),pairs(ra,new)
            stage=result["stages"][name]
            gained,lost=len(current-prev),len(prev-current)
            stage["gained_actor_frames"]+=gained
            stage["lost_actor_frames"]+=lost
            row[name]=dict(first_before=la["first"].get(old),first_after=ra["first"].get(new),
                           gains=gained,losses=lost)
        row["first_camera_only_callout"]=a["first"].get("native_front_callout")
        result["encounters"].append(row)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("baseline","candidate","output"):
        p.add_argument(name,type=Path)
    p.add_argument("--paired-jpeg",action="store_true")
    args=p.parse_args()
    result=compare(json.loads(args.baseline.read_bytes()),json.loads(args.candidate.read_bytes()),args.paired_jpeg)
    with args.output.open("w",encoding="utf8",newline="\n") as f:
        json.dump(result,f,indent=2,ensure_ascii=False)
        f.write("\n")
