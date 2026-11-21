"""Extract exact camera evidence for earlier optical readings and all additional unmatched readings."""
import argparse
import json
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont


def panel(path, rows, columns=2):
    font=ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf",15)
    small=ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf",14)
    sheet=Image.new("RGB",(columns*400,((len(rows)+columns-1)//columns)*216),"#1b222c")
    d=ImageDraw.Draw(sheet)
    for i,(im,b,label,detail) in enumerate(rows):
        x=i%columns*400;y=i//columns*216
        w,h=im.size;l,t,r,z=[round(k*v) for k,v in zip(b,(w,h,w,h))]
        cx=(l+r)//2;cy=(t+z)//2
        crop=im.crop((cx-52,cy-33,cx+52,cy+33)).resize((184,117))
        native=im.resize((214,120))
        for img,box in [(crop,[(l-cx+52)*184/104,(t-cy+33)*117/66,(r-cx+52)*184/104,(z-cy+33)*117/66]),
                        (native,[l*214/w,t*120/h,r*214/w,z*120/h])]:
            ImageDraw.Draw(img).rectangle(box,outline="#ffd661",width=1)
        sheet.paste(native,(x,y+48));sheet.paste(crop,(x+215,y+49))
        d.text((x+5,y+5),label,fill="white",font=font);d.text((x+5,y+26),detail,fill="#cfd8e0",font=small)
        d.text((x+5,y+173),"Forward exposure",fill="#cfd8e0",font=small)
        d.text((x+220,y+173),"Optical crop",fill="#cfd8e0",font=small)
    sheet.save(path,quality=93)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("raw","report","unmatched","output"):
        parser.add_argument(name,type=Path)
    parser.add_argument("--targets",nargs="+",help="Optional reference IDs for the first-observation figure")
    parser.add_argument("--max-misses",type=int,default=24)
    args=parser.parse_args()
    report=json.loads(args.report.read_bytes())
    f0=json.loads((args.raw/"perception.jsonl").open().readline())["frame"]
    changed=[]
    full,combined=("same_jpeg_full","same_jpeg_multiscale") if "same_jpeg_full" in report["bands"] else (
        "native_full_only","native_optical")
    for a in report["encounters"]:
        if args.targets and str(a["id"]) not in args.targets:
            continue
        before=a["first"].get(full);after=a["first"].get(combined)
        if before and after and before["frame"]!=after["frame"]:
            for source,r in [("Full frame",before),("Full frame + horizon",after)]:
                changed.append((r["frame"],r["box"],f"{source} · {a['label']} {a['id']}",
                                f"Frame {r['frame']} · {r['range_m']:.2f} m · {a['role']}"))
    negatives=[json.loads(line) for line in args.unmatched.open()]
    def key(r):return (r["frame"],r["label"],tuple(r["box"]))
    base={key(r) for r in negatives if r["stage"]==full}
    misses=[(r["frame"],r["box"],f"Unmatched · {r['label']}",f"Frame {r['frame']}") for r in negatives
            if r["stage"]==combined and key(r) not in base]
    if len(misses)>args.max_misses:
        misses=[misses[round(i*(len(misses)-1)/max(1,args.max_misses-1))] for i in range(args.max_misses)]
    selections={r[0] for r in changed+misses}; images={}
    if not selections:return
    args.output.mkdir(parents=True,exist_ok=True)
    camera=cv2.VideoCapture(str(args.raw/"front.mjpeg"))
    try:
        for frame in range(f0,max(selections)+1):
            if not camera.grab():raise ValueError(f"Truncated front stream at {frame}")
            if frame in selections:
                images[frame]=Image.fromarray(cv2.cvtColor(camera.retrieve()[1],cv2.COLOR_BGR2RGB))
    finally:camera.release()
    for name,rows in [("first-optical.jpg",changed),("unmatched-optical.jpg",misses)]:
        if rows:panel(args.output/name,[(images[f],*details) for f,*details in rows])


if __name__=="__main__":main()
