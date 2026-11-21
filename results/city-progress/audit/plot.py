"""Render time-resolved native velocities by station, preserving every recorded sample."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def load(drive):
    manifest = drive / "manifest.json"
    if manifest.exists():
        return json.loads(manifest.read_bytes()), json.loads((drive / "telemetry.json").read_bytes())
    return json.loads((drive/"capture.json").read_bytes()),json.loads((drive/"raw/telemetry.json").read_bytes())


def plot(baseline, output, candidate=None):
    m,old=load(baseline)
    runs=[("v0.5",old,"#496681")]
    if candidate:
        _,new=load(candidate)
        runs.append(("Candidate",new,"#b24e1b"))
    fig,axes=plt.subplots(3,1,figsize=(12.8,8.8),layout="constrained",dpi=155)
    windows=[(0,475),(475,950),(950,1425)]
    for ax,(lo,hi) in zip(axes,windows):
        for j in m["summary"]["junction_traversals"]:
            if lo<=j["exit_m"] and hi>=j["entry_m"]:
                ax.axvspan(j["entry_m"],j["exit_m"],color="#d6dfdb",alpha=.62,linewidth=0)
        for name,rows,color in runs:
            x=[r["progress_m"] for r in rows]
            ax.plot(x,[r["speed_mps"]*3.6 for r in rows],color=color,linewidth=1.7,label=name)
        if not candidate:
            ax.plot(x,[r["speed_request_mps"]*3.6 for r in rows],color="#a0552d",linestyle="--",linewidth=1,label="Requested")
        ax.set_xlim(lo,hi)
        ax.set_ylim(-1,44)
        ax.set_yticks([0,10,20,30,40])
        ax.grid(axis="y",alpha=.22)
        ax.spines[["top","right"]].set_visible(False)
        ax.set_ylabel("km/h")
        ax.set_axisbelow(True)
    axes[-1].set_xlabel("Route station (m)")
    fig.suptitle("Measured speed along the City Life route",fontsize=18)
    handles=[Line2D([0],[0],color=color,linewidth=2,label=label) for label,_,color in runs]
    if not candidate:handles.append(Line2D([0],[0],color="#a0552d",linestyle="--",label="Requested"))
    handles.append(Patch(facecolor="#d6dfdb",label="Junction interior"))
    axes[0].legend(handles=handles,loc="upper left",frameon=False,ncol=len(handles))
    fig.savefig(output)
    plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("baseline",type=Path);p.add_argument("output",type=Path)
    p.add_argument("--candidate",type=Path); a=p.parse_args()
    plot(a.baseline,a.output,a.candidate)
