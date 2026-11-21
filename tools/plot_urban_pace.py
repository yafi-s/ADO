"""Render the v0.4 native speed and stopping comparisons from recorded frame logs."""
from pathlib import Path
import gzip
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TEAL = "#007a73"
GRAY = "#677589"
ORANGE = "#b75619"


def trace(relative):
    with gzip.open(RESULTS / relative, "rt", encoding="utf8") as stream:
        return [json.loads(line) for line in stream]


def curve(ax, rows, key, name, color, style="-"):
    ax.plot([r[key] for r in rows], [3.6*r["speed_mps"] for r in rows], style,
            label=name, color=color, linewidth=2.25, solid_capstyle="round")


def main():
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titlelocation":"left", "axes.titleweight":"bold", "axes.labelcolor":"#344256"})
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 8.1), layout="constrained")
    for ax in axes:
        ax.grid(axis="y", color="#e1e6eb", linewidth=.8)
        ax.set_axisbelow(True)
        ax.set_ylabel("Speed (km/h)")
        ax.set_ylim(-1.5,44)
        ax.set_yticks([0,10,20,30,40])
        ax.margins(x=.01)
    ax=axes[0]
    curve(ax,trace("city-pace/boulevard/city-cruise/trace.jsonl.gz"),"progress_m","v0.3",GRAY)
    curve(ax,trace("city-pace/boulevard/city-pace/trace.jsonl.gz"),"progress_m","v0.4 pace fit",TEAL)
    ax.set(title="Cruising on the same empty boulevard",xlabel="Route progress (m)")
    ax.legend(loc="lower right",frameon=False,ncol=2)
    ax=axes[1]
    before=trace("city-pace/signed-transition/without-preview/trace.jsonl.gz")
    after=trace("city-pace/signed-transition/with-preview/trace.jsonl.gz")
    curve(ax,before,"progress_m","Current limit only",GRAY)
    curve(ax,after,"progress_m","Upcoming sign included",TEAL)
    entry=next(row["progress_m"] for i,row in enumerate(after) if i and row["posted_limit_kmh"]==30 and after[i-1]["posted_limit_kmh"]>30)
    ax.axvline(entry,linewidth=1.3,color=ORANGE,linestyle=":")
    ax.annotate("30 km/h zone",xy=(entry,3),xytext=(-6,0),textcoords="offset points",ha="right",color=ORANGE)
    ax.set(title="Slowing before the lower speed limit",xlabel="Route progress (m)",xlim=(205,286))
    ax.legend(loc="lower left",frameon=False)
    ax=axes[2]
    curve(ax,trace("workzone-stop/curbside.jsonl.gz"),"sim_time","Board at curb",GRAY)
    curve(ax,trace("workzone-stop/closed-then-open.jsonl.gz"),"sim_time","Board initially in lane",TEAL)
    ax.axvspan(17,20,color=ORANGE,alpha=.13)
    ax.text(18.5,35,"Board moved",ha="center",va="bottom",fontsize=9,color=ORANGE)
    ax.set(title="Stopping and pulling away in the work zone",xlabel="Simulation time (s)")
    ax.legend(loc="upper left",frameon=False,ncol=2)
    fig.savefig(RESULTS/"urban-pace"/"speed-traces.png",dpi=155)
    plt.close(fig)


if __name__=="__main__":
    main()
