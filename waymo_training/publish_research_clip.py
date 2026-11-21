"""Add a verified, simulator-generated research illustration to the presentation catalog."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    media = root / "web" / "media"
    clip = media / "waymo-research-pilot"
    for line in (clip / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        path = (clip / name.strip().lstrip("*")).resolve()
        if path.parent != clip.resolve() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("Research illustration checksum mismatch")
    read = lambda p: json.loads(p.read_text(encoding="utf-8-sig"))
    entry = read(clip / "entry.json")
    run = read(clip / "run.json")
    if run["trial_id"] != entry["trial_id"] or run["cohort"] != "waymo_research" or run["outcome"] != "lane_departure":
        raise ValueError("Unexpected research run metadata")
    if run.get("benchmark_member", False) or not (clip / "WAYMO_DATASET_AGREEMENT.txt").exists():
        raise ValueError("The research illustration requires separate attribution and scope.")
    build = read(media / "build.json")
    official = build["runs"]
    if build["status"] != "evaluated" or any(v["trial_id"] == entry["trial_id"] for v in official):
        raise ValueError("An independently published CARLA benchmark is required first.")
    build["development_runs"] = [v for v in build.get("development_runs", []) if v["trial_id"] != entry["trial_id"]] + [run]
    urls = {
        "Real Waymo data · Colab results and transfer limitations": "https://github.com/yafi-s/ADO/blob/main/waymo_training/RESULTS.md",
        "Waymo-trained pilot · simulated research clip and controller report": "./media/waymo-research-pilot/report.json",
        "Waymo research illustration · non-commercial terms": "./media/waymo-research-pilot/WAYMO_DATASET_AGREEMENT.txt",
    }
    build["evidence"] = [e for e in build.get("evidence", []) if e["url"] not in urls.values()]
    build["evidence"] += [{"label": k, "url": v} for k, v in urls.items()]
    catalog = [v for v in read(media / "viewer-catalog.json") if v["trial_id"] != entry["trial_id"]]
    catalog.append({**entry, "label":"Non-commercial Waymo research · stopped lead · lane departure"})
    for name, content in (("build.json", build), ("viewer-catalog.json", catalog)):
        tmp = (media / name).with_suffix(".partial")
        tmp.write_text(json.dumps(content,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
        tmp.replace(media / name)
    print(f"Added one research illustration. Retained {len(official)} scored and supplemental records.")


if __name__ == "__main__":
    main()
