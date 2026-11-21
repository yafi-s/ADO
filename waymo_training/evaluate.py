"""Score fixed checkpoints in meters on the recorded source-segment split."""
import argparse
import json
from pathlib import Path

import numpy as np

from .model import MODES, configure_device, file_hash, load_data, make_model, predict, prepare_arrays


def write_json(path, obj):
    path = Path(path)
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n", "utf8")
    pending.replace(path)


def scores(errors):
    return {
        "ADE_m": float(errors.mean()),
        "FDE_m": float(errors[:, -1].mean()),
        **{f"displacement_{s}s_m": float(errors[:, 4 * s - 1].mean()) for s in (1, 3, 5)},
        "median_window_ADE_m": float(np.median(errors.mean(axis=1))),
    }


def grouped_summary(values, groups):
    _, inv = np.unique(groups, return_inverse=True)
    means = np.bincount(inv, weights=values) / np.bincount(inv)
    out = {"segments": len(means), "mean_segment_delta_m": float(means.mean()),
           "mean_window_delta_m": float(np.mean(values))}
    if len(means) >= 2:
        rng = np.random.default_rng(2026)
        bootstrap = [rng.choice(means, len(means), replace=True).mean() for _ in range(2000)]
        out["segment_bootstrap_95pct_m"] = np.percentile(bootstrap, [2.5, 97.5]).tolist()
    return out


def different_group_images(rows, groups, seed=912):
    # Preserve the complete RGB triplet and its marginal distribution when feasible.
    rng = np.random.default_rng(seed)
    order = rng.permutation(rows)
    order = order[np.argsort(groups[order], kind="stable")]
    unique, counts = np.unique(groups[rows], return_counts=True)
    if len(unique) < 2 or counts.max() * 2 > len(rows):
        return None
    mapped = dict(zip(order.tolist(), np.roll(order, int(counts.max())).tolist()))
    return np.array([mapped[i] for i in rows])


def evaluate(data, run, config):
    denom = prepare_arrays(data, config["scales"])
    output = {
        "unit": "meters", "dataset_source_sha256": config["dataset_source_sha256"],
        "coordinate_system": "ego frame, +x forward, +y left",
        "task": "open-loop prediction of recorded ego trajectory; descriptive displacement, not official RFS",
        "seed_summary": "arithmetic mean of per-seed errors, not a prediction ensemble",
        "paired_deltas": "negative favors first model; resample whole held-out segments conditional on the fitted seeds",
        "weights": {}, "splits": {},
    }
    trained = {}
    for mode in MODES:
        for seed in config["seeds"]:
            path = run / f"checkpoint-{mode}-{seed}.weights.h5"
            model = make_model(mode, seed)
            model.load_weights(path)
            trained[mode, seed] = model
            output["weights"][f"{mode}-{seed}"] = {"sha256": file_hash(path), "parameters": model.count_params()}
    for split in ("dev", "test"):
        rows = np.flatnonzero(data["split"] == split)
        if len(rows) == 0:
            raise ValueError(f"No {split} examples")
        groups = data["group"][rows]
        future = data["future"][rows]
        swapped = different_group_images(rows, data["group"])
        cv = np.linalg.norm(data["cv"][rows] - future, axis=-1)
        report = {"windows": len(rows), "segments": len(np.unique(groups)),
                  "constant_velocity": scores(cv), "models": {}, "seed_mean": {},
                  "paired_ADE": {}}
        means = {}
        for mode in MODES:
            across_seeds = []
            swaps = []
            for seed in config["seeds"]:
                model = trained[mode, seed]
                predictions = predict(model, data, rows, mode, denom)
                errors = np.linalg.norm(predictions - future, axis=-1)
                entry = scores(errors)
                across_seeds.append(errors)
                if mode == "vision" and swapped is not None:
                    shuffled = predict(model, data, rows, mode, denom, image_rows=swapped)
                    shuffled_errors = np.linalg.norm(shuffled - future, axis=-1)
                    swaps.append(shuffled_errors)
                    entry["unrelated_camera_triplets"] = {
                        **scores(shuffled_errors),
                        "mean_waypoint_change_m": float(np.linalg.norm(predictions - shuffled, axis=-1).mean()),
                        "ADE_increase_m": float(shuffled_errors.mean() - errors.mean()),
                    }
                report["models"][f"{mode}-{seed}"] = entry
            means[mode] = np.mean(across_seeds, axis=0)
            report["seed_mean"][mode] = scores(means[mode])
            if swaps:
                shuffled_window_delta = (np.mean(swaps, axis=0) - means[mode]).mean(axis=1)
                report["paired_ADE"]["unrelated_images_minus_original"] = grouped_summary(shuffled_window_delta, groups)
        for label, delta in (("vision_minus_history", means["vision"] - means["history"]),
                             ("vision_minus_cv", means["vision"] - cv),
                             ("history_minus_cv", means["history"] - cv)):
            report["paired_ADE"][label] = grouped_summary(delta.mean(axis=1), groups)
        by_intent = {}
        for intent, name in enumerate(("unknown", "straight", "left", "right")):
            selected = data["intent"][rows] == intent
            if selected.any():
                by_intent[name] = {"windows": int(selected.sum()), "segments": len(np.unique(groups[selected])),
                                   **{k: scores(v[selected]) for k, v in {"constant_velocity": cv, **means}.items()}}
        report["by_intent_seed_mean"] = by_intent
        output["splits"][split] = report
    write_json(run / "evaluation.json", output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    args = parser.parse_args()
    configure_device(args.device)
    config = json.loads((args.run / "config.json").read_text("utf8"))
    data = load_data(args.features)
    if data["sources"] != config["sources"]:
        raise ValueError("Feature snapshot differs from training. Use the same shard set.")
    result = evaluate(data, args.run, config)
    print(json.dumps(result["splits"]["test"]["seed_mean"], indent=2))


if __name__ == "__main__":
    main()
