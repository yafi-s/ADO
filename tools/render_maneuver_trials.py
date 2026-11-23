"""Join matched baseline and candidate native maneuver films on one elapsed clock."""

import argparse
from contextlib import ExitStack
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np


FPS = 10
WIDTH, HEIGHT = 640, 360
HEADER, FOOTER = 42, 78
TITLES = {
    "cones": "Cone group blocking the route lane",
    "board": "Rigid board blocking the route lane",
    "slow": "Slow lead with a passing lane",
    "blocked": "Slow lead with the passing lane occupied",
    "no-lead": "Open road without a lead",
    "departing": "Lead accelerates away",
    "clear-bend": "Open bend",
    "inside-curb": "Board inside the curb",
    "outside-curb": "Board beyond the curb",
    "person": "Occluded pedestrian crossing",
    "reopened": "Blocked single lane reopens",
}
SELECTED = ("cones", "board", "slow", "blocked", "person", "reopened")


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class Arm:
    def __init__(self, root, case, label):
        folder = root / case
        self.name = label
        self.file = folder / "front.mp4"
        self.summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        with gzip.open(folder / "trace.jsonl.gz", "rt", encoding="utf-8") as stream:
            self.rows = [json.loads(line) for line in stream]
        self.digest = sha256(self.file)
        if self.summary["case"] != case or self.digest != self.summary["front_sha256"] or not self.rows:
            raise ValueError(f"Film manifest mismatch in {folder}")
        first = self.rows[0]["frame"]
        if any(row["frame"] != first + i or abs(row["sim_time"] - i / FPS) > 1e-6
               for i, row in enumerate(self.rows)):
            raise ValueError(f"Trace has a discontinuous native clock in {folder}")

    def open(self, stack):
        # Limit decoder workers independently of the two-arm encoder.
        self.video = cv2.VideoCapture(str(self.file), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
        stack.callback(self.video.release)
        properties = (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT,
                      cv2.CAP_PROP_FPS, cv2.CAP_PROP_FRAME_COUNT)
        actual = tuple(self.video.get(prop) for prop in properties)
        if not self.video.isOpened() or actual != (WIDTH, HEIGHT, FPS, len(self.rows)):
            raise ValueError(f"Film does not match its native trace in {self.file}")
        self.image = None

    def pixels(self, index):
        if index < len(self.rows):
            ok, self.image = self.video.read()
            if not ok:
                raise ValueError(f"Exposure {index} could not be decoded in {self.file}")
        return self.image

    def manifest(self):
        return dict(label=self.name, outcome=self.summary["outcome"], reached_end=self.summary["reached_end"],
                    capture_complete=self.summary["capture_complete"], source_frames=len(self.rows),
                    first_native_frame=self.rows[0]["frame"], last_native_frame=self.rows[-1]["frame"],
                    first_s=self.rows[0]["native_s"], last_s=self.rows[-1]["native_s"],
                    source_film_sha256=self.digest, model_sha256=self.summary["model_sha256"],
                    execution_source_sha256=self.summary["execution_source_sha256"])


def text(image, value, at, size, color=(234, 232, 228)):
    cv2.putText(image, value, at, cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)


def driving_caption(row, geometry):
    lane = row.get("ego_native_lane") or {}
    if lane.get("road_id") == geometry["road_id"] and lane.get("lane_id") == geometry["lane_id"]:
        location = "Route lane"
    elif lane.get("road_id") == geometry["road_id"] and lane.get("lane_id") == geometry["lane_id"] + 1:
        location = "Passing lane"
    else:
        location = "Approach"
    action = {"yield": "Yielding for a gap", "signal_left": "Signaling left",
              "change_left": "Changing left", "passing": "Passing",
              "wait_rejoin": "Checking the return gap", "signal_right": "Signaling right",
              "change_right": "Returning right", "rejoined": "Back on route",
              "resumed_following": "Following again"}.get(row.get("tactical_pass", {}).get("maneuver_phase"))
    return f"{location} | {action}" if action else location


def render_case(command, arms, case, start_frame, trim_tail):
    count = len(arms[1].rows) if trim_tail else max(len(arm.rows) for arm in arms)
    with ExitStack() as stack:
        for arm in arms:
            arm.open(stack)
        for i in range(count):
            screen = np.full((HEIGHT + HEADER + FOOTER, WIDTH * 2, 3), (28, 27, 25), dtype=np.uint8)
            text(screen, TITLES[case], (15, 29), .74)
            text(screen, f"{i / FPS:05.1f} s from trial start", (1030, 29), .57)
            for j, arm in enumerate(arms):
                x = j * WIDTH
                screen[HEADER:HEADER + HEIGHT, x:x + WIDTH] = arm.pixels(i)
                row = arm.rows[min(i, len(arm.rows) - 1)]
                text(screen, arm.name, (x + 15, 427), .68)
                text(screen, f'road s {row["native_s"]:.1f} m', (x + 486, 427), .54)
                if i >= len(arm.rows):
                    reason = "window complete" if arm.summary["reached_end"] else arm.summary["outcome"].replace("_", " ")
                    status = f"Final exposure held at {(len(arm.rows) - 1) / FPS:.1f} s | {reason}"
                else:
                    status = f'{row["speed_mps"] * 3.6:.1f} km/h'
                    if row.get("speed_request_mps") is not None:
                        status += f' | requested {row["speed_request_mps"] * 3.6:.1f} km/h'
                    if row["collision_count"]:
                        status += f' | contacts {row["collision_count"]}'
                text(screen, status, (x + 15, 453), .55)
                text(screen, driving_caption(row, arm.summary["geometry"]), (x + 15, 474), .48, (174, 183, 190))
            command.write(screen.tobytes())
    return dict(case=case, title=TITLES[case], start_frame=start_frame, frames=count,
                geometry=arms[0].summary["geometry"], arms=[arm.manifest() for arm in arms])


def render(baseline, candidate, output, cases, labels, trim_tail=False):
    metadata = output.with_suffix(".json")
    if output.exists() or metadata.exists():
        raise FileExistsError(f"Choose a fresh movie and metadata location: {output}")
    cv2.setNumThreads(1)
    plans = []
    for case in cases:
        arms = [Arm(root, case, label) for root, label in zip((baseline, candidate), labels, strict=True)]
        a, b = (arm.summary for arm in arms)
        if any(a[key] != b[key] for key in ("seed", "geometry", "sensor_configuration")):
            raise ValueError(f"The two arms for {case} used different setup or sensors")
        plans.append((case, arms))
    output.parent.mkdir(parents=True, exist_ok=True)
    size = f"{WIDTH * 2}x{HEIGHT + HEADER + FOOTER}"
    command = ["ffmpeg", "-nostdin", "-v", "error", "-n", "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", size, "-framerate", str(FPS), "-i", "-", "-c:v", "libx264", "-threads", "2",
               "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "18", "-movflags", "+faststart", str(output)]
    chapters = []
    total = 0
    with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
        try:
            for case, arms in plans:
                chapter = render_case(encoder.stdin, arms, case, total, trim_tail)
                chapters.append(chapter)
                total += chapter["frames"]
        finally:
            encoder.stdin.close()
        if encoder.wait():
            raise RuntimeError("Comparison film encoding failed")
    info = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-threads", "1", "-select_streams", "v:0",
        "-count_frames", "-show_entries", "stream=r_frame_rate,nb_read_frames", "-of", "json", str(output)]))
    stream, = info["streams"]
    if stream["r_frame_rate"] != f"{FPS}/1" or int(stream["nb_read_frames"]) != total:
        raise ValueError("Comparison film frame count differs from the matched traces")
    with metadata.open("x", encoding="utf-8") as stream:
        json.dump(dict(fps=FPS, total_frames=total, chapters=chapters, sha256=sha256(output),
                       renderer_sha256=sha256(Path(__file__)), trimmed_to_candidate=trim_tail), stream, indent=2)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cases", nargs="+", choices=TITLES, default=SELECTED)
    parser.add_argument("--trim-tail", action="store_true", help="End each matched clip at the final candidate exposure")
    parser.add_argument("--baseline-label", default="Baseline")
    parser.add_argument("--candidate-label", default="Candidate")
    args = parser.parse_args()
    if args.output.suffix.lower() != ".mp4":
        parser.error("The comparison output must use an .mp4 extension")
    if len(set(args.cases)) != len(args.cases):
        parser.error("Each requested comparison case must appear once")
    render(args.baseline, args.candidate, args.output, args.cases, (args.baseline_label, args.candidate_label), args.trim_tail)


if __name__ == "__main__":
    main()
