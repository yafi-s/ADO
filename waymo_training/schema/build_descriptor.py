"""Rebuild e2ed.pb from the pinned Waymo source. Requires grpcio-tools and requests."""
import hashlib
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import requests

REVISION = "99a4cb3ff07e2fe06c2ce73da001f850f628e45a"
BASE = f"https://raw.githubusercontent.com/waymo-research/waymo-open-dataset/{REVISION}/src/"
FRAME = "waymo_open_dataset/protos/end_to_end_driving_data.proto"


def main():
    out = Path(__file__).with_name("e2ed.pb")
    with tempfile.TemporaryDirectory() as folder:
        todo = [FRAME]
        with requests.Session() as client:
            while todo:
                name = todo.pop()
                target = Path(folder) / name
                if target.exists():
                    continue
                response = client.get(BASE + name, timeout=30)
                response.raise_for_status()
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(response.content)
                todo.extend(re.findall(r'^import "([^"]+)"', response.text, re.MULTILINE))
        subprocess.run([sys.executable, "-m", "grpc_tools.protoc", "-I" + folder, "--include_imports",
                        "--descriptor_set_out=" + str(out), FRAME], check=True)
    print(f"{hashlib.sha256(out.read_bytes()).hexdigest()}  {out.name}")


if __name__ == "__main__":
    main()
