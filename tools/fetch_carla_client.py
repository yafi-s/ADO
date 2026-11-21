"""Fetch the CPython 3.12 wheel alone from the official CARLA 1.0 Windows ZIP."""
import hashlib
from pathlib import Path
import sys
from urllib.request import Request, urlopen
import zlib


URL = "https://downloads.carlasim.com/Windows/CARLA_1.0.zip"
# Deflated wheel payload from the pinned release ZIP's central directory and local header.
FIRST, LAST, ARCHIVE_SIZE = 13878500014, 13880294563, 13919371161
WHEEL = "carla-1.0-cp312-cp312-win_amd64.whl"
SHA256 = "07b5d86f11e47636f3886358b5d32b47435f32d06cad00723ecad44035915a61"


def fetch(directory):
    destination = directory / WHEEL
    if destination.exists():
        if hashlib.sha256(destination.read_bytes()).hexdigest() != SHA256:
            raise ValueError(f"Existing CARLA wheel has an unexpected checksum: {destination}")
        return destination
    request = Request(URL, headers={"Range": f"bytes={FIRST}-{LAST}", "User-Agent": "Crosswalk/1.0"})
    with urlopen(request, timeout=60) as response:
        if response.status != 206 or response.headers.get("Content-Range") != f"bytes {FIRST}-{LAST}/{ARCHIVE_SIZE}":
            raise ValueError("The CARLA server did not return the pinned wheel byte range.")
        wheel = zlib.decompress(response.read(LAST - FIRST + 1), -15)
    if hashlib.sha256(wheel).hexdigest() != SHA256:
        raise ValueError("The downloaded wheel does not match the CARLA 1.0 release checksum.")
    directory.mkdir(parents=True, exist_ok=True)
    staging = destination.with_suffix(".partial")
    staging.write_bytes(wheel)
    staging.replace(destination)
    return destination


if __name__ == "__main__":
    print(fetch(Path(sys.argv[1])))
