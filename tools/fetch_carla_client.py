"""Fetch the CPython 3.12 wheel alone from the official CARLA 1.0 Windows ZIP."""
import hashlib
from http.client import HTTPException
from pathlib import Path
import sys
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from uuid import uuid4
import zipfile
import zlib


URL = "https://downloads.carlasim.com/Windows/CARLA_1.0.zip"
# Deflated wheel payload from the pinned release ZIP's central directory and local header.
FIRST, LAST, ARCHIVE_SIZE = 13878500014, 13880294563, 13919371161
WHEEL = "carla-1.0-cp312-cp312-win_amd64.whl"
SHA256 = "07b5d86f11e47636f3886358b5d32b47435f32d06cad00723ecad44035915a61"


def download():
    # Some CDN locations return the full ZIP headers on the first request for
    # a new cache key, then serve the requested range on the second request.
    keyed_url = f"{URL}?wheel-range={FIRST}-{LAST}&request={uuid4().hex}"
    urls = [URL, keyed_url, keyed_url, URL]
    length = LAST - FIRST + 1
    for attempt, url in enumerate(urls, 1):
        request = Request(url, headers={"Range": f"bytes={FIRST}-{LAST}", "Accept-Encoding": "identity",
                                         "User-Agent": "Crosswalk/1.0"})
        try:
            with urlopen(request, timeout=45) as response:
                content_range = response.headers.get("Content-Range")
                if response.status != 206 or content_range != f"bytes {FIRST}-{LAST}/{ARCHIVE_SIZE}":
                    raise ValueError(f"HTTP {response.status}, Content-Range={content_range!r}")
                compressed = response.read(length)
                if len(compressed) != length:
                    raise ValueError(f"Truncated range: received {len(compressed)} of {length} bytes")
                wheel = zlib.decompress(compressed, -15)
                if hashlib.sha256(wheel).hexdigest() != SHA256:
                    raise ValueError("The downloaded wheel does not match the CARLA 1.0 release checksum.")
                return wheel
        except (OSError, HTTPException, ValueError, zlib.error) as error:
            detail = str(error)
            if isinstance(error, HTTPError):
                detail += f", Content-Range={error.headers.get('Content-Range')!r}"
                error.close()
            print(f"CARLA wheel request {attempt}/{len(urls)}: {detail}", file=sys.stderr, flush=True)
            if attempt == len(urls):
                raise
            time.sleep(2 ** (attempt - 1))


def fetch(directory):
    destination = directory / WHEEL
    if destination.exists():
        if hashlib.sha256(destination.read_bytes()).hexdigest() != SHA256:
            raise ValueError(f"Existing CARLA wheel has an unexpected checksum: {destination}")
        return destination
    archive = directory / "CARLA_1.0.zip"
    if archive.is_file() and archive.stat().st_size == ARCHIVE_SIZE:
        with zipfile.ZipFile(archive) as package:
            wheel = package.read(f"Carla-1.0-Win64-Shipping/PythonAPI/carla/dist/{WHEEL}")
        if hashlib.sha256(wheel).hexdigest() != SHA256:
            raise ValueError(f"The CARLA wheel in {archive} has an unexpected checksum.")
    else:
        wheel = download()
    directory.mkdir(parents=True, exist_ok=True)
    staging = destination.with_suffix(".partial")
    staging.write_bytes(wheel)
    staging.replace(destination)
    return destination


if __name__ == "__main__":
    print(fetch(Path(sys.argv[1])))
