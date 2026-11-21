"""Pack replay JSON for fast browser loading without changing the original evidence."""
import argparse
import gzip
from io import TextIOWrapper
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile


FORMAT = "ado-replay-pack-v1"
CHUNK_SIZE = 64 * 1024


def _array_items(stream):
    """Read the native replay's top-level array one frame at a time."""
    decoder = json.JSONDecoder()
    buffer = ""
    offset = 0

    def read_more():
        nonlocal buffer, offset
        chunk = stream.read(CHUNK_SIZE)
        buffer, offset = buffer[offset:] + chunk, 0
        return bool(chunk)

    def next_character():
        nonlocal offset
        while True:
            while offset < len(buffer) and buffer[offset] in " \t\r\n":
                offset += 1
            if offset < len(buffer):
                return buffer[offset]
            if not read_more():
                return ""

    if next_character() != "[":
        raise ValueError("Expected a replay array")
    offset += 1
    if next_character() != "]":
        while True:
            next_character()
            while True:
                try:
                    frame, end = decoder.raw_decode(buffer, offset)
                    break
                except json.JSONDecodeError:
                    if not read_more():
                        raise
            offset = end
            yield frame
            separator = next_character()
            if separator == "]":
                break
            if separator != ",":
                raise ValueError("Expected a comma between replay frames")
            offset += 1
    offset += 1
    if next_character():
        raise ValueError("Extra content after the replay array")


def _json(value):
    return json.dumps(value, allow_nan=False, separators=(",", ":"))


def _write_pack(source, output):
    for chunk in iter(lambda: source.read(CHUNK_SIZE), ""):
        initial = chunk.lstrip(" \t\r\n")
        if initial:
            break
    else:
        raise ValueError("Empty replay JSON")
    source.seek(0)
    metadata = None
    if initial.startswith("["):
        frames = _array_items(source)
    else:
        metadata = json.load(source)
        if not isinstance(metadata, dict) or not isinstance(metadata.get("frames"), list):
            raise ValueError("Expected a replay array or an object containing frames")
        frames = metadata.pop("frames")

    output.write(f'{{"format":"{FORMAT}","replay":')
    if metadata is not None:
        output.write(_json(metadata)[:-1] + ("," if metadata else "") + '"frames":')
    output.write("[")
    shared_path = None
    ranges = []
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValueError(f"Replay frame {index} is not an object")
        observer = frame.get("observer_map")
        path = observer.get("path") if isinstance(observer, dict) else None
        if isinstance(path, list):
            serialized = _json(path)
            if shared_path is None:
                shared_path = serialized
            if serialized == shared_path:
                del observer["path"]
                if ranges and ranges[-1][1] == index:
                    ranges[-1][1] += 1
                else:
                    ranges.append([index, index + 1])
        if index:
            output.write(",")
        output.write(_json(frame))
    output.write("]")
    if metadata is not None:
        output.write("}")
    output.write(f',"shared_path":{shared_path or "null"},"path_ranges":{_json(ranges)}}}\n')


def pack(source, output=None):
    """Write a deterministic gzip file beside the replay, preserving any existing output."""
    source = Path(source)
    output = Path(output) if output is not None else source.with_name("replay-pack.json.gz")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Replay pack already exists at {output}")
    with NamedTemporaryFile(prefix=".replay-pack-", suffix=".part", dir=output.parent, delete=False) as pending:
        temporary = Path(pending.name)
    try:
        with source.open(encoding="utf-8-sig") as replay, temporary.open("wb") as file:
            with gzip.GzipFile(filename="", fileobj=file, mode="wb", mtime=0, compresslevel=6) as compressed:
                with TextIOWrapper(compressed, encoding="utf-8", newline="\n") as text:
                    _write_pack(replay, text)
        if os.name == "nt":
            temporary.rename(output)
        else:
            os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = pack(args.replay, args.output)
    print(f"{output} ({output.stat().st_size:,} bytes from {args.replay.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
