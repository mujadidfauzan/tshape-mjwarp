"""Atomic snapshots and append-only reports."""
import json
import math
from pathlib import Path


def nullable(data):
    if isinstance(data, float) and not math.isfinite(data):
        return None
    if isinstance(data, dict):
        return {k: nullable(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [nullable(v) for v in data]
    return data


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(nullable(data), indent=2, allow_nan=False)+"\n")
    tmp.replace(path)


def append_jsonl(path, data):
    with Path(path).open("a") as f:
        f.write(json.dumps(nullable(data), allow_nan=False)+"\n")
