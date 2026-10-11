"""Compare a regenerated JSON(.gz) run with its archived numerical record.

Ignores only the root-level elapsed-time field `segundos` and gzip metadata.
All other keys and list lengths must match. Counts, flags, strings and integers
must match exactly; floating-point tolerance is explicit and configurable.
"""
import argparse
import gzip
import json
import math
from pathlib import Path


def read(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)


def compare(left, right, *, rtol, atol, path="$"):
    if isinstance(left, dict) and isinstance(right, dict):
        ignored = {"segundos"} if path == "$" else set()
        assert left.keys()-ignored == right.keys()-ignored, f"{path}: different keys"
        for key in left.keys()-ignored:
            compare(left[key], right[key], rtol=rtol, atol=atol, path=f"{path}.{key}")
    elif isinstance(left, list) and isinstance(right, list):
        assert len(left) == len(right), f"{path}: different lengths"
        for i, (a, b) in enumerate(zip(left, right)):
            compare(a, b, rtol=rtol, atol=atol, path=f"{path}[{i}]")
    elif isinstance(left, float) and isinstance(right, (float, int)) and not isinstance(right, bool):
        assert math.isclose(left, right, rel_tol=rtol, abs_tol=atol), f"{path}: {left} != {right}"
    else:
        assert type(left) is type(right) and left == right, f"{path}: {left!r} != {right!r}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archived", type=Path)
    parser.add_argument("regenerated", type=Path)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--atol", type=float, default=1e-8)
    args = parser.parse_args()
    assert args.rtol >= 0 and args.atol >= 0
    compare(read(args.archived), read(args.regenerated), rtol=args.rtol, atol=args.atol)
    print(f"PASS: numerical records agree (rtol={args.rtol:g}, atol={args.atol:g}); elapsed time excluded")


if __name__ == "__main__":
    main()
