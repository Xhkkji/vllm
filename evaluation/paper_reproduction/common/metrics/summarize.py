"""Summarize metric JSON records into one CSV table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .schema import METRIC_FIELDS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path,
                        help="directory containing metric JSON files")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.input.glob("*.json")):
        payload = json.loads(path.read_text())
        rows.append({field: payload.get(field) for field in METRIC_FIELDS})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
