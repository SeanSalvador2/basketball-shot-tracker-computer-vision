#!/usr/bin/env python
"""Step through pipeline-proposed shot events and verify/correct them into a labels CSV.

The semi-automatic labeling loop from docs/DATA_PROTOCOL.md §6: the pipeline PROPOSES
(outcome, zone, timestamps, location); the human confirms or fixes; the CSV becomes the
session's ground-truth label file, and every correction is a future training example.

Input: JSON from the pipeline's session report — either a top-level list of event dicts or
{"shots": [...]}. Recognized keys per event (all optional except an index/id):
  shot_id, t_release_s, t_rim_s, outcome, zone, court_xy, shot_type, miss_direction,
  make_quality, spot_id
Output: CSV with one row per shot: the DATA_PROTOCOL per-shot label schema plus `verified`
("accepted" | "corrected" | "excluded").

Interactive commands at each shot (default = accept the proposal as-is):
  [enter] accept   o <make|miss>   z <zone>   t <type>   d <short|long|left|right|...>
  q <swish|rattle|...>   p <spot_id>   x exclude (not a real attempt)   s accept all rest
Non-interactive: --accept-all writes every proposal unchanged (useful for a first pass and
for tests; you can re-run later on the shots that need attention).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

FIELDS = [
    "shot_id", "t_release_s", "t_rim_s", "outcome", "zone", "spot_id",
    "shot_type", "miss_direction", "make_quality", "court_x_m", "court_y_m", "verified",
]


def load_proposals(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    events = data["shots"] if isinstance(data, dict) and "shots" in data else data
    if not isinstance(events, list):
        raise SystemExit(f"{path}: expected a list of events or {{'shots': [...]}}")
    out = []
    for i, e in enumerate(events):
        xy = e.get("court_xy") or e.get("xy") or (None, None)
        out.append({
            "shot_id": e.get("shot_id", i),
            "t_release_s": e.get("t_release_s", e.get("release_s", "")),
            "t_rim_s": e.get("t_rim_s", e.get("rim_s", "")),
            "outcome": e.get("outcome", ""),
            "zone": e.get("zone", ""),
            "spot_id": e.get("spot_id", ""),
            "shot_type": e.get("shot_type", ""),
            "miss_direction": e.get("miss_direction", ""),
            "make_quality": e.get("make_quality", ""),
            "court_x_m": (xy[0] if xy and xy[0] is not None else ""),
            "court_y_m": (xy[1] if xy and len(xy) > 1 and xy[1] is not None else ""),
            "verified": "",
        })
    return out


def review_interactive(rows: list[dict]) -> list[dict]:
    accept_rest = False
    kept: list[dict] = []
    edit_keys = {"o": "outcome", "z": "zone", "t": "shot_type",
                 "d": "miss_direction", "q": "make_quality", "p": "spot_id"}
    for row in rows:
        if accept_rest:
            row["verified"] = row["verified"] or "accepted"
            kept.append(row)
            continue
        print(f"\nshot {row['shot_id']}  release={row['t_release_s']}  rim={row['t_rim_s']}\n"
              f"  proposed: outcome={row['outcome'] or '?'} zone={row['zone'] or '?'} "
              f"type={row['shot_type'] or '-'} dir={row['miss_direction'] or '-'} "
              f"quality={row['make_quality'] or '-'}")
        while True:
            cmd = input("  [enter]=accept  o/z/t/d/q/p <val>  x=exclude  s=accept rest > ").strip()
            if cmd == "":
                row["verified"] = row["verified"] or "accepted"
                kept.append(row)
                break
            if cmd == "x":
                row["verified"] = "excluded"
                kept.append(row)
                break
            if cmd == "s":
                accept_rest = True
                row["verified"] = row["verified"] or "accepted"
                kept.append(row)
                break
            key, _, val = cmd.partition(" ")
            if key in edit_keys and val:
                row[edit_keys[key]] = val.strip()
                row["verified"] = "corrected"
                continue
            print("  ? unrecognized — try again")
    return kept


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("proposals", type=Path, help="pipeline session-report JSON")
    ap.add_argument("-o", "--out", type=Path, required=True, help="labels CSV to write")
    ap.add_argument("--accept-all", action="store_true", help="non-interactive: accept every proposal")
    args = ap.parse_args(argv)

    rows = load_proposals(args.proposals)
    if args.accept_all:
        for r in rows:
            r["verified"] = "accepted"
    else:
        rows = review_interactive(rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    n_corr = sum(r["verified"] == "corrected" for r in rows)
    n_excl = sum(r["verified"] == "excluded" for r in rows)
    print(f"wrote {args.out}: {len(rows)} shots ({n_corr} corrected, {n_excl} excluded)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
