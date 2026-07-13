"""Shared label schema + CSV IO for the review CLI and the web app.

One row per shot; `verified` is "accepted" | "corrected" | "excluded" | "" (unreviewed).
Rows added by hand for detector misses carry `source` = "manual".
"""
from __future__ import annotations

import csv
from pathlib import Path

FIELDS = [
    "shot_id", "t_release_s", "t_rim_s", "outcome", "zone", "spot_id",
    "shot_type", "miss_direction", "make_quality", "court_x_m", "court_y_m",
    "verified", "source",
]


def rows_from_report(shots: list[dict]) -> list[dict]:
    """Normalize pipeline session-report shot dicts into label rows."""
    out = []
    for i, e in enumerate(shots):
        xy = e.get("court_xy") or (None, None)
        out.append({
            "shot_id": e.get("shot_id", i),
            "t_release_s": e.get("t_release_s", e.get("release_t", "")),
            "t_rim_s": e.get("t_rim_s", e.get("rim_t", "")),
            "outcome": e.get("outcome", ""),
            "zone": e.get("zone", ""),
            "spot_id": e.get("spot_id", ""),
            "shot_type": e.get("shot_type", ""),
            "miss_direction": e.get("miss_direction", ""),
            "make_quality": e.get("make_quality", ""),
            "court_x_m": "" if xy[0] is None else xy[0],
            "court_y_m": "" if len(xy) < 2 or xy[1] is None else xy[1],
            "verified": e.get("verified", ""),
            "source": e.get("source", "pipeline"),
        })
    return out


def save_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))
