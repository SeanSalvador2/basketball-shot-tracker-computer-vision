"""Review CLI: proposals JSON -> labels CSV (non-interactive path)."""
import csv
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_accept_all_round_trip(tmp_path):
    proposals = {
        "shots": [
            {"shot_id": 0, "t_release_s": 1.2, "t_rim_s": 2.1, "outcome": "make",
             "zone": "three", "court_xy": [0.1, 7.5]},
            {"outcome": "miss", "zone": "midrange", "miss_direction": "short",
             "court_xy": [2.0, 4.0]},
        ]
    }
    src = tmp_path / "session.json"
    src.write_text(json.dumps(proposals))
    out = tmp_path / "labels.csv"

    res = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "review_events.py"),
         str(src), "-o", str(out), "--accept-all"],
        capture_output=True, text=True, check=True,
    )
    assert "2 shots" in res.stdout

    rows = list(csv.DictReader(open(out)))
    assert len(rows) == 2
    assert rows[0]["outcome"] == "make" and rows[0]["zone"] == "three"
    assert rows[0]["court_y_m"] == "7.5" and rows[0]["verified"] == "accepted"
    assert rows[1]["shot_id"] == "1"  # auto-indexed when absent
    assert rows[1]["miss_direction"] == "short"
