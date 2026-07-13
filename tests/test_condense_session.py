"""Condense tool: window merging, negative gaps, and cv2-engine extraction."""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "condense_session.py"


def _write_video(path, seconds=30, fps=20, size=(64, 48)):
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(seconds * fps):
        frame = np.full((size[1], size[0], 3), (i * 3) % 255, dtype=np.uint8)
        out.write(frame)
    out.release()


def test_condense_events_and_negatives(tmp_path):
    video = tmp_path / "session.mp4"
    _write_video(video, seconds=30)
    proposals = tmp_path / "report.json"
    proposals.write_text(json.dumps({"shots": [
        {"t_release_s": 5.0, "t_rim_s": 6.5},
        {"t_release_s": 7.0, "t_rim_s": 8.0},   # overlaps previous window -> merged
        {"t_release_s": 24.0, "t_rim_s": 25.0},
    ]}))
    outdir = tmp_path / "out"

    res = subprocess.run(
        [sys.executable, str(SCRIPT), str(video), str(proposals), "-o", str(outdir),
         "--pre", "2", "--post", "1", "--negatives", "6", "--neg-chunk", "3",
         "--engine", "cv2"],
        capture_output=True, text=True, check=True,
    )
    manifest = json.loads((outdir / "condense_manifest.json").read_text())
    events = [c for c in manifest["clips"] if c["kind"] == "event"]
    negs = [c for c in manifest["clips"] if c["kind"] == "negative"]
    assert len(events) == 2, res.stdout          # first two proposals merged into one clip
    assert events[0]["source_t0_s"] == 3.0 and events[0]["source_t1_s"] == 9.0
    assert len(negs) >= 1
    for c in manifest["clips"]:                  # clips exist and are non-trivial
        f = outdir / c["file"]
        assert f.exists() and f.stat().st_size > 0
        # negatives never overlap event windows
        if c["kind"] == "negative":
            for e in events:
                assert c["source_t1_s"] <= e["source_t0_s"] or c["source_t0_s"] >= e["source_t1_s"]
    assert "Delete the original ONLY after review" in res.stdout
