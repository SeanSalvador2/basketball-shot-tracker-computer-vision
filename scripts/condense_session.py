#!/usr/bin/env python
"""Condense a session recording to shot-event clips + negative samples (storage saver).

Record the whole session (docs/DATA_PROTOCOL.md); afterwards, run the pipeline to get event
proposals, then condense: keep a generous window around every proposed attempt plus a budget
of shot-free footage (negatives — required for honest false-positive rates), and only then
consider deleting the original. Typical shrink is 5-10x.

SAFETY: this tool never deletes the source. Delete the original yourself only after the
review pass (scripts/review_events.py) confirms the clip count matches your session's shot
script — a detector false NEGATIVE that got condensed away is unrecoverable.

Clip timestamps are preserved in the manifest (source_t0_s per clip), so labels and session
timelines still align after condensing.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def load_windows(proposals: Path, pre_s: float, post_s: float) -> list[tuple[float, float]]:
    data = json.loads(proposals.read_text())
    events = data["shots"] if isinstance(data, dict) and "shots" in data else data
    wins = []
    for e in events:
        t0 = e.get("t_release_s", e.get("release_s"))
        t1 = e.get("t_rim_s", e.get("rim_s", t0))
        if t0 is None and t1 is None:
            continue
        t0 = float(t0 if t0 is not None else t1)
        t1 = float(t1 if t1 is not None else t0)
        wins.append((max(0.0, t0 - pre_s), t1 + post_s))
    wins.sort()
    merged: list[tuple[float, float]] = []
    for a, b in wins:
        if merged and a <= merged[-1][1] + 1.0:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def video_duration_s(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        return float(n / fps) if fps > 0 else 0.0
    finally:
        cap.release()


def negative_gaps(
    wins: list[tuple[float, float]], duration: float, budget_s: float, chunk_s: float
) -> list[tuple[float, float]]:
    """Longest shot-free gaps, cut into chunks, until the negatives budget is spent."""
    edges = [0.0] + [t for w in wins for t in w] + [duration]
    gaps = [(edges[i], edges[i + 1]) for i in range(0, len(edges) - 1, 2)]
    gaps = sorted((g for g in gaps if g[1] - g[0] > 3.0),
                  key=lambda g: g[1] - g[0], reverse=True)
    out, left = [], budget_s
    for a, b in gaps:
        t = a
        while left > 3.0 and t < b - 3.0:
            end = min(b, t + min(chunk_s, left))
            out.append((t, end))
            left -= end - t
            t = end
    return out


def extract(src: Path, t0: float, t1: float, dst: Path, engine: str) -> None:
    if engine == "ffmpeg":
        cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-ss", f"{t0:.3f}",
               "-to", f"{t1:.3f}", "-i", str(src), "-c", "copy", str(dst)]
        subprocess.run(cmd, check=True)
        return
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t0 * fps)))
    for _ in range(int(round((t1 - t0) * fps))):
        ok, frame = cap.read()
        if not ok:
            break
        out.write(frame)
    cap.release()
    out.release()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("video", type=Path)
    ap.add_argument("proposals", type=Path, help="pipeline session-report JSON")
    ap.add_argument("-o", "--outdir", type=Path, required=True)
    ap.add_argument("--pre", type=float, default=8.0, help="seconds kept before release")
    ap.add_argument("--post", type=float, default=4.0, help="seconds kept after rim arrival")
    ap.add_argument("--negatives", type=float, default=300.0,
                    help="seconds of shot-free footage to keep (0 disables — NOT recommended)")
    ap.add_argument("--neg-chunk", type=float, default=60.0)
    ap.add_argument("--engine", choices=["auto", "ffmpeg", "cv2"], default="auto",
                    help="ffmpeg = lossless stream copy (preferred); cv2 = re-encode fallback")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    engine = args.engine
    if engine == "auto":
        engine = "ffmpeg" if shutil.which("ffmpeg") else "cv2"
    if engine == "cv2" and cv2 is None:
        raise SystemExit("neither ffmpeg nor cv2 available")

    duration = video_duration_s(args.video)
    wins = load_windows(args.proposals, args.pre, args.post)
    wins = [(a, min(b, duration)) for a, b in wins if a < duration]
    negs = negative_gaps(wins, duration, args.negatives, args.neg_chunk)

    kept = sum(b - a for a, b in wins) + sum(b - a for a, b in negs)
    print(f"{args.video.name}: {duration:.0f}s -> keep {kept:.0f}s "
          f"({len(wins)} event clips, {len(negs)} negative clips, "
          f"{100 * kept / max(duration, 1e-9):.0f}% of original)")
    if args.dry_run:
        return 0

    args.outdir.mkdir(parents=True, exist_ok=True)
    manifest = {"source": str(args.video), "duration_s": duration, "engine": engine,
                "clips": []}
    for kind, spans in (("event", wins), ("negative", negs)):
        for i, (a, b) in enumerate(spans):
            name = f"{kind}_{i:03d}.mp4"
            extract(args.video, a, b, args.outdir / name, engine)
            manifest["clips"].append(
                {"file": name, "kind": kind, "source_t0_s": round(a, 3),
                 "source_t1_s": round(b, 3)})
    (args.outdir / "condense_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(manifest['clips'])} clips + manifest to {args.outdir}\n"
          "Delete the original ONLY after review confirms the shot count matches your "
          "session script.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
