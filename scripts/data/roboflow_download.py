#!/usr/bin/env python3
"""
Stage-B downloader for Roboflow basketball ball / rim / player datasets.

WHY THIS SCRIPT EXISTS
----------------------
The basketball CV project wants a few public Roboflow datasets for real
ball / rim / player detection. In the Stage-A sandbox this could NOT be run
because:
  1. The Roboflow hosts are policy-blocked by the outbound proxy.
     `curl https://api.roboflow.com` and `https://app.roboflow.com` both return
     `CONNECT tunnel failed, response 403` (org policy denial).
  2. Roboflow downloads require a personal API key, which the Stage-A agent
     did not have.

So this file is a ready-to-run script for a Stage-B environment that (a) can
reach roboflow.com and (b) has a ROBOFLOW_API_KEY.

HOW A STAGE-B USER RUNS IT
--------------------------
  # 1. Get a free API key: https://app.roboflow.com  ->  Settings -> API
  # 2. Install the client into the project venv:
  #      .venv/bin/pip install roboflow
  # 3. Export the key (do NOT hardcode it, do NOT commit it):
  #      export ROBOFLOW_API_KEY=xxxxxxxxxxxxxxxx
  # 4. Run one of the presets, or pass your own workspace/project/version:
  #      .venv/bin/python scripts/data/roboflow_download.py --preset ball_video_analysis
  #      .venv/bin/python scripts/data/roboflow_download.py \
  #            --workspace <ws> --project <proj> --version <n> --format yolov8

Data is written under data/external/<name>/ (gitignored). Nothing is committed.

TARGET DATASETS (from the project research notes)
-------------------------------------------------
  * "Basketball Video Analysis"  ~6076 imgs  MIT     (ball/rim)
  * "Basketball and Rim"         ~6270 imgs  CC-BY   (ball/rim)
  * "basketball-player-detection-3"           (players)

NOTE: Roboflow Universe slugs (workspace/project/version) change over time and
are tied to whoever exported them. Fill in the exact workspace + project +
version for the dataset you picked on Universe. The three presets below carry
the values most commonly seen for these datasets, but VERIFY them on the
Universe page ("Download Dataset" -> shows the `rf.workspace(...).project(...)`
snippet) before trusting them.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Repo-root-relative default output dir (…/data/external). This file lives at
# <repo>/scripts/data/roboflow_download.py, so parents[2] is the repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "data" / "external"

# Convenience presets. VERIFY workspace/project/version on the Universe page.
PRESETS = {
    # name: (workspace, project, version, expected_license)
    "ball_video_analysis": ("ownership-hnqvo", "basketball-video-analysis", 1, "MIT"),
    "basketball_and_rim": ("uni-bxfxu", "basketball-and-rim", 1, "CC-BY-4.0"),
    "player_detection": ("roboflow-jvuqo", "basketball-player-detection-3", 1, "unspecified"),
}


def download(workspace: str, project: str, version: int, fmt: str, out_dir: Path) -> Path:
    """Download one Roboflow dataset version into out_dir/<project>.

    Uses the official `roboflow` pip package. Equivalent REST call is:
      GET https://api.roboflow.com/{workspace}/{project}/{version}
          /{format}?api_key=$ROBOFLOW_API_KEY
    which returns a signed URL to a zip export.
    """
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        sys.exit(
            "ERROR: ROBOFLOW_API_KEY is not set.\n"
            "  export ROBOFLOW_API_KEY=<your key from https://app.roboflow.com>"
        )
    try:
        from roboflow import Roboflow
    except ImportError:
        sys.exit("ERROR: roboflow not installed. Run: .venv/bin/pip install roboflow")

    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / project
    print(f"[roboflow] {workspace}/{project} v{version} -> {dest} (format={fmt})")

    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    # location= controls where the zip is extracted.
    proj.version(version).download(fmt, location=str(dest))
    print(f"[roboflow] done: {dest}")
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=sorted(PRESETS), help="Use a named preset from PRESETS.")
    ap.add_argument("--workspace", help="Roboflow workspace slug.")
    ap.add_argument("--project", help="Roboflow project slug.")
    ap.add_argument("--version", type=int, help="Dataset version number.")
    ap.add_argument("--format", default="yolov8",
                    help="Export format: yolov8 | coco | voc | createml | ... (default yolov8)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help=f"Output root (default {DEFAULT_OUT})")
    args = ap.parse_args()

    if args.preset:
        ws, proj, ver, lic = PRESETS[args.preset]
        print(f"[preset {args.preset}] license≈{lic}")
        args.workspace = args.workspace or ws
        args.project = args.project or proj
        args.version = args.version if args.version is not None else ver

    if not (args.workspace and args.project and args.version is not None):
        ap.error("Provide --preset OR all of --workspace/--project/--version.")

    download(args.workspace, args.project, args.version, args.format, args.out)


if __name__ == "__main__":
    main()
