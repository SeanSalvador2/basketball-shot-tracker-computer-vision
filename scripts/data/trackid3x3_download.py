#!/usr/bin/env python3
"""
Stage-B downloader for the TrackID3x3 3x3-basketball tracking dataset.

Repo:    https://github.com/open-starlab/TrackID3x3   (MMSports'25)
Paper:   https://arxiv.org/abs/2503.18282
License: Dataset (videos + annotations + intermediate files) = CC BY 4.0.
         Original code = Apache-2.0. NOTE the vendored `jersey-number-pipeline/`
         subtree is CC BY-NC 3.0 (non-commercial) -- keep that in mind if you
         use that submodule, but it does not affect the dataset itself.

WHAT STAGE-A ALREADY FETCHED (no Drive needed)
----------------------------------------------
`git clone` of the repo succeeds through the proxy, and the clone ALREADY
CONTAINS the CC BY 4.0 ground-truth annotations checked into git:
    data/external/TrackID3x3/ground_truth/{Indoor,Outdoor,Drone}/
        MOT/            <- per-frame bounding boxes for the 6 on-court players
        pose/           <- 10 pose keypoints (some frames)
        court_keypoints/, coordinates_with_color/, transformed_MOT*/ ...
    data/external/TrackID3x3/court_images/   <- court reference images
    data/external/TrackID3x3/videos/gif/     <- preview GIFs only
So the annotations are usable RIGHT NOW without touching Google Drive.

WHAT IS ONLY ON GOOGLE DRIVE (blocked in Stage-A)
-------------------------------------------------
The raw VIDEO files and the baseline "intermediate products" (CAMELTrack
outputs, color-histogram .npy files, etc.) live in ONE Google Drive folder:
    https://drive.google.com/drive/folders/1aWqMwQKr5xKMjqms7-raYluSlxPsGvwX
In the Stage-A sandbox this was UNREACHABLE: the proxy returns
`Tunnel connection failed: 403 Forbidden` for drive.google.com, and
`gdown --folder ...` fails with that same ProxyError. Run this script from an
environment where drive.google.com is reachable.

HOW A STAGE-B USER RUNS IT
--------------------------
  # 1. Make sure the repo (and its checked-in annotations) is present:
  #      git clone --depth 1 https://github.com/open-starlab/TrackID3x3 \
  #            data/external/TrackID3x3
  # 2. Install gdown into the venv:
  #      .venv/bin/pip install gdown
  # 3. Pull the Drive folder (this is the FULL folder -- it can be several GB;
  #    inspect before committing to it):
  #      .venv/bin/python scripts/data/trackid3x3_download.py
  #    or list-only to see what's inside without downloading:
  #      .venv/bin/python scripts/data/trackid3x3_download.py --list-only

Everything lands under data/external/TrackID3x3/drive/ (gitignored).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEST = REPO_ROOT / "data" / "external" / "TrackID3x3" / "drive"

# The single Drive folder that holds videos + baseline intermediate products.
# (All four README references point at this same folder id.)
DRIVE_FOLDER_ID = "1aWqMwQKr5xKMjqms7-raYluSlxPsGvwX"
DRIVE_FOLDER_URL = f"https://drive.google.com/drive/folders/{DRIVE_FOLDER_ID}"


def run(dest: Path, list_only: bool, url: str = DRIVE_FOLDER_URL) -> None:
    try:
        import gdown
    except ImportError:
        sys.exit("ERROR: gdown not installed. Run: .venv/bin/pip install gdown")

    dest.mkdir(parents=True, exist_ok=True)
    print(f"[trackid3x3] Drive folder: {url}")
    print(f"[trackid3x3] dest: {dest}")

    # gdown.download_folder respects `skip_download` to only enumerate the tree.
    # A full folder pull can be large -- prefer --list-only first, then fetch
    # only the subfolder you need (e.g. just the Indoor subset) by browsing the
    # Drive UI and passing that subfolder's URL via --url.
    try:
        gdown.download_folder(
            url=url,
            output=str(dest),
            quiet=False,
            use_cookies=False,
            skip_download=list_only,
        )
    except Exception as e:  # noqa: BLE001 - surface the proxy/rate-limit reason
        sys.exit(
            f"ERROR: gdown failed ({type(e).__name__}: {e}).\n"
            "If this is a 403 'Tunnel connection failed', drive.google.com is "
            "policy-blocked here -- run from a network where Drive is reachable. "
            "If it's a rate-limit, retry later or download the folder manually."
        )
    print("[trackid3x3] done." if not list_only else "[trackid3x3] listing complete.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST, help=f"Output dir (default {DEFAULT_DEST})")
    ap.add_argument("--list-only", action="store_true",
                    help="Enumerate the Drive folder without downloading (skip_download).")
    ap.add_argument("--url", default=DRIVE_FOLDER_URL,
                    help="Override the Drive folder/file URL (e.g. a single subset subfolder).")
    args = ap.parse_args()
    run(args.dest, args.list_only, args.url)


if __name__ == "__main__":
    main()
