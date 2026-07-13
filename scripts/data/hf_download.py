#!/usr/bin/env python3
"""
Stage-B (and Stage-A) downloader for public basketball datasets on the
Hugging Face Hub, using `huggingface_hub.snapshot_download`.

In Stage-A, huggingface.co WAS reachable through the proxy and this pattern
worked -- emirsahin/basketball-ball (MIT, ~225 MB) was fetched with it.

LICENSE POLICY (enforced by this project)
-----------------------------------------
Only download datasets whose license is clearly MIT or CC-BY (any CC-BY
variant WITHOUT NC/ND). Skip NC, ND, "other", "unknown", or missing licenses.
Check the license BEFORE downloading (this script prints it and refuses
non-permissive licenses unless you pass --force).

CANDIDATES SURVEYED (Stage-A findings, sizes are real, via HfApi)
-----------------------------------------------------------------
  DOWNLOADABLE (permissive + small):
    emirsahin/basketball-ball                 MIT       ~225 MB  ball-detection zip  <- FETCHED in Stage-A
    sapphire1626/spinning-basketballs-dataset MIT       ~371 MB  image parquet (1K-10K imgs)
    ZhiChengAI/Basketball_V0                  CC-BY-4.0  ~1.17 GB 1097 short mp4 clips (grab a SAMPLE)
  PERMISSIVE BUT TOO BIG:
    BestWJH/VRU_Basketball                    CC-BY-4.0  ~9.6 GB  36-view 1080p/4K court video (skip / sample)
  EXCLUDED (license):
    UniqueData/basketball_tracking            CC-BY-NC-ND-4.0   (NC+ND -> excluded)
    toxsltech/ki-images-...-jersey-v1         license:other     (excluded)
    mlnomad/imnet1k_basketball                no license        (excluded)
    vahn98/basketball-audio-dataset           no license, audio (excluded)
  GATED (needs accepted terms + HF_TOKEN):
    leharris3/basketball-shot-test-dataset    MIT, gated  -> set HF_TOKEN and accept terms on the dataset page

HOW A STAGE-B USER RUNS IT
--------------------------
  .venv/bin/pip install huggingface_hub
  # Whole small repo:
  .venv/bin/python scripts/data/hf_download.py --repo emirsahin/basketball-ball
  # Just a sample of a bigger clip repo (pattern-filtered), e.g. two clips:
  .venv/bin/python scripts/data/hf_download.py --repo ZhiChengAI/Basketball_V0 \
        --allow 'videos/1/*' 'README.md'
  # Gated repo (after accepting terms on the HF page):
  export HF_TOKEN=hf_xxx
  .venv/bin/python scripts/data/hf_download.py --repo leharris3/basketball-shot-test-dataset

Data lands under data/external/hf-<name>/ (gitignored).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "data" / "external"

# License strings we accept (lower-cased, substring match). MIT + CC-BY only.
PERMISSIVE = ("mit", "cc-by-4.0", "cc-by-3.0", "cc-by-2.0", "cc-by-sa")
# Explicitly reject these even if "cc-by" appears as a substring.
FORBIDDEN = ("-nc", "-nd", "noncommercial", "noderiv")


def is_permissive(license_str: str | None) -> bool:
    if not license_str:
        return False
    s = license_str.lower()
    if any(f in s for f in FORBIDDEN):
        return False
    return any(p in s for p in PERMISSIVE)


def get_license(repo_id: str) -> str | None:
    from huggingface_hub import HfApi
    info = HfApi().repo_info(repo_id, repo_type="dataset")
    cd = info.card_data or {}
    return cd.get("license")


def download(repo_id: str, out_dir: Path, allow: list[str] | None, force: bool) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("ERROR: huggingface_hub not installed. Run: .venv/bin/pip install huggingface_hub")

    lic = None
    try:
        lic = get_license(repo_id)
    except Exception as e:  # noqa: BLE001
        print(f"[hf] WARN: could not read license for {repo_id}: {e}")
    print(f"[hf] {repo_id} license={lic!r}")
    if not is_permissive(lic) and not force:
        sys.exit(
            f"REFUSING: license {lic!r} is not clearly MIT/CC-BY (no NC/ND).\n"
            "Re-run with --force ONLY if you have verified the terms yourself."
        )

    name = "hf-" + repo_id.split("/")[-1]
    dest = out_dir / name
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[hf] -> {dest}" + (f"  (allow_patterns={allow})" if allow else "  (full repo)"))

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(dest),
        allow_patterns=allow,          # None = whole repo; list = sample only
        token=os.environ.get("HF_TOKEN"),  # needed only for gated repos
    )
    print(f"[hf] done: {dest}")
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="dataset repo id, e.g. emirsahin/basketball-ball")
    ap.add_argument("--allow", nargs="*", default=None,
                    help="glob patterns to fetch only a sample (e.g. 'videos/1/*'). Omit for whole repo.")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"output root (default {DEFAULT_OUT})")
    ap.add_argument("--force", action="store_true",
                    help="download even if license is not auto-classified as permissive (verify yourself!).")
    args = ap.parse_args()
    download(args.repo, args.out, args.allow, args.force)


if __name__ == "__main__":
    main()
