#!/usr/bin/env python
"""Tone-map an HDR (iPhone Dolby Vision/HLG) clip to SDR BT.709 for analysis/training.

Usage:  python scripts/tonemap_hdr.py IN.MOV OUT_sdr.mp4 [--ffmpeg PATH]

Finds ffmpeg from --ffmpeg, PATH, or the imageio-ffmpeg binary bundled with this project's
venv. Tries the reference chain (zscale linearization + Hable tone mapping); if that build
lacks the zscale filter, falls back to a direct HLG->BT.709 transfer conversion via the
colorspace filter (HLG is quasi-backward-compatible; highlights clip slightly — acceptable
for CV, and the used chain is printed and should be recorded in session metadata).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys

# Canonical tonemap chain: linearize -> float -> 709 primaries -> hable -> 709 transfer.
# The float step (gbrpf32le) is required by the tonemap filter.
_TAIL = ("format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,"
         "zscale=t=bt709:m=bt709:r=tv,format=yuv420p")
# Reference chain: trusts the file's HDR tags (normal for straight-off-iPhone footage).
CHAIN_TAGGED = f"zscale=t=linear:npl=100,{_TAIL}"
# Fallback: some transfers strip HDR metadata — tell zscale the input is HLG/BT.2020.
CHAIN_ASSUME_HLG = (f"zscale=rin=tv:tin=arib-std-b67:pin=bt2020:min=bt2020nc:"
                    f"t=linear:npl=100,{_TAIL}")


def find_ffmpeg(explicit: str | None) -> str:
    if explicit:
        return explicit
    on_path = shutil.which("ffmpeg")
    if on_path:
        return on_path
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        sys.exit("no ffmpeg found: pass --ffmpeg PATH, or `pip install imageio-ffmpeg`")


def run_chain(ffmpeg: str, src: str, dst: str, vf: str) -> subprocess.CompletedProcess:
    cmd = [ffmpeg, "-y", "-nostdin", "-loglevel", "error", "-stats", "-i", src,
           "-vf", vf, "-c:v", "libx264", "-crf", "18", "-c:a", "copy", dst]
    return subprocess.run(cmd, capture_output=True, text=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--ffmpeg", default=None)
    args = ap.parse_args()

    ffmpeg = find_ffmpeg(args.ffmpeg)
    print(f"using ffmpeg: {ffmpeg}")

    print("trying reference chain (tag-driven zscale + hable tonemap)…")
    r = run_chain(ffmpeg, args.src, args.dst, CHAIN_TAGGED)
    if r.returncode == 0:
        print(f"done -> {args.dst}\nchain used: zscale+hable, tag-driven (record in metadata)")
        return 0
    if "No such filter" in (r.stderr or ""):
        sys.exit("this ffmpeg build lacks zscale — install a full build "
                 "(Windows: winget install Gyan.FFmpeg, or the gyan.dev zip) and re-run")
    print("tags unusable — retrying with explicit HLG/BT.2020 input assumption…")
    r2 = run_chain(ffmpeg, args.src, args.dst, CHAIN_ASSUME_HLG)
    if r2.returncode == 0:
        print(f"done -> {args.dst}\nchain used: zscale+hable, assumed-HLG input "
              "(record in metadata)")
        return 0
    sys.stderr.write((r.stderr or "") + "\n" + (r2.stderr or ""))
    return r2.returncode


if __name__ == "__main__":
    sys.exit(main())
