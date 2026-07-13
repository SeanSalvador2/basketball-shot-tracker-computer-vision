#!/usr/bin/env python3
"""Run the ablation matrix (plan §7). Each ablation is a module bball.ablations.<name> with
a run(cfg) -> dict; its config is configs/ablations/<name>.yaml (committed). Results log to
the MLflow file store (mlruns/, gitignored) and export a CSV+JSON summary to mlruns-export/.

    python scripts/run_ablations.py --which A5 A6 A7 A8 A9
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from bball.utils.config import load_config  # noqa: E402
from bball.utils.seed import set_seed  # noqa: E402

MODULES = {
    "A1": "bball.ablations.a1_association",
    "A5": "bball.ablations.a5_bridging",
    "A6": "bball.ablations.a6_azimuth",
    "A7": "bball.ablations.a7_homography",
    "A8": "bball.ablations.a8_fsm_grid",
    "A9": "bball.ablations.a9_calibration",
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run ablations")
    ap.add_argument("--which", nargs="+", default=["A5", "A6", "A7", "A8", "A9"],
                    help="ablation ids to run (subset of %s)" % list(MODULES))
    ap.add_argument("--config-dir", default=str(REPO / "configs" / "ablations"))
    args = ap.parse_args(argv)

    for name in args.which:
        if name not in MODULES:
            print(f"[skip] unknown ablation {name}")
            continue
        cfg_path = Path(args.config_dir) / f"{name.lower()}.yaml"
        cfg = load_config(cfg_path) if cfg_path.exists() else {}
        set_seed(int(cfg.get("seed", 20260713)))
        mod = importlib.import_module(MODULES[name])
        print(f"\n===== Running {name} ({MODULES[name]}) =====")
        mod.run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
