"""Shared ablation infrastructure: MLflow tracking + figure / summary export.

Every ablation logs params, metrics and figures to a local MLflow file store (mlruns/,
gitignored) and exports a small CSV+JSON summary to mlruns-export/ (committed) so the repo
carries results without bloat and reports can cite run IDs. Seeds are fixed; configs are
committed (plan §6, §11.3).
"""
from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")  # keep the self-contained file store

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
FIG_DIR = REPO_ROOT / "reports" / "figures" / "ablations"
EXPORT_DIR = REPO_ROOT / "mlruns-export"


def setup_mlflow(experiment: str = "bball-stageA", tracking_uri: str = None) -> None:
    mlflow.set_tracking_uri(tracking_uri or f"file:{REPO_ROOT / 'mlruns'}")
    mlflow.set_experiment(experiment)


def save_fig(fig, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def export_summary(name: str, rows: list[dict], meta: dict | None = None) -> Path:
    """Write mlruns-export/<name>.csv and .json (the committed record of a run)."""
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = EXPORT_DIR / f"{name}.csv"
    if rows:
        keys = list(dict.fromkeys(k for r in rows for k in r))  # union, first-seen order
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, restval="")
            w.writeheader()
            w.writerows(rows)
    json_path = EXPORT_DIR / f"{name}.json"
    with open(json_path, "w") as f:
        json.dump({"meta": meta or {}, "rows": rows}, f, indent=2, default=str)
    return csv_path


def log_run(experiment: str, run_name: str, params: dict, metrics: dict,
            figures: dict | None = None, summary_rows: list[dict] | None = None) -> str:
    """One-call logging: params + metrics + figures + committed summary. Returns run_id."""
    setup_mlflow(experiment)
    with mlflow.start_run(run_name=run_name) as run:
        def clean(k):  # MLflow allows alnum _ - . space : /
            return re.sub(r"[^A-Za-z0-9_.:/ -]", "_", str(k))

        mlflow.log_params({clean(k): str(v) for k, v in params.items()})
        for k, v in metrics.items():
            try:
                mlflow.log_metric(clean(k), float(v))
            except (TypeError, ValueError):
                mlflow.log_param(clean(k), str(v))
        if figures:
            for fig_name, path in figures.items():
                mlflow.log_artifact(str(path))
        if summary_rows is not None:
            export_summary(run_name, summary_rows, meta={"run_id": run.info.run_id, **params})
        return run.info.run_id
