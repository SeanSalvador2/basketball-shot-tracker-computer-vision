# Reproducing Every Figure and Number

One command per artifact. All seeds are fixed in the committed configs; runs log to the
MLflow file store (`mlruns/`, gitignored) and export committed summaries to
`mlruns-export/`. Setup once: `make setup && make test` (89 tests).

> Container note: the Stage-A build container firewalls `download.pytorch.org`; on such a
> network install torch from PyPI instead (see `reports/phase1_pipeline.md` D1). On any
> normal machine `make setup` is the clean path.

## Environment

| What | Command |
|---|---|
| venv + pinned deps + editable package | `make setup` |
| test suite (geometry, FSM, bridging, leakage guards) | `make test` |

## Figures → commands

| Artifact | Command | Config (seed inside) |
|---|---|---|
| `reports/figures/eda/*.png` + `eda_findings.json` (6 analyses) | `make eda` | `configs/eda.yaml` |
| `reports/figures/ablations/a1_association_arms.png` + `mlruns-export/a1_association.*` | `.venv/bin/python scripts/run_ablations.py --which A1` | `configs/ablations/a1.yaml` |
| `a5_bridging_gap.png` + `a5_bridging.*` | `... --which A5` | `configs/ablations/a5.yaml` |
| `a6_azimuth_sweep.png` + `a6_azimuth_sweep.*` (headline) | `... --which A6` | `configs/ablations/a6.yaml` |
| `a7_homography_error.png`, `a7_error_isolines.png` + `a7_homography.*` | `... --which A7` | `configs/ablations/a7.yaml` |
| `a8_fsm_sensitivity.png` + `a8_fsm_grid.*` | `... --which A8` | `configs/ablations/a8.yaml` |
| `a9_reliability.png` + `a9_calibration.*` | `... --which A9` | `configs/ablations/a9.yaml` |
| all non-droppable ablations in one go | `make ablations` | all of the above |
| demo clip + metadata (`notebooks/assets/`) | `.venv/bin/python scripts/build_demo_clip.py` | seed hardcoded (20260713) |
| demo notebook source | `.venv/bin/python scripts/build_demo_notebook.py` | — |
| executed demo notebook (G1) | `make demo` | — |
| synthetic session bundle (`data/synthetic/`) | `make synth` | `configs/synth_bundle.yaml` |

## Numbers quoted in reports → sources

| Number | Source |
|---|---|
| A5/A6/A7/A8/A9/A1 tables in `phase1_experiments.md` | the matching `mlruns-export/*.csv`; MLflow run ID in the `.json` twin |
| EDA numbers in `phase1_eda.md` | `reports/figures/eda/eda_findings.json` |
| FSM batch accuracy 89–96% (S) | inline sweep in `phase1_experiments.md` context; re-run: FSM over `generate_session` at az ∈ {30,45,55,60}, h=1.5 m, seed 7 (see `tests/test_events.py` for the construction) |
| Demo 5/6 | `make demo` (accuracy cell prints it) |

## Determinism notes

- Every config carries its `seed`; `bball.utils.seed.set_seed` seeds python/numpy/torch.
- TrackNet-lite training (A1) sets `torch.manual_seed` + fixed thread count; exact loss
  values may still vary across BLAS builds at the 1e-3 level — the reported completeness/F1
  table is robust to that.
- MLflow run IDs are minted fresh on re-run; the committed CSV/JSON summaries are the
  stable citation surface.
