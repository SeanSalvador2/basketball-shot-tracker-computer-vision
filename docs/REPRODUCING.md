# Reproducing Every Figure and Number

One command per artifact. All seeds are fixed in the committed configs; runs log to the
MLflow file store (`mlruns/`, gitignored) and export committed summaries to
`mlruns-export/`. Setup once: `make setup && make test` (100 tests).

> Container note: the Stage-A build container's `download.pytorch.org` block was intermittent
> — the torch/torchvision *wheels* were blocked at build time (install torch from PyPI
> instead, `phase1_pipeline.md` D1), but the *COCO weight* `.pth` files later fetched fine
> (D2). On any normal machine `make setup` is the clean path.

## COCO detector weights (needed only for A3 and the zero-shot real smoke test)

The torchvision wrapper loads these from the torch hub cache; fetch once (each is a plain
HTTPS GET, ~74 MB and ~180 MB):

```
mkdir -p ~/.cache/torch/hub/checkpoints
curl -L -o ~/.cache/torch/hub/checkpoints/fasterrcnn_mobilenet_v3_large_fpn-fb6a3cc7.pth \
  https://download.pytorch.org/models/fasterrcnn_mobilenet_v3_large_fpn-fb6a3cc7.pth
curl -L -o ~/.cache/torch/hub/checkpoints/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth \
  https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth
```

If the cache is empty the detector falls back to random init and records `pretrained=False`;
A3's synthetic panel still runs (and still reads 0.0 — a domain gap independent of weights),
but the real fire-rate panel is meaningless without real weights. A3's real panel also needs
the two HF fetches under `data/external/` (`scripts/data/hf_download.py`).

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
| `a3_resolution.png` + `a3_resolution.*` (needs COCO weights + HF real sets) | `... --which A3` | `configs/ablations/a3.yaml` |
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
