# License Ledger

A maintained record of every dependency, its version, license, and role (plan §2.2, §10).
The project's license posture is **permissive-only for anything that could ship**: no
AGPL/GPL/CC-NC/unlicensed code or weights is a runtime or shipped dependency. Re-verify each
entry's LICENSE at every dependency addition.

## Python runtime dependencies (pinned in `pyproject.toml`)

| Package | Version | License | Role |
|---|---|---|---|
| numpy | 2.2.6 | BSD-3-Clause | arrays, linear algebra (geometry, physics) |
| scipy | 1.17.1 | BSD-3-Clause | `linear_sum_assignment` (Hungarian), `least_squares` (LM), optimizers |
| opencv-python-headless | 5.0.0.93 | Apache-2.0 | drawing, MOG2 background subtraction, image ops (headless = no GUI deps) |
| matplotlib | 3.11.0 | matplotlib (PSF-based, BSD-compatible) | all figures (Agg backend) |
| torch | 2.8.0 | BSD-3-Clause | TrackNet-lite, learned heads (CPU) |
| torchvision | 0.23.0 | BSD-3-Clause | Faster R-CNN detector architectures (COCO weights: Stage B) |
| mlflow | 3.14.0 | Apache-2.0 | experiment tracking (local file store) |
| pyyaml | 6.0.3 | MIT | config files |
| imageio | 2.37.3 | BSD-2-Clause | mp4 read/write |
| imageio-ffmpeg | 0.6.0 | BSD-2-Clause (wheel bundles an ffmpeg build) | mp4 codec backend — see note |

## Dev dependencies

| Package | Version | License | Role |
|---|---|---|---|
| pytest | 9.1.1 | MIT | test suite |
| nbconvert | 7.17.1 | BSD-3-Clause | `make demo` notebook execution |
| jupyter / ipykernel | 1.1.1 / 7.3.0 | BSD-3-Clause | notebook kernel |

## Notes & deviations

- **torch CPU wheels vs the build container.** The intended install path (plan) is the
  CPU-only wheel index `https://download.pytorch.org/whl/cpu`, which produces a small
  BSD-3-licensed torch with no CUDA payload. That host is **firewalled in the Stage-A build
  container** (HTTP 403 from the egress proxy). The documented fallback installed the default
  PyPI `torch==2.8.0` wheel plus the `nvidia-*-cu12` CUDA runtime packages it requires at
  import. **Those CUDA libraries are NVIDIA-proprietary and are never used**: execution is
  CPU-only (`torch.cuda.is_available() == False`). On any networked machine, `make setup`
  uses the clean CPU index and no NVIDIA package is installed. The nvidia deps are not part of
  the shipped/product dependency set — they are a container artifact.
- **ffmpeg.** `imageio-ffmpeg` downloads/bundles an ffmpeg binary; upstream ffmpeg is LGPL/GPL
  depending on build. It is a *tooling* dependency for writing demo mp4s, not linked into any
  shipped artifact. For a shipped product, use the platform's native AVFoundation encoder.

## Data & model provenance

| Asset | License | Use here | Ships? |
|---|---|---|---|
| Synthetic engine output (this repo) | project (MIT) | all Stage-A experiments | yes |
| TrackID3x3 annotations (fetched) | CC BY 4.0 | Stage-B player det/track transfer | attribution required |
| HF `emirsahin/basketball-ball` (fetched) | MIT | Stage-B ball-detector seeding | yes |
| HF `ZhiChengAI/Basketball_V0` (fetched) | CC BY 4.0 | Stage-B real-clip sanity checks | attribution required |
| Roboflow ball/rim sets (blocked here) | MIT / CC BY 4.0 | Stage-B detector seeding | yes (per-badge re-verify) |
| COCO-pretrained torchvision weights (Stage B) | BSD-3-Clause | detector baseline | yes |

Fetched data lives under `data/external/` (gitignored); the Stage-B downloader scripts are in
`scripts/data/` with a license gate. **Excluded by policy:** Ultralytics YOLO (AGPL-3.0),
KaliCalib (CeCILL), PnLCalib/T-DEED (GPL), VideoMAE weights and DeepSportRadar images (CC-NC),
and any unlicensed repo — see `reports/phase0_research.md` for the full survey.
