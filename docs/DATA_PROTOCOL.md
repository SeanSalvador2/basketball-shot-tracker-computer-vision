# Data Collection Protocol (Stage B)

The printable field checklist, session metadata sheet, and labeling workflow for
self-collecting the fixed-camera half-court footage that Stage B trains and evaluates on
(plan §2.1). The collection is designed to **force generalization** along the axes that vary
at deployment, and to make every headline metric honest (a held-out cross-venue test set,
explicit negatives, tape-measured ground truth for shot location).

---

## 0. One-time gear

- Phone with slow-mo (1080p, 60/240 fps), a tripod (1.5 m) **and** an elevated mount option
  (clearly above the rim, ≥ ~4 m, or a fence/bleacher clamp) — **avoid ~2.7–3.3 m**: at ≈ rim
  height the rim is imaged nearly edge-on (EDA `eda_rim_geometry`), which degrades the
  rim-normalized logic and short/long estimation.
- Tape measure (cm), painter's tape for floor marks, a notebook / the metadata sheet below.

## 1. Session rules (one camera setup = one session)

- **Any change of camera position/height/angle starts a NEW session ID.** Splits are by
  session; a moved camera is a new session or the split discipline leaks.
- 20–40 min per session. Record the metadata sheet (below) **before** shooting.
- Start every session by **bouncing the ball once in frame** — a natural clapperboard for
  audio/video alignment verification (matters for T6).
- Prefer **60 fps normal-speed** capture for audio-critical (T6) sessions (slow-mo retimes
  video but records audio at normal rate — alignment risk, review R11).

## 2. Variation grid (minimum viable across the whole collection)

| Axis | Target coverage |
|---|---|
| Venue | ≥ 4: two indoor gyms (different floors), two outdoor (different backboards) |
| Ball | ≥ 3: leather indoor, rubber outdoor, worn/discolored |
| Lighting | daylight, dusk, indoor artificial |
| Camera azimuth | ≈ 15°, 30°, 45°, 60°, 90° (sideline). **45° wing is the recommended default** (both miss axes observable — A6); the sweep exists to *measure* the recommendation |
| Camera height | 1.5 m tripod **and** elevated (≥ ~4 m). Avoid ≈ rim height |

Hold **≥ 1 entire venue** out of all training/tuning — it is the cross-venue test set and the
number the README reports.

## 3. Per-session shot script (≥ 60 shots)

- **Zones × sides:** paint / midrange / 3PT × left / center / right.
- **Deliberate makes AND misses** — aim off on purpose: short, long, left, right. (The shooter
  *knows* the intended error → free miss-direction ground truth.)
- **Swish block** and **bank/rattle block** (for T6).
- **Pull-up block** (dribble before the shot) and **catch-and-shoot block** (a passer feeds) —
  for T4.
- **5 min free play** with rebounds and multiple-ball chaos.
- **≥ 5 min explicit negative blocks:** dribbling/passing drills with **zero shot attempts**,
  plus a **lob-pass block** (the adversarial near-positive). T1 precision is measured here as
  false-attempts-per-hour — without pure-negative footage the FP rate is measured on an easy
  grader (review R7).

## 4. Ground-truth shot spots (for T3 cm-error)

- Mark **9–12 floor positions** with tape; measure each from two court landmarks with the tape
  measure (cm-level). Shots from marked spots give T3 its cm-error denominator.

## 5. Session metadata sheet (fill one per session)

```
session_id:            ______   (venue_setup_date, e.g. gymA_wing45_2026-08-01)
date / time:           ______
venue:                 ______   floor type: ______
hoop type:             ______   (breakaway / fixed / netless?)  backboard: ______
ball:                  ______   (leather / rubber / worn)
lighting:              ______   (daylight / dusk / indoor artificial)
camera model / lens:   ______   fps: ______   resolution: ______
camera azimuth (deg):  ______   height (m): ______   distance to hoop (m): ______
                                 (paced or tape-measured — circle one)
marked spots measured: [ ] yes  n = ____
opening ball-bounce:   [ ] done (audio/video sync)
negative block (min):  ______   lob-pass block: [ ] done
notes:                 ______
```

Commit the sheet as `data/raw/<session_id>/metadata.yaml`; the raw video is immutable and
gitignored, its SHA-256 recorded in `data/raw/<session_id>/manifest.yaml` (the manifest is the
version, plan §2.4).

## 6. Labeling workflow (semi-automatic from day one)

1. Run the Stage-A pipeline on the clip → it **proposes** events + timestamps + make/miss +
   court location.
2. **Human verifies/corrects** in a lightweight per-shot CSV editor (events) and CVAT (boxes).
   Every correction is a training example — active learning is the loop, not a stage.
3. **Frame-level boxes** for detector fine-tuning via the data engine (plan §0: zero-shot seed
   → SAM 2.1 propagate → verify), targeting 5–10 k frames across the grid. Rim boxes are
   labeled once per session and propagated (static camera).
4. Per-shot labels: outcome, zone + marked-spot ID, shot type, miss direction (shooter-called,
   video-verified), make quality (swish/rattle), release frame (coarse).

## 7. Splits (enforced in code — `bball.eval.splits`)

- Unit = session. Cross-venue test venue held out entirely.
- Remaining sessions → train / val (stratified by venue). **Val is split into val-tune**
  (hyperparameters, FSM grid) **and val-cal** (calibration only) so tuning and calibration
  never double-dip (review R6). `assert_no_leakage` / `assert_test_venue_held_out` guard it;
  the anti-leakage unit test runs them.
