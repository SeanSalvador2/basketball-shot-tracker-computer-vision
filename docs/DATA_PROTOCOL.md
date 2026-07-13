# Data Collection Protocol (Stage B)

The printable field checklist, session metadata sheet, and labeling workflow for
self-collecting the fixed-camera half-court footage that Stage B trains and evaluates on
(plan §2.1). The collection is designed to **force generalization** along the axes that vary
at deployment, and to make every headline metric honest (a held-out cross-venue test set,
explicit negatives, landmark-anchored ground truth for shot location).

---

## Quick start — the simplified tiers (solo collector, 1–2 courts)

The full grid below (§2) is the *ideal*; it degrades gracefully. The two cheapest,
highest-value variation axes need no extra venues or scheduling: **camera placement** (move
the tripod between sessions) and **ball** (rotate the ones you own). Lighting is whatever
time you naturally train — two natural variants is plenty; do not schedule around it.

**Tier 1 — core (one court, ~5 sessions ≈ 3–4 h total):** each session = the 4-minute ritual
(tripod set → 5 s empty-court clip → one ball bounce in frame → metadata row) + **~60 shots**
(≈ 20 close / 20 mid / 20 threes spread left–center–right; **call ~15 deliberate misses out
loud before shooting** — "short", "left" — free miss-direction labels; one swish-hunting
block, one bang-the-rim block) + **5 min non-shooting** (dribble/passing + a few lobs toward
the rim). Across the 5 sessions vary what is free: ≥ 3 tripod azimuths (~30°, ~45–60°,
sideline; one elevated if possible — never ≈ rim height), one ball per session, rotating.
Tier 1 alone funds detector fine-tuning, real FSM validation, make/miss + location numbers
with held-out *sessions*, calibration, and first miss-direction results.

**Tier 2 — the generalization number (second court, 2–3 sessions ≈ 1.5 h):** any different
court counts (different hoop/background/park). It is **never trained on** — it is the test
set, and it upgrades claims from "works on my court" to a real transfer number.

**Tier 3 — only if convenient:** an evening/indoor lighting extra, a busy-gym multi-ball
session, a netless or double-rim hoop.

Scope honesty: with one venue, reports say "single-venue, session-level held-out"; with two,
a true (n = 1) cross-venue transfer number. Three tripod placements validate three points on
the synthetic A6 azimuth curve rather than re-measuring it — a respectable design.

---

## 0. One-time gear

- Phone with slow-mo (1080p, 60/240 fps), a tripod (1.5 m) **and** an elevated mount option
  (clearly above the rim, ≥ ~4 m, or a fence/bleacher clamp) — **avoid ~2.7–3.3 m**: at ≈ rim
  height the rim is imaged nearly edge-on (EDA `eda_rim_geometry`), which degrades the
  rim-normalized logic and short/long estimation.
- Tape measure (cm), painter's tape, **string + a weight** (keys work — the rim plumb in §4),
  a notebook / the metadata sheet below.

## 1. Session rules (one camera setup = one session)

- **Any change of camera position/height/angle starts a NEW session ID.** Splits are by
  session; a moved camera is a new session or the split discipline leaks.
- 20–40 min per session. Record the metadata sheet (below) **before** shooting.
- Start every session by **bouncing the ball once in frame** — a natural clapperboard for
  audio/video alignment verification (matters for T6).
- Prefer **60 fps normal-speed** capture for audio-critical (T6) sessions (slow-mo retimes
  video but records audio at normal rate — alignment risk, review R11).

## 2. Variation grid (the ideal — see the tiers above for the solo-collector minimum)

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

## 4. Ground-truth shot spots (for T3 cm-error) — the paint does most of the work

On a court with regulation paint, **standing on painted landmarks gives cm-level ground truth
for free** — their positions are court-spec constants the pipeline already knows
(`bball.lift.court_model`). No taping required except the origin X and optional extras.

**4a. Mark the origin (once per court).** The origin is the floor point **directly under the
rim centre** (it is the centre of the 3PT arc — every radial distance references it). Hang a
string with a weight from the **front edge** of the rim, mark the floor point, then move
**23 cm (9 in — one rim radius) straight toward centre court**, perpendicular to the
backboard. Tape an X. Sanity check: on a regulation court the X lands ≈ **1.60 m from the
inside of the baseline**; park hoops with non-standard overhang are why the plumb wins.

**4b. Identify the court's paint standard (once per court, ~2 min).** Measure origin-X → top
of the 3PT arc and match: **6.02 m** (19'9", high school) / **6.75 m** (22'1¾", FIBA-NCAA) /
**7.24 m** (23'9", NBA); cross-check FT-line→baseline = **5.79 m** (19 ft, all standards) and
lane width (3.66 m HS-NCAA / 4.88 m NBA). Match ⇒ set that court spec in the config and every
painted landmark becomes trusted ground truth. No match ⇒ non-regulation paint: enter the
measured values as a `custom` court spec (this check is what keeps landmark ground truth from
being circular — the paint anchors to physical reality exactly once).

**4c. The nine free spots (zero tape):** left/right block · FT-line centre · left/right elbow
· top of the key · **3PT apex** (stand centred, in line with the rim) · left/right
**corner-3**. Densest exactly where cm-accuracy matters — the 3-point boundary. Convention
when shooting from any GT spot: **mark under the middle of your feet** (matches the
pipeline's mid-feet-at-last-ground-contact read; consistent well under 10 cm).

**4d. Optional extra spots (2–4)** where the paint has gaps (wing midrange): tape an X and
record **two numbers** — straight-line distance from origin-X, and perpendicular distance
from the baseline — plus the side (L/R). That pins (x, y); a third distance to a lane corner
on one spot is a cheap tape-error check.

**4e. Everything else: shoot from anywhere, measure nothing.** Free-position shots carry a
shooter-called zone (behind the arc vs. not is obvious from the floor) and feed every metric
except cm-error. Zone *category* labels are then a function of position and the active zone
partition (`docs/ZONES.md`) — categories re-bucket retroactively if the taxonomy changes.

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
2. **Human verifies/corrects** with `scripts/review_events.py` (steps through proposed events;
   accept or fix outcome/zone/type/direction/quality; writes the labels CSV) and CVAT (boxes).
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
