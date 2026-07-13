"""Split discipline (plan §2.3, review R6).

Unit of splitting = session (real) / scene config (synthetic) — never frames, never shots:
adjacent frames are near-duplicates and shots within a session share background/ball/light.
A whole venue is held out as the cross-venue test set (the headline number). The remaining
sessions split into train and val, and val is further split into **val-tune** (hyperparams,
FSM grids) and **val-cal** (calibration fitting only) so that tuning and calibrating never
double-dip the same sessions. `assert_no_leakage` enforces the invariants in code (the
anti-leakage unit test calls it).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Split:
    train: list[str] = field(default_factory=list)
    val_tune: list[str] = field(default_factory=list)
    val_cal: list[str] = field(default_factory=list)
    test: list[str] = field(default_factory=list)

    @property
    def val(self) -> list[str]:
        return self.val_tune + self.val_cal

    def as_dict(self) -> dict:
        return {"train": self.train, "val_tune": self.val_tune, "val_cal": self.val_cal, "test": self.test}


def make_split(
    scenes: list[dict],
    *,
    test_venues: list[str],
    val_frac: float = 0.3,
    val_cal_frac: float = 0.5,
    seed: int = 0,
) -> Split:
    """Build a leakage-safe split.

    `scenes` is a list of {'scene_id':..., 'venue':...}. All scenes of a `test_venues` venue
    go to test. The rest are split train/val by scene, stratified by venue; val is split into
    val_tune and val_cal.
    """
    rng = np.random.default_rng(seed)
    test = [s["scene_id"] for s in scenes if s["venue"] in set(test_venues)]
    remaining = [s for s in scenes if s["venue"] not in set(test_venues)]

    # Stratify the train/val split by venue.
    train, val = [], []
    by_venue: dict[str, list[str]] = {}
    for s in remaining:
        by_venue.setdefault(s["venue"], []).append(s["scene_id"])
    for venue, ids in by_venue.items():
        ids = list(ids)
        rng.shuffle(ids)
        n_val = int(round(len(ids) * val_frac))
        val.extend(ids[:n_val])
        train.extend(ids[n_val:])

    val = list(val)
    rng.shuffle(val)
    n_cal = int(round(len(val) * val_cal_frac))
    val_cal = val[:n_cal]
    val_tune = val[n_cal:]
    return Split(train=sorted(train), val_tune=sorted(val_tune), val_cal=sorted(val_cal), test=sorted(test))


def assert_no_leakage(split: Split) -> None:
    """Every scene appears in exactly one split; raises AssertionError otherwise."""
    groups = {"train": split.train, "val_tune": split.val_tune, "val_cal": split.val_cal, "test": split.test}
    seen: dict[str, str] = {}
    for name, ids in groups.items():
        for sid in ids:
            if sid in seen:
                raise AssertionError(f"scene {sid!r} leaks across splits: {seen[sid]} and {name}")
            seen[sid] = name


def assert_test_venue_held_out(split: Split, scenes: list[dict], test_venues: list[str]) -> None:
    """No test-venue scene may appear in train/val, and every test scene is a test-venue scene."""
    venue_of = {s["scene_id"]: s["venue"] for s in scenes}
    tv = set(test_venues)
    for sid in split.train + split.val_tune + split.val_cal:
        if venue_of.get(sid) in tv:
            raise AssertionError(f"test-venue scene {sid!r} leaked into a training/val split")
    for sid in split.test:
        if venue_of.get(sid) not in tv:
            raise AssertionError(f"non-test-venue scene {sid!r} in the test split")
