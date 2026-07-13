"""M1 smoke tests: the package imports, config composition works, seeding is
reproducible. Deeper geometry/FSM/bridging tests arrive with their milestones."""
from __future__ import annotations

import numpy as np

import bball
from bball.utils.config import apply_override, config_hash, load_config
from bball.utils.seed import new_rng, set_seed


def test_version():
    assert bball.__version__ == "0.1.0"


def test_seed_reproducible():
    set_seed(7)
    a = np.random.rand(5)
    set_seed(7)
    b = np.random.rand(5)
    assert np.allclose(a, b)


def test_new_rng_is_local_and_reproducible():
    r1 = new_rng(123).normal(size=10)
    r2 = new_rng(123).normal(size=10)
    assert np.allclose(r1, r2)


def test_config_compose_and_override(tmp_path):
    base = tmp_path / "base.yaml"
    exp = tmp_path / "exp.yaml"
    base.write_text("seed: 1\ncamera:\n  azimuth_deg: 45\n  height_m: 3.0\n")
    exp.write_text("entry: 'os:getcwd'\ncamera:\n  azimuth_deg: 60\n")
    cfg = load_config(exp, base=base, overrides=["camera.height_m=1.5", "seed=9"])
    assert cfg["seed"] == 9
    assert cfg["camera"]["azimuth_deg"] == 60      # exp overrides base
    assert cfg["camera"]["height_m"] == 1.5        # CLI overrides both
    assert cfg["entry"] == "os:getcwd"


def test_config_hash_is_order_independent():
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_apply_override_creates_nested():
    d: dict = {}
    apply_override(d, "x.y.z", 5)
    assert d == {"x": {"y": {"z": 5}}}
