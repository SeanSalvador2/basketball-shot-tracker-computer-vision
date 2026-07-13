"""Lightweight config system: YAML files + deep-merge composition + dotted overrides.

We deliberately avoid a heavy dependency (hydra/omegaconf). Plan §6 requires only that
"every experiment is a committed YAML config" and that runs are reproducible from it.
A config is a plain nested dict; `load_config` composes a base with an experiment file
and applies `key.subkey=value` CLI overrides, and `config_hash` gives the stable hash
logged next to every MLflow run.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


def _deep_merge(base: dict, override: Mapping) -> dict:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, Mapping):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_yaml(path: str | Path) -> dict:
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must parse to a mapping, got {type(data)}")
    return data


def _coerce(value: str) -> Any:
    """Coerce a CLI string override to int/float/bool/None/list where unambiguous."""
    low = value.lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"null", "none"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if value.startswith("[") and value.endswith("]"):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    return value


def apply_override(cfg: dict, dotted_key: str, value: Any) -> dict:
    parts = dotted_key.split(".")
    node = cfg
    for p in parts[:-1]:
        node = node.setdefault(p, {})
        if not isinstance(node, dict):
            raise ValueError(f"Override path {dotted_key!r} traverses a non-dict at {p!r}")
    node[parts[-1]] = value
    return cfg


def load_config(
    path: str | Path,
    *,
    base: str | Path | None = None,
    overrides: Iterable[str] = (),
) -> dict:
    """Compose a config.

    base <- experiment file <- dotted CLI overrides (highest precedence).
    """
    cfg: dict = {}
    if base is not None:
        cfg = load_yaml(base)
    exp = load_yaml(path)
    cfg = _deep_merge(cfg, exp)
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"Override {ov!r} must be key.subkey=value")
        key, raw = ov.split("=", 1)
        apply_override(cfg, key.strip(), _coerce(raw.strip()))
    return cfg


def config_hash(cfg: Mapping, length: int = 12) -> str:
    """Stable content hash of a config, order-independent for dict keys."""
    payload = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:length]
