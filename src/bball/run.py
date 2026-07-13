"""Experiment entry point.

    python -m bball.run --config configs/<experiment>.yaml [key.sub=value ...]

Plan §6: "No experiment exists unless its config is committed." This is the single
reproducer: it composes the config, seeds, and dispatches to the experiment's runner
declared by the config's `entry` field.
"""
from __future__ import annotations

import argparse
import importlib
from typing import Any

from bball.utils.config import config_hash, load_config
from bball.utils.logging import get_logger
from bball.utils.seed import set_seed


def _resolve_entry(dotted: str):
    """Resolve 'module.path:function' to the callable."""
    if ":" not in dotted:
        raise ValueError(f"entry must be 'module:function', got {dotted!r}")
    mod_name, func_name = dotted.split(":", 1)
    mod = importlib.import_module(mod_name)
    return getattr(mod, func_name)


def run_config(cfg: dict[str, Any]) -> Any:
    log = get_logger("bball.run")
    seed = int(cfg.get("seed", 1729))
    set_seed(seed, strict=bool(cfg.get("strict_determinism", False)))
    log.info("config_hash=%s seed=%d entry=%s", config_hash(cfg), seed, cfg.get("entry"))
    entry = cfg.get("entry")
    if not entry:
        raise ValueError("config has no 'entry' (module:function) to dispatch to")
    fn = _resolve_entry(entry)
    return fn(cfg)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="bball experiment runner")
    ap.add_argument("--config", required=True, help="path to experiment YAML")
    ap.add_argument("--base", default=None, help="optional base config to compose under")
    ap.add_argument("overrides", nargs="*", help="key.sub=value overrides")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, base=args.base, overrides=args.overrides)
    run_config(cfg)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
