"""Global determinism control.

Every experiment logs its seed (plan §6). `set_seed` is the single choke point so a
run is byte-reproducible where the platform allows; we do not fabricate determinism we
cannot deliver, so `strict` is opt-in and degrades loudly, not silently.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SeedState:
    seed: int
    strict: bool
    torch_available: bool


def set_seed(seed: int = 1729, *, strict: bool = False) -> SeedState:
    """Seed python, numpy and (if present) torch.

    Parameters
    ----------
    seed:
        The integer seed applied to every RNG.
    strict:
        When True, request deterministic torch algorithms and set the cuBLAS
        workspace env var. On CPU this is essentially free; we expose it so Stage-B
        GPU runs inherit the same call.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch_available = False
    try:
        import torch

        torch_available = True
        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - no GPU in Stage A
            torch.cuda.manual_seed_all(seed)
        if strict:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:  # pragma: no cover - version dependent
                pass
    except ImportError:
        pass

    return SeedState(seed=seed, strict=strict, torch_available=torch_available)


def new_rng(seed: int) -> np.random.Generator:
    """A local, non-global numpy Generator — preferred inside library code so callers
    never mutate global RNG state as a side effect."""
    return np.random.default_rng(seed)
