"""Logging helpers. Training/experiment logs stream to files (context discipline);
console shows a compact line."""
from __future__ import annotations

import logging
import sys
from pathlib import Path


def get_logger(name: str = "bball", *, level: int = logging.INFO, logfile: str | Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if getattr(logger, "_bball_configured", False):
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S")

    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if logfile is not None:
        Path(logfile).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logfile)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger._bball_configured = True  # type: ignore[attr-defined]
    return logger
