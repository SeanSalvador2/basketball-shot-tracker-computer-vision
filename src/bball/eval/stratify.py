"""Stratified reporting (plan §8): slice every metric by scene-config axes so the
generalization gap is explicit rather than hidden in an aggregate."""
from __future__ import annotations

from collections import defaultdict
from typing import Callable, Iterable


def group_by(records: Iterable[dict], key: str) -> dict:
    groups: dict = defaultdict(list)
    for r in records:
        groups[r.get(key)].append(r)
    return dict(groups)


def stratified_metric(records: list[dict], by: str, metric_fn: Callable[[list[dict]], dict]) -> dict:
    """Compute `metric_fn` over the whole set and within each group of axis `by`."""
    out = {"overall": metric_fn(records)}
    for value, group in sorted(group_by(records, by).items(), key=lambda kv: str(kv[0])):
        out[f"{by}={value}"] = metric_fn(group)
    return out


def occlusion_bucket(occlusion_fraction: float) -> str:
    if occlusion_fraction < 0.2:
        return "clean"
    if occlusion_fraction < 0.6:
        return "partial"
    return "heavy"
