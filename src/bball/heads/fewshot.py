"""Few-shot custom shot types: prototypical head + episodic evaluation (plan §5.4, A11).

Status: **harness-validated scaffold (regime S)** — exercised on synthetic sequences; the
real embedding (pose-net penultimate layer or X-CLIP) and real clips arrive in Stage B.

Theory (plan §5.4): with K ∈ {5..40} examples per user-defined class, fitting weights
overfits; metric learning sidesteps estimation — class prototype = mean embedding of the
K support clips, prediction = softmax over negative squared distances (Prototypical
Networks). Adding a class is a no-retraining operation, which is the product requirement.

Baseline underneath it (plan §4): nearest-class-mean on raw-sequence DTW distance — if
prototypes on a good embedding cannot beat DTW-NCM, the embedding is not earning its keep.

Split discipline (plan §2.3): episodes draw support and query from *different groups*
(sessions), so a prototype can never match on venue features. `episodic_eval` enforces it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# --------------------------------------------------------------------------- #
# Prototypical head
# --------------------------------------------------------------------------- #
@dataclass
class PrototypicalHead:
    prototypes: dict = field(default_factory=dict)   # label -> mean embedding

    def fit(self, embeddings: np.ndarray, labels: list) -> "PrototypicalHead":
        E = np.asarray(embeddings, float)
        self.prototypes = {}
        for lab in sorted(set(labels)):
            idx = [i for i, l in enumerate(labels) if l == lab]
            self.prototypes[lab] = E[idx].mean(axis=0)
        return self

    def add_class(self, label, embeddings: np.ndarray) -> None:
        """The product operation: add a user-defined class from K example clips."""
        self.prototypes[label] = np.asarray(embeddings, float).mean(axis=0)

    def predict_proba(self, embeddings: np.ndarray) -> tuple[list, np.ndarray]:
        E = np.atleast_2d(np.asarray(embeddings, float))
        labs = list(self.prototypes.keys())
        P = np.stack([self.prototypes[l] for l in labs])       # (C, D)
        d2 = ((E[:, None, :] - P[None, :, :]) ** 2).sum(-1)     # (N, C)
        logits = -d2
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        return labs, probs

    def predict(self, embeddings: np.ndarray) -> list:
        labs, probs = self.predict_proba(embeddings)
        return [labs[i] for i in probs.argmax(axis=1)]


# --------------------------------------------------------------------------- #
# NCM-DTW baseline
# --------------------------------------------------------------------------- #
def dtw_distance(a: np.ndarray, b: np.ndarray, *, band: int | None = None) -> float:
    """Plain O(T^2) dynamic-time-warping distance between two (T, D) or (T,) sequences,
    optional Sakoe-Chiba band. Small and dependency-free — it is a baseline, not a product."""
    A = np.atleast_2d(np.asarray(a, float).T).T if np.asarray(a).ndim == 1 else np.asarray(a, float)
    B = np.atleast_2d(np.asarray(b, float).T).T if np.asarray(b).ndim == 1 else np.asarray(b, float)
    n, m = len(A), len(B)
    band = band or max(n, m)
    INF = float("inf")
    D = np.full((n + 1, m + 1), INF)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        j0, j1 = max(1, i - band), min(m, i + band)
        for j in range(j0, j1 + 1):
            cost = float(np.linalg.norm(A[i - 1] - B[j - 1]))
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m])


def ncm_dtw_predict(support_seqs: list, support_labels: list, query_seqs: list,
                    *, band: int | None = 10) -> list:
    """Nearest class by MEAN DTW distance to that class's support sequences."""
    labs = sorted(set(support_labels))
    out = []
    for q in query_seqs:
        best_lab, best_d = None, float("inf")
        for lab in labs:
            ds = [dtw_distance(q, s, band=band) for s, l in zip(support_seqs, support_labels) if l == lab]
            d = float(np.mean(ds))
            if d < best_d:
                best_lab, best_d = lab, d
        out.append(best_lab)
    return out


# --------------------------------------------------------------------------- #
# Episodic evaluation harness
# --------------------------------------------------------------------------- #
def episodic_eval(
    items: list[dict],
    *,
    k_shot: int,
    n_query: int,
    n_episodes: int = 20,
    method: str = "proto",
    embed_fn=None,
    seed: int = 0,
    dtw_band: int = 10,
) -> dict:
    """Few-shot accuracy over episodes with support/query drawn from DIFFERENT groups.

    `items`: [{'seq': (T,) or (T,D) array, 'label': str, 'group': str}, ...] where group is
    the session/scene id. `method`: 'proto' (needs embed_fn: seq -> vector) or 'ncm_dtw'.
    Returns mean accuracy + a percentile CI across episodes.
    """
    rng = np.random.default_rng(seed)
    labels = sorted({it["label"] for it in items})
    groups = sorted({it["group"] for it in items})
    if len(groups) < 2:
        raise ValueError("episodic_eval needs >= 2 groups (sessions) to avoid venue leakage")
    accs = []
    for _ in range(n_episodes):
        gs = list(groups)
        rng.shuffle(gs)
        cut = max(1, len(gs) // 2)
        sup_groups, qry_groups = set(gs[:cut]), set(gs[cut:])
        support = [it for it in items if it["group"] in sup_groups]
        query = [it for it in items if it["group"] in qry_groups]
        # sample K support per class, n_query queries per class
        sup_sel, qry_sel = [], []
        ok = True
        for lab in labels:
            s_pool = [it for it in support if it["label"] == lab]
            q_pool = [it for it in query if it["label"] == lab]
            if len(s_pool) < k_shot or len(q_pool) < 1:
                ok = False
                break
            sup_sel += list(rng.choice(s_pool, size=k_shot, replace=False))
            qry_sel += list(rng.choice(q_pool, size=min(n_query, len(q_pool)), replace=False))
        if not ok:
            continue
        y_true = [it["label"] for it in qry_sel]
        if method == "proto":
            if embed_fn is None:
                raise ValueError("method='proto' needs embed_fn")
            E_s = np.stack([embed_fn(it["seq"]) for it in sup_sel])
            E_q = np.stack([embed_fn(it["seq"]) for it in qry_sel])
            head = PrototypicalHead().fit(E_s, [it["label"] for it in sup_sel])
            y_pred = head.predict(E_q)
        elif method == "ncm_dtw":
            y_pred = ncm_dtw_predict([it["seq"] for it in sup_sel], [it["label"] for it in sup_sel],
                                     [it["seq"] for it in qry_sel], band=dtw_band)
        else:
            raise ValueError(f"unknown method {method!r}")
        accs.append(float(np.mean([p == t for p, t in zip(y_pred, y_true)])))
    if not accs:
        return {"accuracy": float("nan"), "lo": float("nan"), "hi": float("nan"), "n_episodes": 0}
    return {"accuracy": float(np.mean(accs)), "lo": float(np.percentile(accs, 2.5)),
            "hi": float(np.percentile(accs, 97.5)), "n_episodes": len(accs)}


def stats_embed(seq: np.ndarray) -> np.ndarray:
    """A trivially cheap fixed embedding (mean/std/min/max/band-energy) used ONLY to validate
    the harness on synthetic data. Stage B swaps in a pose-net or X-CLIP embedding."""
    z = np.asarray(seq, float).ravel()
    zc = z - z.mean()
    spec = np.abs(np.fft.rfft(zc)) ** 2
    lo = spec[: max(len(spec) // 8, 1)].sum()
    hi = spec[max(len(spec) // 8, 1):].sum()
    return np.array([z.mean(), z.std(), z.min(), z.max(), lo / (lo + hi + 1e-12)])
