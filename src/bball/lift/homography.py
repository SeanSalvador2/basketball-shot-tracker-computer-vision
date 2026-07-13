"""Homography estimation from point correspondences — own implementation.

Pipeline (plan §5.3): **normalized DLT -> RANSAC -> Levenberg-Marquardt refinement**.

Why implement DLT ourselves rather than call `cv2.findHomography`: it is portfolio-
relevant code, and the normalization step (Hartley) is exactly the numerical subtlety
that a reviewer checks understanding of — raw pixel-coordinate DLT is ill-conditioned
because the entries of the design matrix span pixel^2 (~10^6) down to 1. `cv2.findHomography`
is used in the tests only as an independent cross-check.

Everything here is pure numpy + one scipy optimizer for the LM step.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


# --------------------------------------------------------------------------- #
# Core building blocks
# --------------------------------------------------------------------------- #
def _hartley_normalize(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Translate points to their centroid and scale so the mean distance to the origin
    is sqrt(2). Returns (normalized Nx2 points, 3x3 similarity transform T) with
    pts_h_norm = T @ pts_h."""
    pts = np.asarray(pts, float)
    centroid = pts.mean(axis=0)
    shifted = pts - centroid
    mean_dist = np.sqrt((shifted**2).sum(axis=1)).mean()
    if mean_dist < 1e-12:
        raise ValueError("degenerate correspondence set (all points coincide)")
    scale = np.sqrt(2.0) / mean_dist
    T = np.array(
        [[scale, 0.0, -scale * centroid[0]],
         [0.0, scale, -scale * centroid[1]],
         [0.0, 0.0, 1.0]],
        dtype=float,
    )
    normalized = (shifted * scale)
    return normalized, T


def apply_homography(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Map Nx2 points through a 3x3 homography. Returns Nx2 (NaN where w -> 0)."""
    pts = np.atleast_2d(np.asarray(pts, float))
    homog = np.hstack([pts, np.ones((pts.shape[0], 1))])
    proj = homog @ H.T
    w = proj[:, 2:3]
    with np.errstate(divide="ignore", invalid="ignore"):
        out = proj[:, :2] / w
    out[np.abs(w[:, 0]) < 1e-12] = np.nan
    return out


def reprojection_errors(H: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Per-correspondence Euclidean error (in dst units) of H mapping src -> dst."""
    pred = apply_homography(H, src)
    return np.sqrt(((pred - np.asarray(dst, float)) ** 2).sum(axis=1))


def dlt_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Normalized Direct Linear Transform. Needs >= 4 correspondences (Nx2 each).

    Builds the 2N x 9 design matrix on Hartley-normalized coordinates, solves A h = 0 by
    SVD (h = right singular vector of the smallest singular value), reshapes and
    denormalizes: H = inv(T_dst) @ H_norm @ T_src.
    """
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    if src.shape[0] < 4 or dst.shape[0] < 4 or src.shape[0] != dst.shape[0]:
        raise ValueError("need >= 4 matched correspondences of equal count")

    src_n, T_src = _hartley_normalize(src)
    dst_n, T_dst = _hartley_normalize(dst)

    n = src_n.shape[0]
    A = np.zeros((2 * n, 9), dtype=float)
    for i in range(n):
        x, y = src_n[i]
        u, v = dst_n[i]
        A[2 * i] = [-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u]
        A[2 * i + 1] = [0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v]

    _, s, Vt = np.linalg.svd(A)
    # Conditioning guard: a well-posed system has a clear gap to the last singular value.
    if s[-2] < 1e-12:
        raise ValueError("degenerate/collinear correspondences: DLT is rank-deficient")
    H_norm = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(T_dst) @ H_norm @ T_src
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]
    return H


# --------------------------------------------------------------------------- #
# RANSAC
# --------------------------------------------------------------------------- #
def homography_ransac(
    src: np.ndarray,
    dst: np.ndarray,
    *,
    threshold_px: float = 3.0,
    max_iters: int = 2000,
    confidence: float = 0.999,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """RANSAC over 4-point minimal samples. Returns (H refit on the largest inlier set,
    boolean inlier mask). Adaptive iteration count shrinks with the inlier ratio."""
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    n = src.shape[0]
    if n < 4:
        raise ValueError("RANSAC needs >= 4 correspondences")
    rng = np.random.default_rng(seed)

    best_inliers = np.zeros(n, dtype=bool)
    best_count = 0
    iters = max_iters
    i = 0
    while i < iters:
        i += 1
        idx = rng.choice(n, size=4, replace=False)
        try:
            H = dlt_homography(src[idx], dst[idx])
        except (ValueError, np.linalg.LinAlgError):
            continue
        err = reprojection_errors(H, src, dst)
        inliers = err < threshold_px
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers = inliers
            # Adapt the iteration budget to the observed inlier ratio.
            w = max(count / n, 1e-6)
            denom = np.log(max(1.0 - w**4, 1e-12))
            if denom < 0:
                iters = min(max_iters, int(np.log(1.0 - confidence) / denom) + 1)
    if best_count < 4:
        raise RuntimeError("RANSAC failed to find a 4-point consensus")
    H = dlt_homography(src[best_inliers], dst[best_inliers])
    return H, best_inliers


# --------------------------------------------------------------------------- #
# Levenberg-Marquardt refinement (geometric reprojection error)
# --------------------------------------------------------------------------- #
def refine_homography_lm(H0: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Minimize forward reprojection error over the 8 DoF (H[2,2] fixed to 1) with LM."""
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    H0 = H0 / H0[2, 2]
    p0 = H0.flatten()[:8]

    def residuals(p):
        H = np.append(p, 1.0).reshape(3, 3)
        pred = apply_homography(H, src)
        return (pred - dst).ravel()

    sol = least_squares(residuals, p0, method="lm", max_nfev=200)
    H = np.append(sol.x, 1.0).reshape(3, 3)
    return H


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
@dataclass
class HomographyResult:
    H: np.ndarray
    inlier_mask: np.ndarray
    rms_reproj_error: float
    n_points: int
    n_inliers: int
    method: str

    @property
    def max_reproj_error(self) -> float:
        return float(self._errs.max()) if self._errs.size else float("nan")

    def __post_init__(self):
        object.__setattr__(self, "_errs", np.array([]))


def estimate_homography(
    src: np.ndarray,
    dst: np.ndarray,
    *,
    use_ransac: bool = True,
    refine: bool = True,
    threshold_px: float = 3.0,
    seed: int = 0,
) -> HomographyResult:
    """Full estimation: normalized DLT, optional RANSAC, optional LM refinement.

    `src` are court-plane (or any source-plane) points, `dst` the image points (both Nx2).
    """
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    n = src.shape[0]
    method_parts = ["DLT"]

    if use_ransac and n >= 5:
        H, inliers = homography_ransac(src, dst, threshold_px=threshold_px, seed=seed)
        method_parts.append("RANSAC")
    else:
        H = dlt_homography(src, dst)
        inliers = np.ones(n, dtype=bool)

    if refine and int(inliers.sum()) >= 4:
        H = refine_homography_lm(H, src[inliers], dst[inliers])
        method_parts.append("LM")

    errs = reprojection_errors(H, src[inliers], dst[inliers])
    rms = float(np.sqrt((errs**2).mean())) if errs.size else float("nan")
    res = HomographyResult(
        H=H,
        inlier_mask=inliers,
        rms_reproj_error=rms,
        n_points=n,
        n_inliers=int(inliers.sum()),
        method="+".join(method_parts),
    )
    object.__setattr__(res, "_errs", errs)
    return res
