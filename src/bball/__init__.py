"""bball — basketball shot tracking from a fixed phone camera (Stage A research pipeline).

Spine: DETECT -> TRACK -> LIFT -> CLASSIFY. See docs/PROJECT_PLAN.md.

Stage A is CPU-only and makes no real-world accuracy claims: it delivers validated
machinery, tuned logic, quantified geometry, and the harness Stage B fills with real
numbers. Every reported number carries a regime label (S = synthetic, R = real).
"""

__version__ = "0.1.0"

DEFAULT_SEED = 1729

__all__ = ["__version__", "DEFAULT_SEED"]
