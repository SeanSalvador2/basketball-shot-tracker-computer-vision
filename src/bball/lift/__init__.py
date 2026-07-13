"""LIFT — projective geometry: image <-> court plane.

projection.py   pinhole primitives (Camera, intrinsics, ground homography, error model)
homography.py   normalized DLT + RANSAC + LM refinement (own implementation)
court_model.py  NBA/FIBA/HS/custom dims, zone polygons, on-the-line band, radial mode
rim_frame.py    rim ellipse annotation + rim-normalized coords, rim 3D anchor
"""
from bball.lift.court_model import (
    CourtSpec,
    classify_with_band,
    classify_zone,
    classify_zone_radial,
    get_court,
)
from bball.lift.homography import (
    HomographyResult,
    apply_homography,
    dlt_homography,
    estimate_homography,
    homography_ransac,
    refine_homography_lm,
    reprojection_errors,
)
from bball.lift.projection import Camera, intrinsics_from_fov, project_points
from bball.lift.rim_frame import RimAnnotation, RimEllipse, rim_3d_center, rim_circle_3d

__all__ = [
    "Camera",
    "intrinsics_from_fov",
    "project_points",
    "dlt_homography",
    "estimate_homography",
    "homography_ransac",
    "refine_homography_lm",
    "apply_homography",
    "reprojection_errors",
    "HomographyResult",
    "CourtSpec",
    "get_court",
    "classify_zone",
    "classify_with_band",
    "classify_zone_radial",
    "RimEllipse",
    "RimAnnotation",
    "rim_3d_center",
    "rim_circle_3d",
]
