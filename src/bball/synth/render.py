"""Rendering + the detection-noise model.

Two products (plan M3):
  (a) rendered frames/mp4 — a procedural court with a motion-blurred ball, rim, net and
      backboard, plus optional player boxes. Used for end-to-end demos and for the
      pixel-consuming ablation arms (bg-sub, TrackNet-lite in A1).
  (b) a detection-noise model — turns ground-truth ball image positions into a realistic
      candidate stream: per-frame miss probability that *rises with occlusion and blur*
      (this is what manufactures the rim-occlusion gaps A5 must bridge) plus localization
      jitter. Logic-level ablations (A5/A8/A9) run on this stream without a real detector.

Rendering uses OpenCV drawing; geometry is computed in the camera's native pixels and the
frame is optionally scaled down for compact mp4s.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from bball.detect.interfaces import BallCandidate
from bball.lift.court_model import CourtSpec, paint_polygon, three_point_polyline
from bball.lift.projection import Camera
from bball.lift.rim_frame import RimEllipse
from bball.synth.camera import apparent_ball_radius_px
from bball.synth.physics import RIM_HEIGHT_M, RIM_RADIUS_M, Shot


@dataclass
class SceneAppearance:
    floor_bgr: tuple[int, int, int] = (150, 180, 205)   # warm wood default
    line_bgr: tuple[int, int, int] = (245, 245, 245)
    ball_bgr: tuple[int, int, int] = (40, 110, 220)     # basketball orange (BGR)
    backboard_bgr: tuple[int, int, int] = (235, 235, 235)
    rim_bgr: tuple[int, int, int] = (30, 60, 210)
    lighting_gain: float = 1.0                          # multiplies brightness
    lighting_tint: tuple[float, float, float] = (1.0, 1.0, 1.0)


@dataclass
class RimImageGeometry:
    ellipse: RimEllipse
    backboard_poly: np.ndarray        # 4x2 image points
    net_region: tuple[float, float, float, float]  # x0,y0,x1,y1 below-rim net box


def compute_rim_image_geometry(camera: Camera, hoop_ground_xy=(0.0, 0.0)) -> RimImageGeometry:
    """Project the 3D rim circle -> image ellipse; build backboard + net regions."""
    hx, hy = float(hoop_ground_xy[0]), float(hoop_ground_xy[1])
    t = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    circ = np.stack([hx + RIM_RADIUS_M * np.cos(t), hy + RIM_RADIUS_M * np.sin(t),
                     np.full_like(t, RIM_HEIGHT_M)], axis=1)
    circ_img = camera.project(circ)
    circ_img = circ_img[~np.isnan(circ_img).any(axis=1)]
    ell = RimEllipse.from_points(circ_img)
    # Backboard: a rectangle 1.8 m wide, 1.05 m tall, its bottom 2.9 m up, behind the rim.
    bw, bh, bbot = 1.8, 1.05, 2.9
    by = hy - 0.15  # backboard slightly behind the hoop centre (toward baseline)
    board3d = np.array([
        [hx - bw / 2, by, bbot], [hx + bw / 2, by, bbot],
        [hx + bw / 2, by, bbot + bh], [hx - bw / 2, by, bbot + bh],
    ])
    board_img = camera.project(board3d)
    # Net region: below the rim ellipse, ~0.4 m of net hanging down.
    net_bottom3d = np.array([[hx, hy, RIM_HEIGHT_M - 0.4]])
    net_bot_img = camera.project(net_bottom3d)[0]
    x0 = ell.cx - ell.a
    x1 = ell.cx + ell.a
    y0 = ell.cy
    y1 = float(max(net_bot_img[1], ell.cy + ell.b))
    return RimImageGeometry(ellipse=ell, backboard_poly=board_img, net_region=(x0, y0, x1, y1))


def occlusion_fraction(center_px, radius_px: float, rim_geom: RimImageGeometry) -> float:
    """Heuristic fraction of the ball occluded by rim/net/backboard in image space."""
    if center_px is None or np.isnan(center_px).any():
        return 1.0
    c = np.asarray(center_px, float)
    ell = rim_geom.ellipse
    frac_rim = ell.radial_fraction(c[None, :])[0]
    occ = 0.0
    # Near/behind the rim ring: partial occlusion peaking on the ring.
    if frac_rim < 1.6:
        occ = max(occ, float(np.clip(1.0 - abs(frac_rim - 1.0) / 0.6, 0.0, 1.0)) * 0.8)
    # Inside the net region: strong occlusion.
    x0, y0, x1, y1 = rim_geom.net_region
    if x0 <= c[0] <= x1 and y0 <= c[1] <= y1:
        occ = max(occ, 0.9)
    # Behind the backboard polygon.
    if cv2.pointPolygonTest(rim_geom.backboard_poly.astype(np.float32), (float(c[0]), float(c[1])), False) >= 0:
        occ = max(occ, 0.95)
    return float(np.clip(occ, 0.0, 1.0))


# --------------------------------------------------------------------------- #
# Frame drawing
# --------------------------------------------------------------------------- #
def _project_ground_polyline(camera: Camera, pts_xy: np.ndarray) -> np.ndarray:
    pts3d = np.hstack([pts_xy, np.zeros((pts_xy.shape[0], 1))])
    return camera.project(pts3d)


def render_court_background(camera: Camera, court: CourtSpec, appearance: SceneAppearance,
                            scale: float = 0.5) -> np.ndarray:
    W = int(camera.width_px * scale)
    H = int(camera.height_px * scale)
    frame = np.zeros((H, W, 3), np.uint8)
    frame[:] = appearance.floor_bgr

    def draw_poly(pts_xy, closed=False, color=None, thick=2):
        p = _project_ground_polyline(camera, np.asarray(pts_xy, float)) * scale
        p = p[~np.isnan(p).any(axis=1)]
        if len(p) >= 2:
            cv2.polylines(frame, [p.astype(np.int32)], closed, color or appearance.line_bgr,
                          max(1, int(thick * scale * 2)), cv2.LINE_AA)

    sx = court.sideline_x_m
    yb = -court.rim_from_baseline_m
    ytop = court.three_arc_radius_m + 1.5
    draw_poly([[-sx, yb], [sx, yb]])                       # baseline
    draw_poly([[-sx, yb], [-sx, ytop]])                    # left sideline
    draw_poly([[sx, yb], [sx, ytop]])                      # right sideline
    draw_poly(paint_polygon(court), closed=True)           # the key
    draw_poly(three_point_polyline(court))                 # 3pt line
    # FT circle
    ft_c = np.array([0.0, court.ft_line_from_hoop_m])
    circ = ft_c + 1.8 * np.stack([np.cos(np.linspace(0, 2 * np.pi, 40)),
                                  np.sin(np.linspace(0, 2 * np.pi, 40))], axis=1)
    draw_poly(circ, closed=True)

    # Backboard + rim (drawn once; static).
    rim_geom = compute_rim_image_geometry(camera, (0.0, 0.0))
    bp = (rim_geom.backboard_poly * scale)
    if not np.isnan(bp).any():
        cv2.polylines(frame, [bp.astype(np.int32)], True, appearance.backboard_bgr, max(1, int(2 * scale * 2)), cv2.LINE_AA)
    ell = rim_geom.ellipse
    cv2.ellipse(frame, (int(ell.cx * scale), int(ell.cy * scale)),
                (int(ell.a * scale), int(ell.b * scale)), ell.theta_deg, 0, 360,
                appearance.rim_bgr, max(1, int(3 * scale * 2)), cv2.LINE_AA)

    # Lighting.
    if appearance.lighting_gain != 1.0 or appearance.lighting_tint != (1.0, 1.0, 1.0):
        tint = np.array(appearance.lighting_tint) * appearance.lighting_gain
        frame = np.clip(frame.astype(np.float32) * tint, 0, 255).astype(np.uint8)
    return frame


def draw_ball(frame: np.ndarray, center_px, radius_px: float, velocity_px=None,
              color=(40, 110, 220), occlusion: float = 0.0) -> None:
    """Draw a motion-blurred ball (streak of alpha discs along velocity)."""
    if center_px is None or np.isnan(center_px).any():
        return
    r = max(int(round(radius_px)), 1)
    c = np.asarray(center_px, float)
    speed = 0.0 if velocity_px is None else float(np.hypot(*velocity_px))
    n_steps = int(np.clip(speed / max(r, 1), 1, 8))
    overlay = frame.copy()
    for i in range(n_steps):
        frac = (i / max(n_steps - 1, 1)) - 0.5
        offset = np.zeros(2) if velocity_px is None else np.asarray(velocity_px) * frac
        p = (c + offset).astype(int)
        cv2.circle(overlay, (int(p[0]), int(p[1])), r, color, -1, cv2.LINE_AA)
    alpha = (1.0 - 0.6 * occlusion)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def render_clip(shot: Shot, camera: Camera, court: CourtSpec, appearance: SceneAppearance,
                scale: float = 0.5, players: list | None = None) -> dict:
    """Render a full shot to frames. Returns frames + per-frame ball image data + rim geom."""
    bg = render_court_background(camera, court, appearance, scale=scale)
    rim_geom = compute_rim_image_geometry(camera, tuple(shot.hoop_ground_xy))
    ball_img = camera.project(shot.pos) * scale
    ball_rad = apparent_ball_radius_px(camera, shot.pos) * scale
    frames = []
    occl = []
    for i in range(shot.n_frames):
        frame = bg.copy()
        if players is not None:
            for pl in players:
                _draw_player_box(frame, pl, i, camera, scale)
        c = ball_img[i]
        occ = occlusion_fraction(ball_img[i] / scale if not np.isnan(c).any() else None,
                                 ball_rad[i] / max(scale, 1e-6), rim_geom)
        vel = None
        if 0 < i < shot.n_frames:
            prev = ball_img[i - 1]
            if not np.isnan(prev).any() and not np.isnan(c).any():
                vel = c - prev
        draw_ball(frame, c, ball_rad[i], vel, appearance.ball_bgr, occ)
        frames.append(frame)
        occl.append(occ)
    return {"frames": frames, "ball_img_px": ball_img / scale, "ball_radius_px": ball_rad / scale,
            "occlusion": np.array(occl), "rim_geom": rim_geom, "scale": scale}


def _draw_player_box(frame, player, frame_idx, camera, scale):
    """Draw a player as a projected bounding box from their ground position + height."""
    if frame_idx >= len(player["pos_xy"]):
        return
    x, y = player["pos_xy"][frame_idx]
    h = player.get("height_m", 1.9)
    w = 0.55
    corners = np.array([[x - w / 2, y, 0.0], [x + w / 2, y, 0.0], [x, y, h]])
    proj = camera.project(corners) * scale
    if np.isnan(proj).any():
        return
    x0 = int(min(proj[0, 0], proj[1, 0])); x1 = int(max(proj[0, 0], proj[1, 0]))
    ytop = int(proj[2, 1]); ybot = int(max(proj[0, 1], proj[1, 1]))
    cv2.rectangle(frame, (x0, ytop), (x1, ybot), (90, 90, 90), max(1, int(2 * scale * 2)))


# --------------------------------------------------------------------------- #
# Detection-noise model
# --------------------------------------------------------------------------- #
@dataclass
class DetectionNoiseModel:
    """Ground-truth ball image positions -> realistic candidate stream."""

    base_miss_prob: float = 0.03       # clean-flight miss rate (a good detector)
    occlusion_miss_gain: float = 0.95  # extra miss prob at full occlusion
    blur_miss_gain: float = 0.02       # extra miss prob per px of streak length
    jitter_px: float = 1.5             # localization noise sigma (clean)
    occlusion_jitter_gain: float = 6.0 # extra jitter sigma at full occlusion
    score_clean: float = 0.9
    fp_rate: float = 0.0               # spurious candidates per frame (multi-mover chaos)

    def miss_prob(self, occlusion: float, blur_len_px: float) -> float:
        p = self.base_miss_prob + self.occlusion_miss_gain * occlusion + self.blur_miss_gain * blur_len_px
        return float(np.clip(p, 0.0, 1.0))

    def sample(self, frame_idx: int, gt_xy, radius_px: float, occlusion: float,
               blur_len_px: float, rng: np.random.Generator) -> BallCandidate:
        if gt_xy is None or np.isnan(np.asarray(gt_xy, float)).any():
            return BallCandidate(frame_idx=frame_idx, xy=None, score=0.0, occlusion=1.0, source="synthetic")
        if rng.random() < self.miss_prob(occlusion, blur_len_px):
            return BallCandidate(frame_idx=frame_idx, xy=None, score=0.0, occlusion=occlusion, source="synthetic")
        sigma = self.jitter_px + self.occlusion_jitter_gain * occlusion
        xy = np.asarray(gt_xy, float) + rng.normal(0, sigma, size=2)
        score = float(np.clip(self.score_clean * (1.0 - 0.7 * occlusion) + rng.normal(0, 0.03), 0.05, 0.99))
        r = max(radius_px, 1.0)
        bbox = (xy[0] - r, xy[1] - r, xy[0] + r, xy[1] + r)
        return BallCandidate(frame_idx=frame_idx, xy=xy, score=score, radius_px=r,
                             source="synthetic", bbox=bbox, occlusion=occlusion)

    def stream(self, ball_img_px: np.ndarray, ball_radius_px: np.ndarray, occlusion: np.ndarray,
               rng: np.random.Generator) -> list[BallCandidate]:
        """Full per-frame candidate stream for one shot."""
        out = []
        n = ball_img_px.shape[0]
        for i in range(n):
            c = ball_img_px[i]
            gt = None if np.isnan(c).any() else c
            blur = 0.0
            if 0 < i < n:
                prev = ball_img_px[i - 1]
                if not np.isnan(prev).any() and gt is not None:
                    blur = float(np.hypot(*(c - prev)))
            out.append(self.sample(i, gt, float(ball_radius_px[i]) if not np.isnan(ball_radius_px[i]) else 3.0,
                                    float(occlusion[i]), blur, rng))
        return out


def write_mp4(frames: list[np.ndarray], path: str, fps: float = 60.0) -> None:
    """Write BGR frames to an mp4 (converts to RGB for imageio)."""
    import imageio.v2 as imageio

    with imageio.get_writer(path, fps=fps, codec="libx264", quality=6, macro_block_size=8) as w:
        for f in frames:
            w.append_data(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
