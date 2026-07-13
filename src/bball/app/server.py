"""Local web workbench: calibrate → analyze → review/label → zones → results.

Runs the existing pipeline behind a FastAPI service with a vanilla-JS front end
(`make app`, then open http://localhost:8000 — same URL from a phone on the same
network; the page is installable as a PWA). This is the Stage-B labeling tool and the
product-experience preview; the native iOS app (on-device Core ML) is Phase 2.

State: one directory per session under data/app_sessions/<sid>/ (state.json + labels.csv
+ uploaded video). Everything heavy reuses bball.* modules — this file is glue only.
"""
from __future__ import annotations

import json
import shutil
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from bball.app.labels import FIELDS, load_csv, rows_from_report, save_csv
from bball.lift import zones as zones_mod
from bball.lift.court_model import get_court, landmark_points, paint_polygon, three_point_polyline
from bball.lift.homography import apply_homography, estimate_homography
from bball.lift.rim_frame import RimEllipse, conic_to_geometric, fit_ellipse
from bball.pipeline import detect_ball_bgsub, track_and_classify

STATIC_DIR = Path(__file__).parent / "static"
DATA_ROOT = Path(__file__).resolve().parents[3] / "data" / "app_sessions"

app = FastAPI(title="bball workbench")


# --------------------------------------------------------------------------- #
# Session store (JSON-file backed; this is a single-user local tool)
# --------------------------------------------------------------------------- #
def _sdir(sid: str) -> Path:
    d = DATA_ROOT / sid
    if not d.exists():
        raise HTTPException(404, f"unknown session {sid}")
    return d


def _load(sid: str) -> dict:
    return json.loads((_sdir(sid) / "state.json").read_text())


def _save(sid: str, state: dict) -> None:
    (_sdir(sid) / "state.json").write_text(json.dumps(state, indent=2))


def _probe(video: Path) -> dict:
    cap = cv2.VideoCapture(str(video))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        return {"fps": fps, "n_frames": n, "duration_s": n / fps if fps else 0.0,
                "w": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "h": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    finally:
        cap.release()


# --------------------------------------------------------------------------- #
# Court geometry for the UI
# --------------------------------------------------------------------------- #
def _court_payload(spec: str) -> dict:
    c = get_court(spec)
    return {
        "spec": c.name,
        "landmarks": {k: v.tolist() for k, v in landmark_points(c).items()},
        "three": three_point_polyline(c).tolist(),
        "paint": paint_polygon(c).tolist(),
        "sideline_x_m": c.sideline_x_m,
        "rim_from_baseline_m": c.rim_from_baseline_m,
        "halfcourt_y_m": 14.33 - c.rim_from_baseline_m,
    }


@app.get("/api/court")
def api_court(spec: str = "nba") -> dict:
    return _court_payload(spec)


# --------------------------------------------------------------------------- #
# Sessions
# --------------------------------------------------------------------------- #
@app.post("/api/sessions")
async def create_session(request: Request) -> dict:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    sid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    sdir = DATA_ROOT / sid
    sdir.mkdir()
    ctype = request.headers.get("content-type", "")
    if ctype.startswith("multipart/"):
        form = await request.form()
        up = form["file"]
        assert isinstance(up, UploadFile)
        video = sdir / (Path(up.filename or "upload.mp4").name)
        with open(video, "wb") as f:
            shutil.copyfileobj(up.file, f)
    else:
        body = await request.json()
        video = Path(body["video_path"]).expanduser().resolve()
        if not video.exists():
            shutil.rmtree(sdir)
            raise HTTPException(400, f"video not found: {video}")
    state = {"sid": sid, "video": str(video), "probe": _probe(video), "spec": "nba",
             "calibration": None, "rim": None, "partition": {"mode": "basic3", "court": "nba",
                                                             "interior_radius_m": 2.1336}}
    _save_new(sdir, state)
    return state


def _save_new(sdir: Path, state: dict) -> None:
    (sdir / "state.json").write_text(json.dumps(state, indent=2))


@app.get("/api/sessions")
def list_sessions() -> list[dict]:
    if not DATA_ROOT.exists():
        return []
    out = []
    for d in sorted(DATA_ROOT.iterdir()):
        f = d / "state.json"
        if f.exists():
            s = json.loads(f.read_text())
            out.append({"sid": s["sid"], "video": s["video"],
                        "calibrated": s.get("calibration") is not None,
                        "rim": s.get("rim") is not None})
    return out


@app.get("/api/sessions/{sid}")
def get_session(sid: str) -> dict:
    return _load(sid)


@app.get("/api/sessions/{sid}/frame")
def get_frame(sid: str, t: float = 0.0, maxw: int = 1280) -> Response:
    state = _load(sid)
    cap = cv2.VideoCapture(state["video"])
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(t, 0.0) * 1000.0)
        ok, frame = cap.read()
        if not ok:
            raise HTTPException(404, f"no frame at t={t}")
    finally:
        cap.release()
    h, w = frame.shape[:2]
    if w > maxw:
        frame = cv2.resize(frame, (maxw, int(h * maxw / w)))
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 87])
    return Response(content=buf.tobytes(), media_type="image/jpeg",
                    headers={"X-Native-Width": str(w), "X-Native-Height": str(h)})


@app.get("/api/sessions/{sid}/video")
def get_video(sid: str, request: Request):
    """Range-aware video serving so the review player can seek."""
    path = Path(_load(sid)["video"])
    size = path.stat().st_size
    rng = request.headers.get("range")
    if not rng:
        return FileResponse(path, media_type="video/mp4")
    try:
        start_s, _, end_s = rng.split("=")[1].partition("-")
        start = int(start_s)
        end = min(int(end_s) if end_s else start + 4_000_000, size - 1)
    except Exception:
        raise HTTPException(416, "bad range")
    with open(path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start + 1)
    return Response(chunk, status_code=206, media_type="video/mp4",
                    headers={"Content-Range": f"bytes {start}-{end}/{size}",
                             "Accept-Ranges": "bytes"})


# --------------------------------------------------------------------------- #
# Calibration & rim
# --------------------------------------------------------------------------- #
@app.post("/api/sessions/{sid}/calibrate")
async def calibrate(sid: str, request: Request) -> dict:
    body = await request.json()
    pts = body["points"]
    if len(pts) < 4:
        raise HTTPException(400, "need >= 4 correspondences")
    court_pts = np.array([p["court"] for p in pts], float)
    img_pts = np.array([p["img"] for p in pts], float)
    res = estimate_homography(court_pts, img_pts)  # court -> image
    state = _load(sid)
    state["spec"] = body.get("spec", state.get("spec", "nba"))
    state["calibration"] = {
        "H_court2img": res.H.tolist(),
        "H_img2court": np.linalg.inv(res.H).tolist(),
        "rms_px": res.rms_reproj_error, "n_points": res.n_points,
        "n_inliers": res.n_inliers, "points": pts,
    }
    _save(sid, state)
    c = get_court(state["spec"])
    overlay = {
        "three": apply_homography(res.H, three_point_polyline(c)).tolist(),
        "paint": apply_homography(res.H, paint_polygon(c)).tolist(),
    }
    return {"rms_px": res.rms_reproj_error, "n_inliers": res.n_inliers,
            "n_points": res.n_points, "overlay_img": overlay}


@app.post("/api/sessions/{sid}/rim")
async def set_rim(sid: str, request: Request) -> dict:
    body = await request.json()
    pts = np.array(body["points"], float)
    if len(pts) < 5:
        raise HTTPException(400, "need >= 5 points on the rim circle")
    center, a, b, theta = conic_to_geometric(fit_ellipse(pts))
    rim = {"cx": float(center[0]), "cy": float(center[1]), "a": float(a), "b": float(b),
           "theta_deg": float(theta)}
    state = _load(sid)
    state["rim"] = rim
    _save(sid, state)
    tt = np.linspace(0, 2 * np.pi, 72)
    th = np.deg2rad(theta)
    poly = np.stack([
        center[0] + a * np.cos(tt) * np.cos(th) - b * np.sin(tt) * np.sin(th),
        center[1] + a * np.cos(tt) * np.sin(th) + b * np.sin(tt) * np.cos(th)], axis=1)
    return {"rim": rim, "polyline": poly.tolist()}


# --------------------------------------------------------------------------- #
# Analyze (proposals) — bg-sub spine; torchvision shooter-lift when available
# --------------------------------------------------------------------------- #
def _shooter_feet(frame: np.ndarray, ball_xy: np.ndarray | None):
    """Best-effort person detection at the release frame (COCO torchvision). Returns
    bottom-center of the person box nearest the ball, or None. Never raises."""
    try:
        import torch
        from torchvision.models import detection as tvd

        model = getattr(_shooter_feet, "_model", None)
        if model is None:
            model = tvd.fasterrcnn_mobilenet_v3_large_fpn(
                weights=tvd.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT).eval()
            _shooter_feet._model = model
        x = torch.from_numpy(frame[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0
        with torch.no_grad():
            out = model([x])[0]
        keep = [(b, s) for b, s, l in zip(out["boxes"], out["scores"], out["labels"])
                if int(l) == 1 and float(s) > 0.5]
        if not keep:
            return None
        if ball_xy is not None:
            keep.sort(key=lambda bs: float(
                np.hypot((bs[0][0] + bs[0][2]) / 2 - ball_xy[0], bs[0][1] - ball_xy[1])))
        box = keep[0][0].numpy()
        return float((box[0] + box[2]) / 2), float(box[3])
    except Exception:
        return None


@app.post("/api/sessions/{sid}/analyze")
async def analyze(sid: str, request: Request) -> dict:
    body = await request.json() if int(request.headers.get("content-length") or 0) else {}
    stride = int(body.get("stride", 2))
    scale = float(body.get("scale", 0.5))
    state = _load(sid)
    if state.get("rim") is None:
        raise HTTPException(400, "annotate the rim first (Calibrate tab)")
    rim = RimEllipse(**state["rim"])
    cap = cv2.VideoCapture(state["video"])
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames, native = [], []
    keep_native = state.get("calibration") is not None
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(fr, None, fx=scale, fy=scale))
        native.append(fr if keep_native else None)
        for _ in range(stride - 1):
            cap.grab()
    cap.release()
    times = np.arange(len(frames)) * (stride / fps)
    cands = detect_ball_bgsub(frames, scale)
    events = track_and_classify(cands, times, rim)

    shots = []
    for ev in events:
        shot = {"t_release_s": round(float(ev.release_t), 3),
                "t_rim_s": round(float(ev.rim_t), 3),
                "outcome": ev.outcome, "make_prob": round(float(ev.make_prob), 3),
                "court_xy": None, "zone": ""}
        if keep_native and state.get("calibration"):
            k = int(np.clip(round(ev.release_t * fps / stride), 0, len(native) - 1))
            ball = cands[k].xy / scale if cands[k].xy is not None else None
            feet = _shooter_feet(native[k], ball)
            if feet is not None:
                from bball.pipeline import lift_shooter

                H = np.array(state["calibration"]["H_img2court"])
                z = lift_shooter(np.array(feet), H, get_court(state["spec"]))
                shot["court_xy"], shot["zone"] = list(z["court_xy"]), z["zone"]
        shots.append(shot)
    state["analysis"] = {"shots": shots, "stride": stride, "scale": scale,
                         "n_frames": len(frames)}
    _save(sid, state)
    return state["analysis"]


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #
@app.get("/api/sessions/{sid}/labels")
def get_labels(sid: str) -> dict:
    sdir = _sdir(sid)
    csv_path = sdir / "labels.csv"
    if csv_path.exists():
        return {"rows": load_csv(csv_path), "fields": FIELDS, "saved": True}
    state = _load(sid)
    shots = (state.get("analysis") or {}).get("shots", [])
    return {"rows": rows_from_report(shots), "fields": FIELDS, "saved": False}


@app.post("/api/sessions/{sid}/labels")
async def post_labels(sid: str, request: Request) -> dict:
    body = await request.json()
    rows = body["rows"]
    path = _sdir(sid) / "labels.csv"
    save_csv(rows, path)
    return {"saved": str(path), "n": len(rows)}


# --------------------------------------------------------------------------- #
# Zones
# --------------------------------------------------------------------------- #
def _partition_payload(part: zones_mod.ZonePartition) -> dict:
    return {"name": part.name, "zones": list(part.zones), "spec": part.to_dict(),
            "boundaries": {k: v.tolist() for k, v in part.boundaries.items()}}


@app.get("/api/zones/presets")
def zone_presets(spec: str = "nba") -> list[dict]:
    c = get_court(spec)
    return [_partition_payload(p) for p in
            (zones_mod.preset_basic3(c), zones_mod.preset_extended(c), zones_mod.preset_spots(c))]


@app.post("/api/zones/preview")
async def zone_preview(request: Request) -> dict:
    part = zones_mod.from_dict(await request.json())
    return _partition_payload(part)


@app.post("/api/sessions/{sid}/zones")
async def set_zones(sid: str, request: Request) -> dict:
    body = await request.json()
    part = zones_mod.from_dict(body)  # validates
    state = _load(sid)
    state["partition"] = body
    _save(sid, state)
    payload = _partition_payload(part)
    if state.get("calibration"):
        H = np.array(state["calibration"]["H_court2img"])
        payload["overlay_img"] = {k: apply_homography(H, np.array(v)).tolist()
                                  for k, v in payload["boundaries"].items()}
    return payload


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
@app.get("/api/sessions/{sid}/results")
def results(sid: str) -> dict:
    state = _load(sid)
    part = zones_mod.from_dict(state.get("partition") or {"mode": "basic3", "court": state.get("spec", "nba"),
                                                          "interior_radius_m": 2.1336})
    rows = get_labels(sid)["rows"]
    shots, by_zone = [], {}
    n_make = 0
    for r in rows:
        if r.get("verified") == "excluded":
            continue
        outcome = r.get("outcome") or ""
        n_make += outcome == "make"
        x, y = r.get("court_x_m"), r.get("court_y_m")
        entry = {"outcome": outcome, "zone": r.get("zone") or None, "xy": None, "on_line": False}
        if x not in ("", None) and y not in ("", None):
            res = part.classify(float(x), float(y))
            entry.update({"xy": [float(x), float(y)], "zone": res["zone"],
                          "on_line": res["on_line"]})
        shots.append(entry)
        z = entry["zone"] or "unlocated"
        by_zone.setdefault(z, {"attempts": 0, "makes": 0})
        by_zone[z]["attempts"] += 1
        by_zone[z]["makes"] += entry["outcome"] == "make"
    n = len(shots)
    return {"partition": part.name, "attempts": n, "makes": n_make,
            "fg_pct": round(n_make / n, 3) if n else None, "by_zone": by_zone,
            "chart": shots}


# --------------------------------------------------------------------------- #
# Static front end (must be mounted last)
# --------------------------------------------------------------------------- #
@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


def main() -> None:  # pragma: no cover
    import uvicorn

    uvicorn.run("bball.app.server:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":  # pragma: no cover
    main()
