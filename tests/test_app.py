"""Web workbench API: sessions, frames, calibration round-trip, rim fit, zones, labels, results."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

import bball.app.server as srv
from bball.lift.court_model import get_court, landmark_points
from bball.lift.homography import apply_homography


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(srv, "DATA_ROOT", tmp_path / "app_sessions")
    return TestClient(srv.app)


@pytest.fixture()
def video(tmp_path):
    import cv2

    path = tmp_path / "clip.mp4"
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20, (320, 240))
    for i in range(60):
        frame = np.full((240, 320, 3), 40, np.uint8)
        cv2.circle(frame, (40 + 3 * i, 120), 6, (0, 120, 255), -1)
        out.write(frame)
    out.release()
    return path


def _mksession(client, video):
    r = client.post("/api/sessions", json={"video_path": str(video)})
    assert r.status_code == 200, r.text
    return r.json()


def test_session_create_and_frame(client, video):
    s = _mksession(client, video)
    assert s["probe"]["n_frames"] == 60
    r = client.get(f"/api/sessions/{s['sid']}/frame?t=1.0")
    assert r.status_code == 200 and r.content[:2] == b"\xff\xd8"  # JPEG magic
    assert client.get("/api/sessions").json()[0]["sid"] == s["sid"]
    # range-aware video serving
    rv = client.get(f"/api/sessions/{s['sid']}/video", headers={"range": "bytes=0-99"})
    assert rv.status_code == 206 and len(rv.content) == 100


def test_court_endpoint(client):
    c = client.get("/api/court?spec=hs").json()
    assert c["spec"] == "hs" and "three_apex" in c["landmarks"] and len(c["three"]) > 10


def test_calibrate_round_trip(client, video):
    s = _mksession(client, video)
    court = get_court("nba")
    H_true = np.array([[90.0, 4.0, 640.0], [-3.0, -55.0, 600.0], [0.0, 0.004, 1.0]])
    lms = landmark_points(court)
    names = ["baseline_left_corner", "baseline_right_corner", "ft_left", "ft_right",
             "three_apex", "corner_three_right"]
    pts = []
    for n in names:
        cxy = lms[n]
        ixy = apply_homography(H_true, np.atleast_2d(cxy))[0]
        pts.append({"name": n, "court": cxy.tolist(), "img": ixy.tolist()})
    r = client.post(f"/api/sessions/{s['sid']}/calibrate", json={"spec": "nba", "points": pts})
    body = r.json()
    assert r.status_code == 200, r.text
    assert body["rms_px"] < 1e-3 and body["n_inliers"] == len(names)
    # overlay polylines: projected apex should match H_true's projection
    state = client.get(f"/api/sessions/{s['sid']}").json()
    H = np.array(state["calibration"]["H_court2img"])
    apex_img = apply_homography(H, np.atleast_2d(lms["three_apex"]))[0]
    apex_true = apply_homography(H_true, np.atleast_2d(lms["three_apex"]))[0]
    assert np.allclose(apex_img, apex_true, atol=1e-3)


def test_rim_fit(client, video):
    s = _mksession(client, video)
    tt = np.linspace(0, 2 * np.pi, 9)[:-1]
    pts = np.stack([200 + 50 * np.cos(tt), 100 + 18 * np.sin(tt)], axis=1)
    r = client.post(f"/api/sessions/{s['sid']}/rim", json={"points": pts.tolist()})
    rim = r.json()["rim"]
    assert r.status_code == 200
    assert abs(rim["cx"] - 200) < 1e-3 and abs(rim["cy"] - 100) < 1e-3
    assert abs(max(rim["a"], rim["b"]) - 50) < 0.1 and abs(min(rim["a"], rim["b"]) - 18) < 0.1


def test_zones_presets_preview_and_apply(client, video):
    s = _mksession(client, video)
    presets = client.get("/api/zones/presets?spec=nba").json()
    assert {p["name"] for p in presets} == {"basic3", "extended", "spots"}
    prev = client.post("/api/zones/preview", json={
        "mode": "extended", "court": "nba", "interior_radius_m": 2.13,
        "mid_split_radius_m": 5.0, "deep_three_offset_m": 0.8}).json()
    assert "deep_three_line" in prev["boundaries"]
    custom = client.post(f"/api/sessions/{s['sid']}/zones", json={
        "mode": "polygons", "name": "mine", "default_zone": "rest",
        "polygons": {"left-block": [[-2, 0], [-1, 0], [-1, 1.5], [-2, 1.5]]}}).json()
    assert "left-block_outline" in custom["boundaries"]
    assert client.get(f"/api/sessions/{s['sid']}").json()["partition"]["mode"] == "polygons"


def test_labels_and_results(client, video):
    s = _mksession(client, video)
    rows = [
        {"shot_id": 0, "t_release_s": 1.0, "t_rim_s": 2.2, "outcome": "make", "zone": "",
         "spot_id": "", "shot_type": "", "miss_direction": "", "make_quality": "",
         "court_x_m": 0.0, "court_y_m": 7.5, "verified": "accepted", "source": "pipeline"},
        {"shot_id": 1, "t_release_s": 5.0, "t_rim_s": 6.0, "outcome": "miss", "zone": "",
         "spot_id": "", "shot_type": "", "miss_direction": "short", "make_quality": "",
         "court_x_m": 2.0, "court_y_m": 4.0, "verified": "corrected", "source": "manual"},
        {"shot_id": 2, "t_release_s": 8.0, "t_rim_s": 9.0, "outcome": "make", "zone": "",
         "spot_id": "", "shot_type": "", "miss_direction": "", "make_quality": "",
         "court_x_m": "", "court_y_m": "", "verified": "excluded", "source": "pipeline"},
    ]
    r = client.post(f"/api/sessions/{s['sid']}/labels", json={"rows": rows})
    assert r.status_code == 200 and r.json()["n"] == 3
    got = client.get(f"/api/sessions/{s['sid']}/labels").json()
    assert got["saved"] and len(got["rows"]) == 3

    res = client.get(f"/api/sessions/{s['sid']}/results").json()
    assert res["attempts"] == 2 and res["makes"] == 1          # excluded row dropped
    assert res["partition"] == "basic3"
    zones = {c["zone"] for c in res["chart"] if c["xy"]}
    assert zones == {"three", "midrange"}                      # rebucketted from positions


def test_analyze_requires_rim(client, video):
    s = _mksession(client, video)
    r = client.post(f"/api/sessions/{s['sid']}/analyze", json={})
    assert r.status_code == 400 and "rim" in r.json()["detail"]


def test_calibrate_merges_coincident_hs_landmarks(client, video):
    s = _mksession(client, video)
    court = get_court("hs")
    lms = landmark_points(court)
    # On HS courts three_apex == top_of_key (both 6.02 m): clicked at slightly
    # different pixels, they must merge into one correspondence, not fight.
    assert np.allclose(lms["three_apex"], lms["top_of_key"], atol=0.01)
    H_true = np.array([[80.0, 2.0, 620.0], [-2.0, -50.0, 580.0], [0.0, 0.003, 1.0]])
    names = ["baseline_left_corner", "baseline_right_corner", "ft_left", "ft_right",
             "three_apex", "top_of_key"]
    pts = []
    for n in names:
        ixy = apply_homography(H_true, np.atleast_2d(lms[n]))[0]
        jitter = [1.5, -1.0] if n == "top_of_key" else [0.0, 0.0]
        pts.append({"name": n, "court": lms[n].tolist(),
                    "img": (ixy + jitter).tolist()})
    r = client.post(f"/api/sessions/{s['sid']}/calibrate", json={"spec": "hs", "points": pts})
    body = r.json()
    assert r.status_code == 200, r.text
    assert body["n_merged_duplicates"] == 1 and body["n_points"] == 5
    assert body["rms_px"] < 2.0


def test_calibrate_spec_ranking_identifies_true_court(client, video):
    s = _mksession(client, video)
    hs = get_court("hs")
    lms = landmark_points(hs)
    H_true = np.array([[85.0, 3.0, 630.0], [-2.5, -52.0, 590.0], [0.0, 0.0035, 1.0]])
    names = ["baseline_left_corner", "baseline_right_corner", "lane_baseline_left",
             "lane_baseline_right", "ft_left", "ft_right", "three_apex"]
    pts = [{"name": n, "court": lms[n].tolist(),
            "img": apply_homography(H_true, np.atleast_2d(lms[n]))[0].tolist()}
           for n in names]
    # user selected the WRONG spec (nba) for clicks that came from an hs court
    nba = landmark_points(get_court("nba"))
    wrong = [{**p, "court": nba[p["name"]].tolist()} for p in pts]
    r = client.post(f"/api/sessions/{s['sid']}/calibrate", json={"spec": "nba", "points": wrong})
    body = r.json()
    assert r.status_code == 200, r.text
    assert body["spec_ranking"][0]["spec"] == "hs"
    assert body["spec_ranking"][0]["rms_px"] < 0.5
    nba_rank = next(r for r in body["spec_ranking"] if r["spec"] == "nba")
    assert nba_rank["rms_px"] > 3.0                 # wrong spec fits worse over ALL points
    assert body["rms_all_px"] > body["spec_ranking"][0]["rms_px"]
    assert "three_apex" in body["residuals_px"]


def test_calibrate_overrides_stale_client_court_coords(client, video):
    """Named points get court coords from the POSTED spec — stale client values from a
    mid-flow spec switch cannot poison the fit."""
    s = _mksession(client, video)
    hs = landmark_points(get_court("hs"))
    H_true = np.array([[85.0, 3.0, 630.0], [-2.5, -52.0, 590.0], [0.0, 0.0035, 1.0]])
    names = ["baseline_left_corner", "baseline_right_corner", "ft_left", "ft_right",
             "three_apex", "lane_baseline_left"]
    nba = landmark_points(get_court("nba"))
    pts = [{"name": n, "court": nba[n].tolist(),  # STALE nba coords cached client-side
            "img": apply_homography(H_true, np.atleast_2d(hs[n]))[0].tolist()}
           for n in names]
    r = client.post(f"/api/sessions/{s['sid']}/calibrate", json={"spec": "hs", "points": pts})
    body = r.json()
    assert r.status_code == 200, r.text
    assert body["rms_all_px"] < 0.01  # server-side hs lookup wins; stale nba ignored


def test_calibrate_detects_swapped_sides(client, video):
    s = _mksession(client, video)
    lms = landmark_points(get_court("nba"))
    H_true = np.array([[85.0, 3.0, 630.0], [-2.5, -52.0, 590.0], [0.0, 0.0035, 1.0]])
    names = ["baseline_left_corner", "baseline_right_corner", "ft_left", "ft_right",
             "three_apex", "lane_baseline_left", "lane_baseline_right"]
    pts = []
    for n in names:
        img = apply_homography(H_true, np.atleast_2d(lms[n]))[0].tolist()
        pts.append({"name": n, "court": lms[n].tolist(), "img": img})
    # user swapped the two ft clicks (left/right ambiguity from a diagonal camera)
    i_a = names.index("ft_left"); i_b = names.index("ft_right")
    pts[i_a]["img"], pts[i_b]["img"] = pts[i_b]["img"], pts[i_a]["img"]
    r = client.post(f"/api/sessions/{s['sid']}/calibrate", json={"spec": "nba", "points": pts})
    body = r.json()
    assert r.status_code == 200, r.text
    assert body["rms_all_px"] > 3.0
    assert any("ft_left" in sw for sw in body["suspected_swaps"])


def test_overlay_math_round_trip(client, video):
    """The overlay polylines projected back through the inverse homography must land on
    the canonical court lines — certifies the server-side overlay geometry end to end."""
    s = _mksession(client, video)
    from bball.lift.court_model import three_point_polyline
    court = get_court("nba")
    lms = landmark_points(court)
    H_true = np.array([[85.0, 3.0, 630.0], [-2.5, -52.0, 590.0], [0.0, 0.0035, 1.0]])
    names = ["baseline_left_corner", "baseline_right_corner", "ft_left", "ft_right",
             "three_apex", "corner_three_right"]
    pts = [{"name": n, "court": lms[n].tolist(),
            "img": apply_homography(H_true, np.atleast_2d(lms[n]))[0].tolist()}
           for n in names]
    r = client.post(f"/api/sessions/{s['sid']}/calibrate", json={"spec": "nba", "points": pts})
    overlay_three = np.array(r.json()["overlay_img"]["three"])
    state = client.get(f"/api/sessions/{s['sid']}").json()
    H_inv = np.array(state["calibration"]["H_img2court"])
    back = apply_homography(H_inv, overlay_three)
    assert np.allclose(back, three_point_polyline(court), atol=1e-6)


def test_analyze_streams_end_to_end(client, video):
    """Analyze runs the streaming pipeline over a clip and returns the analysis structure
    without loading all frames at once (the OOM regression)."""
    s = _mksession(client, video)
    tt = np.linspace(0, 2 * np.pi, 9)[:-1]
    rim_pts = np.stack([160 + 30 * np.cos(tt), 60 + 11 * np.sin(tt)], axis=1)
    client.post(f"/api/sessions/{s['sid']}/rim", json={"points": rim_pts.tolist()})
    r = client.post(f"/api/sessions/{s['sid']}/analyze", json={})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_frames"] == 60 and isinstance(body["shots"], list)  # 20 fps -> stride 1
    # persisted so the Review tab can load proposals
    assert client.get(f"/api/sessions/{s['sid']}").json()["analysis"]["n_frames"] == 60


def test_calibration_and_rim_persist_and_restore(client, video):
    s = _mksession(client, video)
    court = get_court("nba")
    H_true = np.array([[90.0, 4.0, 640.0], [-3.0, -55.0, 600.0], [0.0, 0.004, 1.0]])
    lms = landmark_points(court)
    names = ["baseline_left_corner", "baseline_right_corner", "ft_left", "ft_right",
             "three_apex", "corner_three_right"]
    pts = [{"name": n, "court": lms[n].tolist(),
            "img": apply_homography(H_true, np.atleast_2d(lms[n]))[0].tolist()} for n in names]
    client.post(f"/api/sessions/{s['sid']}/calibrate", json={"spec": "nba", "points": pts})
    tt = np.linspace(0, 2 * np.pi, 9)[:-1]
    client.post(f"/api/sessions/{s['sid']}/rim",
                json={"points": np.stack([160 + 30 * np.cos(tt), 60 + 11 * np.sin(tt)], 1).tolist()})
    # A fresh GET (as after a page reload) carries redraw data + the clicked points.
    got = client.get(f"/api/sessions/{s['sid']}").json()
    assert got["calibration"]["points"] and "three" in got["overlay_img"]
    assert len(got["rim_polyline"]) > 10 and got["rim"] is not None


def test_reopen_same_video_reuses_session(client, video):
    s = _mksession(client, video)
    court = get_court("nba")
    H_true = np.array([[90.0, 4.0, 640.0], [-3.0, -55.0, 600.0], [0.0, 0.004, 1.0]])
    lms = landmark_points(court)
    names = ["baseline_left_corner", "baseline_right_corner", "ft_left", "ft_right", "three_apex"]
    pts = [{"name": n, "court": lms[n].tolist(),
            "img": apply_homography(H_true, np.atleast_2d(lms[n]))[0].tolist()} for n in names]
    client.post(f"/api/sessions/{s['sid']}/calibrate", json={"spec": "nba", "points": pts})
    # Opening the same path again returns the SAME session, calibration intact.
    again = client.post("/api/sessions", json={"video_path": str(video)}).json()
    assert again["sid"] == s["sid"] and again["calibration"] is not None
    assert "overlay_img" in again
