/* Shot Tracker Workbench — vanilla JS, no build step.
   State flows: session -> calibrate (H + rim) -> analyze -> review/labels -> zones -> results. */
"use strict";

const $ = (id) => document.getElementById(id);
const api = async (url, opts) => {
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error(`${url}: ${r.status} ${await r.text()}`);
  const ct = r.headers.get("content-type") || "";
  return ct.includes("json") ? r.json() : r;
};
const jpost = (url, body) => api(url, {
  method: "POST", headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

const S = {           // client state
  sid: null, probe: null, spec: "nba", court: null,
  calPoints: [], rimPoints: [], mode: "cal", overlay: null, rimPoly: null,
  labels: [], partition: null, zonePreview: null, drawPoly: [], customZones: {},
};

/* ---------------- tabs ---------------- */
document.querySelectorAll("#tabs button").forEach((b) =>
  b.addEventListener("click", () => {
    document.querySelectorAll("#tabs button, .tab").forEach((e) => e.classList.remove("active"));
    b.classList.add("active");
    $(`tab-${b.dataset.tab}`).classList.add("active");
    if (b.dataset.tab === "calibrate") drawCal();
    if (b.dataset.tab === "zones") refreshZoneUI();
    if (b.dataset.tab === "results") refreshResults().catch(console.warn);
    // Load saved labels/proposals when opening Review (survives reload/restart). Only when
    // not already loaded for this session, so in-memory edits aren't clobbered.
    if (b.dataset.tab === "review" && S.sid && !S.labels) loadLabels().catch(console.warn);
  }));

/* ---------------- session ---------------- */
async function loadCourt() { S.court = await api(`/api/court?spec=${S.spec}`); }

function setSession(state) {
  S.sid = state.sid; S.probe = state.probe; S.spec = state.spec || "nba";
  S.partition = state.partition || null;
  // Restore a saved calibration/rim (or clear it when switching to a fresh session) so it
  // survives page reloads and reopening the same clip — no redoing calibration.
  S.calPoints = (state.calibration && state.calibration.points)
    ? state.calibration.points.map((p) => ({ name: p.name, img: p.img, court: p.court }))
    : [];
  S.overlay = state.overlay_img || null;
  S.rimPoly = state.rim_polyline || null;
  S.rimPoints = []; S.zoomView = null; S.pendingZoom = false;
  S.labels = null;                   // force a reload of this session's labels on Review open
  if ($("court-spec")) $("court-spec").value = S.spec;
  $("session-info").textContent =
    `${state.name ? `"${state.name}" · ` : ""}${state.sid} · ` +
    `${state.probe.duration_s.toFixed(1)}s @ ${state.probe.fps.toFixed(0)}fps ` +
    `· ${state.probe.w}x${state.probe.h} · calibrated: ${!!state.calibration} · rim: ${!!state.rim}`;
  $("cal-time").max = Math.max(1, state.probe.duration_s - 0.05);
  $("review-video").src = `/api/sessions/${S.sid}/video`;
  if (state.calibration || state.rim)
    setStatus(`loaded saved setup — calibration: ${!!state.calibration}, rim: ${!!state.rim}. ` +
      `Overlay redrawn below; recalibrate only if you want to change it.`);
  loadCourt().then(() => { fillLandmarks(); drawCal(); });
}

$("btn-path").onclick = async () => {
  const state = await jpost("/api/sessions", { video_path: $("video-path").value.trim() });
  setSession(state); listSessions();
};
$("video-file").onchange = async (e) => {
  const fd = new FormData(); fd.append("file", e.target.files[0]);
  const state = await api("/api/sessions", { method: "POST", body: fd });
  setSession(state); listSessions();
};
$("court-spec").onchange = (e) => { S.spec = e.target.value; loadCourt().then(fillLandmarks); };

async function listSessions() {
  const xs = await api("/api/sessions");
  const host = $("session-list"); host.innerHTML = xs.length ? "" : "none yet";
  xs.forEach((s) => {
    const row = document.createElement("div"); row.className = "session-row";
    const open = document.createElement("button"); open.className = "session-open";
    open.textContent = `${s.name || s.sid} ${s.calibrated ? "📐" : ""}${s.rim ? "⭕" : ""}`;
    open.title = s.video;
    open.onclick = async () => setSession(await api(`/api/sessions/${s.sid}`));
    const rn = document.createElement("button"); rn.textContent = "✎"; rn.title = "rename";
    rn.onclick = async () => {
      const name = prompt("Session name:", s.name || "");
      if (name === null) return;
      await jpost(`/api/sessions/${s.sid}/rename`, { name });
      listSessions();
    };
    const del = document.createElement("button"); del.textContent = "🗑"; del.title = "delete";
    del.onclick = async () => {
      if (!confirm(`Delete "${s.name || s.sid}"? Removes its calibration, rim, labels and ` +
                   `analysis (your original video file is NOT deleted).`)) return;
      await api(`/api/sessions/${s.sid}`, { method: "DELETE" });
      if (S.sid === s.sid) { S.sid = null; S.labels = null; $("session-info").textContent = "no session"; }
      listSessions();
    };
    row.append(open, rn, del);
    host.appendChild(row);
  });
}
listSessions().catch(console.warn);

/* ---------------- calibrate ---------------- */
function fillLandmarks() {
  if (!S.court) return;
  const sel = $("landmark-select"); sel.innerHTML = "";
  Object.keys(S.court.landmarks).forEach((k) => {
    const o = document.createElement("option"); o.value = k;
    o.textContent = `${k} (${S.court.landmarks[k][0].toFixed(1)}, ${S.court.landmarks[k][1].toFixed(1)})`;
    sel.appendChild(o);
  });
  sel.onchange = drawCalMap;
  drawCalMap();
}

function drawCalMap() {
  if (!S.court) return;
  const cv = $("cal-map"), ctx = cv.getContext("2d");
  if (S.mode === "rim") {           // context help switches to the rim-clicking guide
    ctx.fillStyle = "#f7f3e8"; ctx.fillRect(0, 0, cv.width, cv.height);
    // Backboard: the orientation anchor — the ring hangs off it.
    ctx.fillStyle = "#b9b2a0"; ctx.fillRect(45, 42, 180, 26);
    ctx.fillStyle = "#1c1c1c"; ctx.font = "bold 11px sans-serif";
    ctx.fillText("BACKBOARD", 100, 59);
    ctx.strokeStyle = "#777"; ctx.lineWidth = 4;
    ctx.beginPath(); ctx.moveTo(135, 68); ctx.lineTo(135, 106); ctx.stroke();  // bracket
    ctx.strokeStyle = "#c1272d"; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.ellipse(135, 140, 90, 34, 0, 0, 7); ctx.stroke();
    const spots = [[225, 140, "R"], [45, 140, "L"], [135, 174, "C"], [135, 106, "B"],
                   [199, 163, ""], [71, 163, ""]];
    spots.forEach(([x, y, t]) => {
      ctx.fillStyle = "#1a3a5c";
      ctx.beginPath(); ctx.arc(x, y, 8, 0, 7); ctx.fill();
      if (t) { ctx.fillStyle = "white"; ctx.font = "bold 10px sans-serif"; ctx.fillText(t, x - 3.5, y + 3.5); }
    });
    ctx.fillStyle = "#1a3a5c"; ctx.font = "bold 10.5px sans-serif";
    ctx.fillText("backboard side (B)", 150, 100);
    ctx.fillText("court side (C)", 150, 192);
    ctx.fillText("L / R = the ellipse's widest points", 40, 214);
    ctx.fillStyle = "#1c1c1c"; ctx.font = "11px sans-serif";
    ctx.fillText("The ring hangs off the backboard —", 30, 240);
    ctx.fillText("find the board in YOUR frame to orient.", 30, 255);
    ctx.fillText("Click ~6 spots like these on the middle", 30, 270);
    ctx.fillText("of the ring line. ORDER DOESN'T MATTER —", 30, 285);
    ctx.fillText("covering all sides does. Zoom in first!", 30, 300);
    return;
  }
  const c = S.court, m = 18;
  const s = Math.min((cv.width - 2 * m) / (2 * c.sideline_x_m),
                     (cv.height - 2 * m) / (c.halfcourt_y_m + c.rim_from_baseline_m));
  const T = (p) => [cv.width / 2 + p[0] * s,
                    cv.height - m - (p[1] + c.rim_from_baseline_m) * s];
  ctx.fillStyle = "#f7f3e8"; ctx.fillRect(0, 0, cv.width, cv.height);
  ctx.strokeStyle = "#222"; ctx.lineWidth = 1.6;
  strokePoly(ctx, [[-c.sideline_x_m, -c.rim_from_baseline_m],
    [c.sideline_x_m, -c.rim_from_baseline_m], [c.sideline_x_m, c.halfcourt_y_m],
    [-c.sideline_x_m, c.halfcourt_y_m]].map(T), true);
  strokePoly(ctx, S.court.three.map(T));
  ctx.strokeStyle = "#888"; strokePoly(ctx, S.court.paint.map(T), true);
  const placed = new Set(S.calPoints.map((p) => p.name));
  const selName = $("landmark-select").value;
  Object.entries(S.court.landmarks).forEach(([name, xy]) => {
    const q = T(xy);
    ctx.fillStyle = name === selName ? "#c1272d" : placed.has(name) ? "#1a7837" : "#9a927e";
    ctx.beginPath(); ctx.arc(q[0], q[1], name === selName ? 7 : 4.5, 0, 7); ctx.fill();
  });
  ctx.fillStyle = "#c1272d"; ctx.font = "bold 12px sans-serif";
  ctx.fillText(selName || "", 8, 16);
}

const calCanvas = $("cal-canvas"), calCtx = calCanvas.getContext("2d");
let calImg = new Image(), calScale = { sx: 1, sy: 1 };

function frameURL(t) { return `/api/sessions/${S.sid}/frame?t=${t}&maxw=1280`; }

/* Zoomed view: crop rect (canvas coords) magnified to fill the canvas. */
function viewParams() {
  if (!S.zoomView) return { x0: 0, y0: 0, f: 1 };
  const f = S.zoomView.f;
  const w = calCanvas.width / f, h = calCanvas.height / f;
  const x0 = Math.max(0, Math.min(calCanvas.width - w, S.zoomView.xc - w / 2));
  const y0 = Math.max(0, Math.min(calCanvas.height - h, S.zoomView.yc - h / 2));
  return { x0, y0, f };
}
const tx = (p) => { const v = viewParams(); return [(p[0] - v.x0) * v.f, (p[1] - v.y0) * v.f]; };
const toCanvas = (pNative) => [pNative[0] * calScale.sx, pNative[1] * calScale.sy];

function drawCal() {
  if (!S.sid) return;
  const t = parseFloat($("cal-time").value);
  $("cal-time-label").textContent = `${t.toFixed(1)}s`;
  calImg = new Image();
  calImg.onload = () => {
    calCanvas.width = calImg.width; calCanvas.height = calImg.height;
    calScale.sx = calImg.width / S.probe.w; calScale.sy = calImg.height / S.probe.h;
    const v = viewParams();
    if (v.f > 1) {
      calCtx.imageSmoothingEnabled = false;
      calCtx.drawImage(calImg, v.x0, v.y0, calCanvas.width / v.f, calCanvas.height / v.f,
        0, 0, calCanvas.width, calCanvas.height);
    } else calCtx.drawImage(calImg, 0, 0);
    if (S.overlay) {
      calCtx.lineWidth = 2;
      for (const [name, poly] of Object.entries(S.overlay)) {
        calCtx.strokeStyle = name === "three" ? "#7CFC00" : "#00d5ff";
        strokePoly(calCtx, poly.map((p) => tx(toCanvas(p))));
      }
    }
    if (S.rimPoly) {
      calCtx.strokeStyle = "#00e5ff"; calCtx.lineWidth = 2;
      strokePoly(calCtx, S.rimPoly.map((p) => tx(toCanvas(p))), true);
    }
    S.calPoints.forEach((p) => { const q = tx(toCanvas(p.img)); dot(calCtx, q[0], q[1], "#ffd700", p.name); });
    S.rimPoints.forEach((p) => { const q = tx(toCanvas(p)); dot(calCtx, q[0], q[1], "#00e5ff"); });
  };
  calImg.src = frameURL(t);
}
function strokePoly(ctx, pts, close) {
  ctx.beginPath(); pts.forEach((p, i) => (i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])));
  if (close) ctx.closePath(); ctx.stroke();
}
function dot(ctx, x, y, color, label) {
  ctx.beginPath(); ctx.arc(x, y, 5, 0, 7);
  ctx.fillStyle = color; ctx.fill();
  ctx.lineWidth = 1.5; ctx.strokeStyle = "#111"; ctx.stroke();
  if (label) { ctx.font = "12px sans-serif"; ctx.fillStyle = color; ctx.fillText(label, x + 7, y - 5); }
}

function setStatus(msg) { $("cal-status").textContent = msg; }

$("cal-time").oninput = drawCal;
$("btn-mode-cal").onclick = () => setMode("cal");
$("btn-mode-rim").onclick = () => setMode("rim");
function setMode(m) {
  S.mode = m;
  $("btn-mode-cal").classList.toggle("active", m === "cal");
  $("btn-mode-rim").classList.toggle("active", m === "rim");
  setStatus(m === "rim"
    ? `RIM MODE — press "🔍 zoom to rim" first (the rim is small!), then click 5-6 points ` +
      `AROUND the ring edge — see the guide diagram on the right. Points so far: ` +
      `${S.rimPoints.length}`
    : `LANDMARK MODE — pick a landmark name, then click that exact spot on the frame ` +
      `(dropdown auto-advances). Placed: ${S.calPoints.length}. Then press "Calibrate".`);
  drawCalMap();
}
$("btn-zoom-rim").onclick = () => {
  S.pendingZoom = true;
  setStatus("click roughly ON the rim in the frame — the view will magnify 4x there");
};
$("btn-zoom-reset").onclick = () => { S.zoomView = null; S.pendingZoom = false; drawCal(); };
$("btn-undo").onclick = () => {
  (S.mode === "cal" ? S.calPoints : S.rimPoints).pop();
  setMode(S.mode);          // refresh the count in the status line
  drawCal();
};
$("btn-clear-cal").onclick = () => {
  S.calPoints = []; S.rimPoints = []; S.overlay = null; S.rimPoly = null;
  if ($("landmark-select").options.length) $("landmark-select").selectedIndex = 0;
  setMode("cal");
  setStatus("cleared — all points and overlay lines removed; start fresh from the first landmark");
  drawCal();
  drawCalMap();
};

calCanvas.addEventListener("click", (e) => {
  if (!S.sid) return;
  const r = calCanvas.getBoundingClientRect();
  const u = (e.clientX - r.left) * (calCanvas.width / r.width);
  const w = (e.clientY - r.top) * (calCanvas.height / r.height);
  const v = viewParams();
  const xc = v.x0 + u / v.f, yc = v.y0 + w / v.f;   // un-zoomed canvas coords
  if (S.pendingZoom) {
    S.zoomView = { xc, yc, f: 4 };
    S.pendingZoom = false;
    setMode("rim");
    setStatus("zoomed 4x on the rim — now click 5-6 points AROUND the ring edge " +
      "(front, back, both sides), then press Fit rim. 'reset view' zooms back out.");
    drawCal();
    return;
  }
  const x = xc / calScale.sx;
  const y = yc / calScale.sy;
  if (S.mode === "cal") {
    const name = $("landmark-select").value;
    S.calPoints = S.calPoints.filter((p) => p.name !== name);
    S.calPoints.push({ name, img: [x, y], court: S.court.landmarks[name] });
    const opts = $("landmark-select").options;
    const i = [...opts].findIndex((o) => o.value === name);
    if (i < opts.length - 1) $("landmark-select").selectedIndex = i + 1;
    setStatus(`placed "${name}" (${S.calPoints.length} landmarks). ` +
      `Keep going or press "Calibrate" (needs >= 4; spread beats count).`);
    drawCalMap();
  } else {
    S.rimPoints.push([x, y]);
    setStatus(`rim points: ${S.rimPoints.length} — need >= 5 spread around the ring, ` +
      `then press "Fit rim".`);
  }
  drawCal();
});

$("btn-calibrate").onclick = async () => {
  try {
    if (!S.sid) throw new Error("load a session first (Session tab)");
    const res = await jpost(`/api/sessions/${S.sid}/calibrate`,
      { spec: S.spec, points: S.calPoints });
    S.overlay = res.overlay_img;
    let msg = `calibrated: rms ${res.rms_all_px} px over all ${res.n_points} pts ` +
      `(${res.n_inliers} inliers at ${res.rms_px.toFixed(2)} px).`;
    const resid = Object.entries(res.residuals_px || {}).sort((a, b) => b[1] - a[1]);
    if (resid.length && resid[0][1] > 3)
      msg += ` Worst landmarks: ${resid.slice(0, 2).map(([n, e]) => `${n} ${e}px`).join(", ")}` +
        ` — re-click those, or that court dimension differs from the spec.`;
    if ((res.suspected_swaps || []).length)
      msg += ` ⚠ SIDES LOOK SWAPPED: ${res.suspected_swaps.join("; ")} — check the mini-map ` +
        `and re-click those two.`;
    const rank = res.spec_ranking || [];
    const cur = rank.find((r) => r.spec === S.spec), best = rank[0];
    if (cur && best && best.spec !== S.spec && best.rms_px * 1.5 < cur.rms_px)
      msg += ` ⚠ Your clicks fit "${best.spec}" much better ` +
        `(${best.rms_px}px vs ${cur.rms_px}px) — switch Court spec on the Session tab ` +
        `and recalibrate.`;
    else if (rank.length)
      msg += ` Spec fit: ${rank.map((r) => `${r.spec} ${r.rms_px}px`).join(" · ")}.`;
    setStatus(msg);
    drawCal();
  } catch (err) { setStatus(`calibrate: ${err.message}`); }
};
$("btn-fit-rim").onclick = async () => {
  try {
    if (!S.sid) throw new Error("load a session first (Session tab)");
    const res = await jpost(`/api/sessions/${S.sid}/rim`, { points: S.rimPoints });
    S.rimPoly = res.polyline;
    setStatus("rim fitted — the cyan ellipse should trace the rim; if not, undo and re-click");
    drawCal();
  } catch (err) { setStatus(`fit rim: ${err.message}`); }
};

/* ---------------- review ---------------- */
$("btn-analyze").onclick = async () => {
  if (!S.sid) { $("review-status").textContent = "load a session first"; return; }
  $("btn-analyze").disabled = true;
  $("review-status").textContent = "starting analysis…";
  const PHASE = { detecting: "finding the ball in each frame",
                  tracking: "linking the ball into trajectories",
                  locating: "detecting shots & locating them" };
  let polling = true;
  (async () => {
    while (polling) {
      try {
        const p = await api(`/api/sessions/${S.sid}/analyze/progress`);
        if (p.state in PHASE) {
          const pct = p.total ? Math.round((100 * p.done) / p.total) : 0;
          $("review-status").textContent =
            `⏳ ${PHASE[p.state]}… ${p.done}/${p.total} (${pct}%)`;
        }
      } catch (e) { /* progress is best-effort */ }
      await new Promise((r) => setTimeout(r, 1000));
    }
  })();
  try {
    const res = await jpost(`/api/sessions/${S.sid}/analyze`, {});
    polling = false;
    $("review-status").textContent =
      `✅ DONE — ${res.shots.length} proposals from ${res.n_frames} frames. ` +
      `Review & label them below (and add any it missed). Edits auto-save.`;
    await loadLabels(true);            // fresh proposals, even if an old labels.csv exists
  } catch (err) {
    polling = false;
    $("review-status").textContent = `❌ analyze failed: ${err.message}`;
  } finally { $("btn-analyze").disabled = false; }
};

function setAutosave(t) { const el = $("autosave-status"); if (el) el.textContent = t; }

let autosaveTimer = null;
function scheduleAutosave() {
  setAutosave("● unsaved…");
  clearTimeout(autosaveTimer);
  autosaveTimer = setTimeout(doAutosave, 1000);
}
async function doAutosave() {
  if (!S.sid || !S.labels) return;
  try {
    await jpost(`/api/sessions/${S.sid}/labels`, { rows: S.labels });  // writes labels.csv
    setAutosave("auto-saved ✓");
  } catch (e) { setAutosave(`⚠ autosave failed: ${e.message} — use Save labels.csv`); }
}

async function loadLabels(fresh) {
  const res = await api(`/api/sessions/${S.sid}/labels${fresh ? "?fresh=1" : ""}`);
  S.labels = res.rows.map((r) => ({ ...r }));
  if (!S.labels.length) {
    setAutosave("");
    $("review-status").textContent = 'no proposals yet — press "Run analysis".';
  } else {
    setAutosave(res.saved ? `loaded ${S.labels.length} saved labels ✓`
                          : `${S.labels.length} proposals — edits auto-save`);
  }
  renderEvents();
}

const OUTCOMES = ["make", "miss"], DIRS = ["", "short", "long", "left", "right", "short-left", "short-right", "long-left", "long-right"];
const TYPES = ["", "catch-and-shoot", "pull-up", "other"], QUAL = ["", "swish", "rim-in", "rattle"];
const ZONES = ["", "short-range", "midrange", "3PT"];   // pipeline-computed; correct when wrong

function playClip(startS, endS, meta) {
  const v = $("review-video");
  S.activeClip = { start: Math.max(0, startS), end: Math.max(startS + 0.3, endS), ...(meta || {}) };
  v.currentTime = S.activeClip.start;
  drawTimeline();
  v.play();
}

function drawTimeline() {
  const cv = $("clip-timeline"); if (!cv) return;
  const ctx = cv.getContext("2d"); ctx.clearRect(0, 0, cv.width, cv.height);
  const cap = $("clip-caption"), c = S.activeClip, v = $("review-video");
  if (!c) {
    ctx.fillStyle = "#999"; ctx.font = "13px sans-serif";
    ctx.fillText("press ▶ on a shot to play its clip", 12, 30);
    if (cap) cap.textContent = ""; return;
  }
  const span = c.end - c.start || 1;
  const X = (t) => 12 + ((t - c.start) / span) * (cv.width - 24);
  ctx.fillStyle = "#ddd8c8"; ctx.fillRect(12, 24, cv.width - 24, 8);           // track
  ctx.font = "bold 11px sans-serif";
  const xr = X(c.rel);                                                          // release marker
  ctx.strokeStyle = "#1a7837"; ctx.lineWidth = 3;
  ctx.beginPath(); ctx.moveTo(xr, 12); ctx.lineTo(xr, 40); ctx.stroke();
  ctx.fillStyle = "#1a7837"; ctx.fillText("◉ shot here", xr - 26, 10);
  if (c.rim) {                                                                  // rim-arrival marker
    const xm = X(c.rim); ctx.strokeStyle = "#4393c3"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(xm, 18); ctx.lineTo(xm, 40); ctx.stroke();
    ctx.fillStyle = "#4393c3"; ctx.fillText("rim", xm - 8, 51);
  }
  const xp = X(Math.min(Math.max(v.currentTime, c.start), c.end));             // playhead
  ctx.strokeStyle = "#c1272d"; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(xp, 20); ctx.lineTo(xp, 44); ctx.stroke();
  ctx.fillStyle = "#c1272d"; ctx.beginPath(); ctx.arc(xp, 20, 4, 0, 7); ctx.fill();
  if (cap) cap.textContent =
    `clip #${c.shotId}: ${c.start.toFixed(1)}–${c.end.toFixed(1)}s — THIS proposal's shot is ` +
    `at the green “◉ shot here” mark (${c.rel.toFixed(1)}s). If nothing happens there, it's not a shot.`;
}

$("review-video").addEventListener("timeupdate", () => {
  const c = S.activeClip; if (!c) return;
  if ($("review-video").currentTime >= c.end) $("review-video").pause();
  drawTimeline();
});

function renderEvents() {
  const el = $("event-list"); el.innerHTML = "";
  const hideExcl = $("hide-excluded") && $("hide-excluded").checked;
  const nExcl = S.labels.filter((r) => r.verified === "excluded").length;
  const nLabeled = S.labels.filter((r) => r.verified && r.verified !== "excluded").length;
  if ($("event-summary"))
    $("event-summary").textContent =
      `${S.labels.length} proposals · ${nLabeled} labeled · ${nExcl} excluded (not counted)`;
  S.labels.forEach((row, i) => {
    if (hideExcl && row.verified === "excluded") return;
    const rel = +row.t_release_s || 0, rimT = +row.t_rim_s || rel + 1.5;
    const pre = parseFloat(($("clip-pre") || {}).value) || 3;
    const post = parseFloat(($("clip-post") || {}).value) || 5;
    const cs = Math.max(0, rel - pre), ce = Math.max(rimT, rel + 2.0) + post;   // dedicated window
    // Flag a likely duplicate: an earlier, non-excluded proposal whose release is within
    // ~2.5 s (one flight). The FSM cooldown catches most, but broken real-footage tracks
    // can re-trigger — surfacing it lets you exclude the extra.
    let dupOf = null;
    for (let j = 0; j < i; j++) {
      if (S.labels[j].verified === "excluded") continue;
      if (Math.abs((+S.labels[j].t_release_s || 0) - rel) < 2.5) { dupOf = S.labels[j].shot_id; break; }
    }
    const d = document.createElement("div");
    d.className = `event ${row.verified || ""}`;
    d.innerHTML = `<b>#${row.shot_id}</b> ` +
      `<button class="seek">▶ ${cs.toFixed(1)}–${ce.toFixed(1)}s (shot @ ${rel.toFixed(1)})</button>`;
    if (dupOf !== null && row.verified !== "excluded") {
      const w = document.createElement("span");
      w.className = "dup-warn"; w.textContent = ` ⚠ overlaps #${dupOf} — keep the real one, ✕ the other`;
      d.appendChild(w);
    }
    d.appendChild(select(OUTCOMES, row.outcome, (v) => edit(i, "outcome", v)));
    d.appendChild(select(DIRS, row.miss_direction, (v) => edit(i, "miss_direction", v), "dir"));
    d.appendChild(select(TYPES, row.shot_type, (v) => edit(i, "shot_type", v), "type"));
    d.appendChild(select(QUAL, row.make_quality, (v) => edit(i, "make_quality", v), "quality"));
    const zoneWrap = document.createElement("span");
    zoneWrap.className = "muted"; zoneWrap.textContent = " zone: ";
    zoneWrap.appendChild(select(ZONES, row.zone, (v) => edit(i, "zone", v), "zone"));
    d.appendChild(zoneWrap);
    const ex = document.createElement("button");
    ex.textContent = row.verified === "excluded" ? "↺ restore" : "✕ not a shot";
    ex.onclick = () => {
      row.verified = row.verified === "excluded" ? "corrected" : "excluded";
      renderEvents(); scheduleAutosave();
    };
    d.appendChild(ex);
    d.querySelector(".seek").onclick = () =>
      playClip(cs, ce, { rel, rim: rimT, shotId: row.shot_id });
    el.appendChild(d);
  });
}
if ($("hide-excluded")) $("hide-excluded").onchange = () => { if (S.labels) renderEvents(); };
["clip-pre", "clip-post"].forEach((id) => {
  const el = $(id); if (el) el.onchange = () => { if (S.labels) renderEvents(); };
});

function select(opts, val, on, label) {
  const s = document.createElement("select");
  opts.forEach((o) => { const e = document.createElement("option"); e.value = o; e.textContent = o || (label ? `(${label})` : "(—)"); s.appendChild(e); });
  s.value = val || ""; s.onchange = () => on(s.value);
  return s;
}
function edit(i, k, v) {
  S.labels[i][k] = v;
  if (!S.labels[i].verified) S.labels[i].verified = "corrected";
  scheduleAutosave();
}

$("btn-add-missed").onclick = () => {
  const t = $("review-video").currentTime;
  S.labels.push({ shot_id: S.labels.length, t_release_s: t.toFixed(2), t_rim_s: (t + 1.5).toFixed(2),
    outcome: "make", zone: "", spot_id: "", shot_type: "", miss_direction: "", make_quality: "",
    court_x_m: "", court_y_m: "", verified: "corrected", source: "manual" });
  renderEvents(); scheduleAutosave();
};
$("btn-save-labels").onclick = async () => {
  clearTimeout(autosaveTimer);
  S.labels.forEach((r) => { if (!r.verified) r.verified = "accepted"; });  // "I reviewed the rest"
  const res = await jpost(`/api/sessions/${S.sid}/labels`, { rows: S.labels });
  setAutosave("saved ✓");
  $("review-status").textContent = `saved ${res.n} rows → ${res.saved}`;
};

/* ---------------- zones ---------------- */
const zoneCanvas = $("zone-canvas"), zctx = zoneCanvas.getContext("2d");
const PARAM_DEFS = {
  basic3: [["interior_radius_m", "interior radius (m)", 2.13]],
  extended: [["interior_radius_m", "interior (m)", 2.13], ["mid_split_radius_m", "mid split (m)", 5.2],
    ["deep_three_offset_m", "deep-3 offset (m)", 0.9]],
  spots: [["interior_radius_m", "interior (m)", 2.13], ["corner_angle_deg", "corner ∠°", 27],
    ["wing_angle_deg", "wing ∠°", 65]],
  polygons: [],
};

function courtToCanvas(p) {
  const c = S.court, m = 30;                       // margin px
  const w = zoneCanvas.width - 2 * m;
  const scale = w / (2 * c.sideline_x_m);
  const x = m + (p[0] + c.sideline_x_m) * scale;
  const y = zoneCanvas.height - m - (p[1] + c.rim_from_baseline_m) * scale;
  return [x, y];
}
function canvasToCourt(x, y) {
  const c = S.court, m = 30;
  const scale = (zoneCanvas.width - 2 * m) / (2 * c.sideline_x_m);
  return [(x - m) / scale - c.sideline_x_m,
          (zoneCanvas.height - m - y) / scale - c.rim_from_baseline_m];
}

function drawZoneCanvas(part) {
  if (!S.court) return;
  const c = S.court;
  zctx.fillStyle = "#f7f3e8"; zctx.fillRect(0, 0, zoneCanvas.width, zoneCanvas.height);
  zctx.strokeStyle = "#222"; zctx.lineWidth = 2;
  strokePoly(zctx, [[-c.sideline_x_m, -c.rim_from_baseline_m], [c.sideline_x_m, -c.rim_from_baseline_m],
    [c.sideline_x_m, c.halfcourt_y_m], [-c.sideline_x_m, c.halfcourt_y_m]].map(courtToCanvas), true);
  strokePoly(zctx, S.court.three.map(courtToCanvas));
  zctx.strokeStyle = "#888"; strokePoly(zctx, S.court.paint.map(courtToCanvas), true);
  const rim = courtToCanvas([0, 0]);
  dot(zctx, rim[0], rim[1], "#c1272d");
  if (part) {
    zctx.lineWidth = 1.6;
    Object.entries(part.boundaries).forEach(([name, poly], i) => {
      zctx.strokeStyle = ["#1a7837", "#4393c3", "#e08214", "#9944aa"][i % 4];
      strokePoly(zctx, poly.map(courtToCanvas));
    });
  }
  Object.entries(S.customZones).forEach(([name, poly]) => {
    zctx.strokeStyle = "#9944aa"; zctx.lineWidth = 2;
    strokePoly(zctx, poly.map(courtToCanvas), true);
    const c0 = courtToCanvas(poly[0]); zctx.fillStyle = "#9944aa";
    zctx.fillText(name, c0[0] + 4, c0[1] - 4);
  });
  if (S.drawPoly.length) {
    zctx.strokeStyle = "#cc0077"; zctx.lineWidth = 1.5;
    strokePoly(zctx, S.drawPoly.map(courtToCanvas));
    S.drawPoly.forEach((p) => { const q = courtToCanvas(p); dot(zctx, q[0], q[1], "#cc0077"); });
  }
}

function currentPartitionSpec() {
  const mode = $("zone-preset").value;
  if (mode === "polygons")
    return { mode: "polygons", name: "custom", default_zone: "other", polygons: S.customZones };
  const spec = { mode, court: S.spec };
  PARAM_DEFS[mode].forEach(([k]) => { spec[k] = parseFloat($(`param-${k}`).value); });
  return spec;
}

async function refreshZoneUI() {
  if (!S.court) await loadCourt();
  const mode = $("zone-preset").value;
  $("zone-draw-help").style.display = mode === "polygons" ? "" : "none";
  const box = $("zone-params"); box.innerHTML = "";
  PARAM_DEFS[mode].forEach(([k, label, dflt]) => {
    const l = document.createElement("label");
    l.innerHTML = `${label} <input id="param-${k}" type="number" step="0.05" value="${dflt}">`;
    box.appendChild(l);
    l.querySelector("input").onchange = refreshPreview;
  });
  await refreshPreview();
}
async function refreshPreview() {
  const spec = currentPartitionSpec();
  if (spec.mode === "polygons" && !Object.keys(S.customZones).length) { drawZoneCanvas(null); return; }
  try {
    S.zonePreview = await jpost("/api/zones/preview", spec);
    drawZoneCanvas(S.zonePreview);
    $("zone-status").textContent = `zones: ${S.zonePreview.zones.join(", ")}`;
  } catch (err) { $("zone-status").textContent = err.message; }
}
$("zone-preset").onchange = refreshZoneUI;

zoneCanvas.addEventListener("click", (e) => {
  if ($("zone-preset").value !== "polygons") return;
  const r = zoneCanvas.getBoundingClientRect();
  S.drawPoly.push(canvasToCourt((e.clientX - r.left) * (zoneCanvas.width / r.width),
                                (e.clientY - r.top) * (zoneCanvas.height / r.height)));
  drawZoneCanvas(S.zonePreview);
});
zoneCanvas.addEventListener("dblclick", () => {
  if ($("zone-preset").value !== "polygons" || S.drawPoly.length < 3) return;
  const name = prompt("zone name?", `zone-${Object.keys(S.customZones).length + 1}`);
  if (name) S.customZones[name] = S.drawPoly.slice(0, -1);   // drop dbl-click dup vertex
  S.drawPoly = [];
  refreshPreview();
});

$("btn-zone-apply").onclick = async () => {
  if (!S.sid) { $("zone-status").textContent = "load a session first"; return; }
  const res = await jpost(`/api/sessions/${S.sid}/zones`, currentPartitionSpec());
  S.partition = currentPartitionSpec();
  $("zone-status").textContent = `applied "${res.name}" to session` +
    (res.overlay_img ? " — boundaries also projected onto the video frame (Calibrate tab overlay)" : "");
  if (res.overlay_img) { S.overlay = { ...(S.overlay || {}), ...res.overlay_img }; }
};

/* ---------------- results ---------------- */
async function refreshResults() {
  if (!S.sid) return;
  const res = await api(`/api/sessions/${S.sid}/results`);
  $("results-summary").textContent =
    `partition: ${res.partition} · attempts ${res.attempts} · makes ${res.makes}` +
    (res.fg_pct != null ? ` · FG ${(100 * res.fg_pct).toFixed(0)}%` : "");
  if (!S.court) await loadCourt();
  drawZoneCanvas(S.zonePreview);
  const cctx = $("chart-canvas").getContext("2d");
  cctx.drawImage(zoneCanvas, 0, 0);
  res.chart.forEach((s) => {
    if (!s.xy) return;
    const p = courtToCanvas(s.xy);
    cctx.fillStyle = s.outcome === "make" ? "#1a7837" : "#c1272d";
    cctx.beginPath(); cctx.arc(p[0], p[1], s.on_line ? 7 : 5, 0, 7); cctx.fill();
    if (s.on_line) { cctx.strokeStyle = "#e08214"; cctx.lineWidth = 2; cctx.stroke(); }
  });
  const t = $("zone-table");
  t.innerHTML = "<tr><th>zone</th><th>attempts</th><th>makes</th><th>FG%</th></tr>";
  Object.entries(res.by_zone).forEach(([z, v]) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${z}</td><td>${v.attempts}</td><td>${v.makes}</td>` +
      `<td>${v.attempts ? (100 * v.makes / v.attempts).toFixed(0) : "—"}%</td>`;
    t.appendChild(tr);
  });
}
$("btn-refresh-results").onclick = refreshResults;
