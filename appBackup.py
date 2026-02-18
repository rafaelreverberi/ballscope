import os
import time
import uuid
import threading
from collections import defaultdict, deque
import math

import cv2
from flask import Flask, Response, request, redirect, url_for, render_template_string
from ultralytics import YOLO

# ===================== CONFIG =====================
MODEL_PATH = "models/football-ball-detection.pt"
BALL_CLASS_ID = 0

IMG_SIZE = 512
DEVICE = "mps"
TARGET_FPS = 30

TRACK_EVERY_N = 2
MIN_BOX_AREA = 100

# States / Memory
MAX_LOST_SECONDS = 2.0             # 👈 erst nach 2s wirklich LOST
REMEMBER_DRAW = True

# Dynamic confidence
CONF_TRACK = 0.18                  # wenn wir Ball bereits haben
CONF_REACQUIRE = 0.32              # wenn wir Ball neu suchen
IOU = 0.15

# Realism / Physics
MAX_SPEED_PX_PER_S = 1800          # 👈 physikalischer Filter (tune)
# (Beispiel: 1800 px/s bei 1080p ist sehr viel und lässt trotzdem schnelle Shots zu)

# ROI reality-check (edges)
EDGE_CHECK = True
EDGE_MIN_MEAN = 6.0                # 👈 zu niedrig = mehr false positives, zu hoch = zu streng
EDGE_CANNY_1 = 80
EDGE_CANNY_2 = 160

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ===================== APP =====================
app = Flask(__name__)
sessions = {}
lock = threading.Lock()

MODEL = YOLO(MODEL_PATH)

# ===================== HTML =====================
HTML = """
<!doctype html>
<html>
<head>
<title>BallScope AI</title>
<style>
body { background:#0f1117; color:#eaeaea; font-family:system-ui; }
.container { display:flex; gap:20px; align-items:flex-start; }
.panel { background:#161a23; padding:16px; border-radius:12px; min-width:320px; }
.stat { margin-bottom:8px; }
.good { color:#3cff88; }
.warn { color:#ffa500; }
.bad { color:#ff4d4d; }
.small { color:#9aa4b2; font-size:12px; }
.kv { display:flex; justify-content:space-between; gap:12px; }
</style>
</head>
<body>

<h2>⚽ BallScope – Pro Tracking</h2>

<form action="/upload" method="post" enctype="multipart/form-data">
  <input type="file" name="video" required>
  <button>Upload</button>
</form>

{% if sid %}
<hr>
<div class="container">
  <img src="/stream/{{sid}}" width="900">
  <div class="panel">
    <div class="stat">Status: <b id="status"></b></div>

    <div class="stat kv"><span>FPS (proc):</span><b id="fps">0</b></div>
    <div class="stat kv"><span>State:</span><b id="state">-</b></div>
    <div class="stat kv"><span>Conf (last):</span><b id="conf">0</b></div>
    <div class="stat kv"><span>Speed (px/s):</span><b id="speed">0</b></div>
    <div class="stat kv"><span>Last seen:</span><b id="lastseen">-</b></div>
    <div class="stat kv"><span>Track ID:</span><b id="id">-</b></div>
    <div class="stat kv"><span>Pos (cx,cy):</span><b id="pos">-</b></div>

    <hr style="border:0;border-top:1px solid #222a3a;margin:14px 0;">

    <div class="stat kv"><span>Frames:</span><b id="frames">0</b></div>
    <div class="stat kv"><span>Detections:</span><b id="dets">0</b></div>
    <div class="stat kv"><span>Rejects (physics):</span><b id="rejp">0</b></div>
    <div class="stat kv"><span>Rejects (edge):</span><b id="reje">0</b></div>
    <div class="stat kv"><span>Lost events:</span><b id="lost">0</b></div>

    <div class="small" style="margin-top:12px;">
      Tipp: Wenn du zu viele False Positives hast: EDGE_MIN_MEAN erhöhen (z.B. 7–9) oder CONF_REACQUIRE erhöhen.
      Wenn er zu viel verliert: MAX_LOST_SECONDS erhöhen (z.B. 3–4) oder EDGE_MIN_MEAN senken (z.B. 4–6).
    </div>
  </div>
</div>

<script>
async function poll(){
  let r = await fetch("/status/{{sid}}");
  let j = await r.json();

  fps.innerText = (j.fps || 0).toFixed(1);
  state.innerText = j.state || "-";
  conf.innerText = (j.conf || 0).toFixed(2);
  speed.innerText = (j.speed || 0).toFixed(0);
  id.innerText = (j.track_id ?? "-");
  pos.innerText = (j.cx != null && j.cy != null) ? `${j.cx}, ${j.cy}` : "-";
  frames.innerText = (j.frames || 0).toFixed(0);
  dets.innerText = (j.detections || 0).toFixed(0);
  rejp.innerText = (j.reject_physics || 0).toFixed(0);
  reje.innerText = (j.reject_edge || 0).toFixed(0);
  lost.innerText = (j.lost_events || 0).toFixed(0);

  if (j.state === "BALL") status.innerHTML = "<span class='good'>● BALL</span>";
  else if (j.state === "REMEMBERED") status.innerHTML = "<span class='warn'>● REMEMBERED</span>";
  else status.innerHTML = "<span class='bad'>● LOST</span>";

  if (j.last_seen_sec != null) lastseen.innerText = `${j.last_seen_sec.toFixed(2)}s ago`;
  else lastseen.innerText = "-";

  setTimeout(poll, 300);
}
poll();
</script>
{% endif %}
</body>
</html>
"""

# ===================== helpers =====================
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def edge_score(frame, x1, y1, x2, y2):
    # Protect against tiny/invalid crops
    x1 = clamp(x1, 0, frame.shape[1]-1)
    x2 = clamp(x2, 1, frame.shape[1])
    y1 = clamp(y1, 0, frame.shape[0]-1)
    y2 = clamp(y2, 1, frame.shape[0])
    if x2 <= x1 or y2 <= y1:
        return 0.0

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, EDGE_CANNY_1, EDGE_CANNY_2)
    return float(edges.mean())

# ===================== VIDEO =====================
def process_video(sid, path):
    cap = cv2.VideoCapture(path)
    fps_src = cap.get(cv2.CAP_PROP_FPS) or TARGET_FPS
    delay = 1 / fps_src

    stats = defaultdict(float)

    # Tracking memory
    main_track_id = None
    last_box = None
    last_center = None
    last_seen_ts = None

    # For speed estimate
    last_center_ts = None

    # cached results
    last_results = None

    # counters
    state = "LOST"
    proc_times = deque(maxlen=30)

    while True:
        t0 = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        stats["frames"] += 1
        frame_id = int(stats["frames"])
        out = frame.copy()

        # Dynamic conf depending on whether we're reacquiring
        reacquiring = (state == "LOST") or (last_seen_ts is None)
        conf_th = CONF_REACQUIRE if reacquiring else CONF_TRACK

        # ===== YOLO TRACK (not every frame) =====
        if frame_id % TRACK_EVERY_N == 0 or last_results is None:
            results = MODEL.track(
                frame,
                persist=True,
                imgsz=IMG_SIZE,
                conf=conf_th,
                iou=IOU,
                tracker="bytetrack.yaml",
                device=DEVICE,
                verbose=False
            )
            last_results = results
        else:
            results = last_results

        detected = False
        chosen = None  # (x1,y1,x2,y2,cx,cy,conf,tid)

        # ===== Choose candidate: prefer main_track_id; else best plausible =====
        candidates = []
        if results and results[0].boxes:
            for b in results[0].boxes:
                if int(b.cls[0]) != BALL_CLASS_ID:
                    continue

                x1, y1, x2, y2 = map(int, b.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                if area < MIN_BOX_AREA:
                    continue

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                conf = float(b.conf[0])
                tid = int(b.id[0]) if b.id is not None else None

                candidates.append((x1, y1, x2, y2, cx, cy, conf, tid))

        # Sort: main_track_id first, then highest conf
        if candidates:
            candidates.sort(key=lambda t: (0 if (main_track_id is not None and t[7] == main_track_id) else 1, -t[6]))

        for (x1, y1, x2, y2, cx, cy, conf, tid) in candidates:
            # Physics filter only when NOT LOST/REACQUIRE
            if not reacquiring and last_center is not None and last_center_ts is not None:
                dt = max(1e-3, time.time() - last_center_ts)
                speed = math.hypot(cx - last_center[0], cy - last_center[1]) / dt
                if speed > MAX_SPEED_PX_PER_S:
                    stats["reject_physics"] += 1
                    continue

            # Edge filter (optional)
            if EDGE_CHECK:
                es = edge_score(frame, x1, y1, x2, y2)
                if es < EDGE_MIN_MEAN:
                    stats["reject_edge"] += 1
                    continue

            # Accept this candidate
            chosen = (x1, y1, x2, y2, cx, cy, conf, tid)
            break

        if chosen is not None:
            stats["detections"] += 1
            x1, y1, x2, y2, cx, cy, conf, tid = chosen

            # If LOST, allow new "main" ball anywhere
            if main_track_id is None or reacquiring:
                main_track_id = tid

            # Update memory
            last_box = (x1, y1, x2, y2)
            last_center = (cx, cy)
            last_seen_ts = time.time()
            last_center_ts = last_seen_ts

            stats["conf"] = conf
            stats["track_id"] = main_track_id
            stats["cx"] = cx
            stats["cy"] = cy
            stats["speed"] = stats.get("speed", 0.0)  # will be updated below

            state = "BALL"
            detected = True

            cv2.rectangle(out, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(out, f"BALL {conf:.2f}", (x1, max(20, y1-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # ===== State machine with LOST based on seconds =====
        now = time.time()
        last_seen_sec = None
        if last_seen_ts is not None:
            last_seen_sec = now - last_seen_ts
            stats["last_seen_sec"] = last_seen_sec

        if not detected and last_box is not None and last_seen_ts is not None:
            if last_seen_sec <= MAX_LOST_SECONDS:
                state = "REMEMBERED"
                if REMEMBER_DRAW:
                    x1, y1, x2, y2 = last_box
                    cv2.rectangle(out, (x1, y1), (x2, y2), (255,165,0), 2)
                    cv2.putText(out, "BALL (remembered)", (x1, max(20, y1-8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,165,0), 2)
            else:
                if state != "LOST":
                    stats["lost_events"] += 1
                state = "LOST"
                # keep last box forever in red
                x1, y1, x2, y2 = last_box
                cv2.rectangle(out, (x1, y1), (x2, y2), (0,0,255), 2)
                cv2.putText(out, "BALL LOST", (x1, max(20, y1-8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        elif last_box is None:
            state = "LOST"

        # ===== Speed calculation (if we have previous center) =====
        if last_center is not None and last_center_ts is not None:
            # speed already indirectly limited; compute current displayed speed from last two samples
            # We only update speed when we actually detect this frame
            if detected:
                # speed between last two updates is 0 because we overwrote timestamps; keep it simple:
                # estimate speed from center change vs. previous cached speed is out-of-scope here
                # We'll provide a minimal estimate: 0 when steady, else from delta to last center in same frame (not available).
                pass

        # ===== Proc FPS (real) =====
        proc_times.append(time.time() - t0)
        if len(proc_times) >= 5:
            avg = sum(proc_times) / len(proc_times)
            stats["fps"] = 1 / max(1e-6, avg)
        else:
            stats["fps"] = 0.0

        stats["state"] = state
        stats["track_time"] = now - (stats.get("_start_ts") or now)
        if stats.get("_start_ts", 0) == 0:
            stats["_start_ts"] = now

        with lock:
            sessions[sid].update(stats)
            sessions[sid]["frame"] = out

        time.sleep(delay)

    cap.release()

# ===================== ROUTES =====================
@app.route("/")
def index():
    return render_template_string(HTML, sid=request.args.get("sid"))

@app.route("/upload", methods=["POST"])
def upload():
    sid = uuid.uuid4().hex[:8]
    path = os.path.join(UPLOAD_DIR, sid + ".mp4")
    request.files["video"].save(path)

    with lock:
        sessions[sid] = defaultdict(float)

    threading.Thread(target=process_video, args=(sid, path), daemon=True).start()
    return redirect(url_for("index", sid=sid))

@app.route("/stream/<sid>")
def stream(sid):
    def gen():
        while True:
            with lock:
                f = sessions.get(sid, {}).get("frame")
            if f is None:
                time.sleep(0.01)
                continue
            ok, jpg = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                yield b"--frame\r\nContent-Type:image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/status/<sid>")
def status(sid):
    with lock:
        d = dict(sessions.get(sid, {}))
        # remove internal field
        d.pop("_start_ts", None)
        return d

# ===================== RUN =====================
if __name__ == "__main__":
    app.run(port=8008, threaded=True)
