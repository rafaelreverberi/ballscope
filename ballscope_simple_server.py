from flask import Flask, Response, request, render_template_string, redirect, url_for
import cv2
from ultralytics import YOLO
import threading
import torch
import os

# ================= CONFIG =================
MODEL_PATH = "models/football-ball-detection.pt"     # dein Roboflow-exportiertes .pt
BALL_CLASS_ID = 0
IMG_SIZE = 640

confidence = 0.5           # Startwert
lock = threading.Lock()

# ================ UPLOAD CONFIG ================
UPLOAD_PATH = "uploads"
os.makedirs(UPLOAD_PATH, exist_ok=True)

video_source = {"mode": "webcam", "path": None}

# ================= DEVICE =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("🔥 YOLO DEVICE:", DEVICE)

# (optional, hilft auf Jetson oft)
try:
    torch.backends.cudnn.benchmark = True
except Exception:
    pass

# ================= YOLO =================
model = YOLO(MODEL_PATH)
model.to(DEVICE)

# 🔥 WICHTIG: nur EINMAL fuse() -> verhindert Jetson NVMap/NVML Crash
try:
    model.fuse()
except Exception as e:
    print("⚠️ fuse() skipped:", e)

# Safe Device-Print (funktioniert zuverlässig)
try:
    print("📦 Model loaded on:", next(model.model.parameters()).device)
except Exception as e:
    print("📦 Model loaded (device unknown):", e)

# ================= FLASK =================
app = Flask(__name__)

HTML = """
<!doctype html>
<html>
<head>
<title>YOLO Ball Tracking</title>
<style>
body { background:#111; color:white; font-family:Arial; text-align:center; }
input[type=range] { width:300px; }
</style>
</head>
<body>

<h2>🎯 YOLO Ball Tracking</h2>
<form action="/upload" method="post" enctype="multipart/form-data">
  <input type="file" name="video" accept="video/*">
  <button type="submit">Upload Video</button>
  <button type="button" onclick="fetch('/use_webcam')">Use Webcam</button>
</form>
<br>

<p>Confidence: <span id="val">50</span>%</p>
<input type="range" min="1" max="99" value="50" id="slider">

<br><br>
<img src="/stream" width="720">

<script>
const slider = document.getElementById("slider");
const val = document.getElementById("val");

slider.oninput = () => {
  val.innerText = slider.value;
  fetch("/confidence", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({value: slider.value})
  });
};
</script>

</body>
</html>
"""

# ================= ROUTES =================
@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/confidence", methods=["POST"])
def set_conf():
    global confidence
    with lock:
        confidence = float(request.json["value"]) / 100.0
    return ("", 204)

# ============= VIDEO UPLOAD ROUTES =============
@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("video")
    if not file:
        return redirect(url_for("index"))

    path = os.path.join(UPLOAD_PATH, file.filename)
    file.save(path)

    video_source["mode"] = "video"
    video_source["path"] = path

    # optional: Cache leeren nach neuem Video (manchmal hilfreich auf Jetson)
    if DEVICE == "cuda":
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    return redirect(url_for("index"))

@app.route("/use_webcam")
def use_webcam():
    video_source["mode"] = "webcam"
    video_source["path"] = None
    return ("", 204)

@app.route("/stream")
def stream():
    return Response(gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# ================= VIDEO LOOP =================
def gen_frames():
    if video_source["mode"] == "video":
        cap = cv2.VideoCapture(video_source["path"])
    else:
        cap = cv2.VideoCapture(0)

    last_box = None

    # (optional) warmup: verhindert manchmal ersten Lag/alloc spike
    if DEVICE == "cuda":
        try:
            dummy = (torch.zeros((1, 3, IMG_SIZE, IMG_SIZE), device="cuda"))
            del dummy
            torch.cuda.synchronize()
        except Exception:
            pass

    while True:
        ret, frame = cap.read()
        if not ret:
            if video_source["mode"] == "video":
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break

        with lock:
            conf = confidence

        # 🔥 FIX: NICHT model.predict() (re-init/fuse), sondern model(...)
        with torch.inference_mode():
            results = model(
                frame,
                imgsz=IMG_SIZE,
                conf=conf,
                iou=0.5,
                device=DEVICE,
                verbose=False
            )

        best_conf = 0.0
        best_box = None

        # Ultralytics Results parsing
        if results and len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
            for b in results[0].boxes:
                if int(b.cls[0]) != BALL_CLASS_ID:
                    continue
                c = float(b.conf[0])
                if c > best_conf:
                    best_conf = c
                    best_box = tuple(map(int, b.xyxy[0]))

        # Roboflow-Style HOLD
        if best_box:
            last_box = best_box
        elif last_box:
            best_box = last_box

        if best_box:
            x1, y1, x2, y2 = best_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"BALL {int(conf * 100)}%",
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            buffer.tobytes() +
            b"\r\n"
        )

    cap.release()

# ================= MAIN =================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, threaded=True)
