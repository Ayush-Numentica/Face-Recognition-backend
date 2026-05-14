"""
app.py
------
Flask REST API for the Face Recognition application.

Endpoints
---------
GET  /health                  — Health check
POST /recognize               — Recognise a face in an uploaded image
GET  /persons                 — List all known persons
POST /add-person              — Add a new person (name + image files)
DEL  /delete-person/<name>    — Delete a person
GET  /custom-messages         — Get all custom greeting messages
POST /custom-messages         — Overwrite custom greeting messages
GET  /face-log                — Most recent 100 recognition events
POST /rebuild-embeddings      — Force-rebuild all embeddings from dataset
"""

import os
import sys
import json
import logging
import threading
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

# ── Make sure imports from the same directory work ──────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from recognition import FaceRecognitionEngine
from utils import decode_base64_image

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR    = os.path.join(BASE_DIR, "dataset")
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")
MESSAGES_FILE  = os.path.join(BASE_DIR, "custom_messages.json")
LOG_FILE       = os.path.join(BASE_DIR, "face_log.json")

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)   # Allow all origins — fine for local development

# ── Recognition engine (loaded once at startup) ───────────────────────────────
engine = FaceRecognitionEngine(
    dataset_dir    = DATASET_DIR,
    embeddings_dir = EMBEDDINGS_DIR,
)

# ── IP / Wireless camera state ────────────────────────────────────────────────
_ip_cam_lock       = threading.Lock()
_ip_cam_active     = False
_ip_cam_thread     = None
_ip_cam_result     = []    # latest recognized faces from wireless cam
_ip_cam_frame      = None  # latest annotated JPEG bytes for MJPEG preview
_ip_cam_raw_frame  = None  # latest raw BGR frame for recognition thread


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_messages() -> dict:
    if os.path.exists(MESSAGES_FILE):
        with open(MESSAGES_FILE, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def _save_messages(messages: dict) -> None:
    with open(MESSAGES_FILE, "w", encoding="utf-8") as fh:
        json.dump(messages, fh, indent=2, ensure_ascii=False)


def _append_log(name: str, confidence: float) -> None:
    logs = []
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as fh:
                logs = json.load(fh)
        except (json.JSONDecodeError, IOError):
            logs = []

    logs.append({
        "name":       name,
        "confidence": round(confidence, 4),
        "timestamp":  datetime.now().isoformat(),
    })

    # Keep only the latest 1 000 entries to avoid unbounded growth
    logs = logs[-1000:]

    with open(LOG_FILE, "w", encoding="utf-8") as fh:
        json.dump(logs, fh, indent=2)


# ── IP camera background thread ───────────────────────────────────────────────

def _draw_boxes_on_frame(frame: np.ndarray, results: list) -> np.ndarray:
    """Draw bounding boxes + labels on frame, same style as the frontend overlay."""
    COLOR_KNOWN   = (0, 230, 118)   # green  (BGR)
    COLOR_UNKNOWN = (68, 23, 255)   # red    (BGR)
    for r in results:
        if not r.get("box"):
            continue
        x, y, w, h = r["box"]
        isKnown = r.get("name", "Unknown") != "Unknown"
        color   = COLOR_KNOWN if isKnown else COLOR_UNKNOWN
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        label = f"{r['name']}  {round(r.get('confidence', 0) * 100)}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        ly = max(0, y - th - 8)
        cv2.rectangle(frame, (x, ly), (x + tw + 8, ly + th + 8), color, -1)
        cv2.putText(frame, label, (x + 4, ly + th + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def _ip_cam_capture_loop(url: str) -> None:
    """Thread 1 — reads frames, encodes MJPEG preview, stores raw frame."""
    global _ip_cam_active, _ip_cam_frame, _ip_cam_raw_frame
    import time

    logger.info("IP camera capture thread started: %s", url)
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        logger.error("Cannot open IP camera stream: %s", url)
        _ip_cam_active = False
        return

    while _ip_cam_active:
        ret, frame = cap.read()
        if not ret:
            logger.warning("IP camera: lost frame, retrying…")
            time.sleep(0.5)
            cap.open(url)
            continue

        # Grab latest results without holding the lock during encode
        with _ip_cam_lock:
            results = list(_ip_cam_result)
            _ip_cam_raw_frame = frame.copy()

        # Draw boxes and encode — done outside the lock
        annotated = _draw_boxes_on_frame(frame.copy(), results)
        _, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 72])

        with _ip_cam_lock:
            _ip_cam_frame = jpeg.tobytes()

    cap.release()
    logger.info("IP camera capture thread stopped.")


def _ip_cam_recog_loop() -> None:
    """Thread 2 — runs recognition on latest frame, never blocks the stream."""
    global _ip_cam_active, _ip_cam_result
    import time

    logger.info("IP camera recognition thread started.")
    while _ip_cam_active:
        with _ip_cam_lock:
            frame = _ip_cam_raw_frame

        if frame is None:
            time.sleep(0.05)
            continue

        results = engine.recognize_all(frame)
        messages = _load_messages()
        for r in results:
            name = r.get("name", "Unknown")
            if name in messages:
                r["message"] = messages[name]
            elif name != "Unknown":
                r["message"] = f"Welcome {name}!"
            else:
                r["message"] = messages.get("Unknown", "Unknown person detected")
            if name not in ("Unknown",):
                _append_log(name, r.get("confidence", 0.0))

        with _ip_cam_lock:
            _ip_cam_result = results

    logger.info("IP camera recognition thread stopped.")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/ip-camera/start", methods=["POST"])
def ip_camera_start():
    """
    Start reading from a wireless / IP camera stream.

    Body: { "url": "http://192.168.1.x:8080/video" }
          or RTSP: { "url": "rtsp://user:pass@192.168.1.x:554/stream" }
    """
    global _ip_cam_active, _ip_cam_thread, _ip_cam_result

    data = request.get_json(force=True)
    url  = (data.get("url") or "").strip()
    if not url:
        return jsonify({"error": "url is required"}), 400

    # Stop any existing stream first
    if _ip_cam_active:
        _ip_cam_active = False
        if _ip_cam_thread:
            _ip_cam_thread.join(timeout=3)

    _ip_cam_active = True
    _ip_cam_result = []
    # Thread 1: capture frames + encode MJPEG preview
    threading.Thread(target=_ip_cam_capture_loop, args=(url,), daemon=True).start()
    # Thread 2: run recognition independently so it never blocks the stream
    threading.Thread(target=_ip_cam_recog_loop, daemon=True).start()
    return jsonify({"message": f"IP camera started: {url}"})


@app.route("/ip-camera/stop", methods=["POST"])
def ip_camera_stop():
    """Stop the wireless camera stream."""
    global _ip_cam_active, _ip_cam_result
    _ip_cam_active = False
    _ip_cam_result = []
    return jsonify({"message": "IP camera stopped."})


@app.route("/ip-camera/result", methods=["GET"])
def ip_camera_result():
    """Return the latest recognition result from the wireless camera."""
    with _ip_cam_lock:
        return jsonify(_ip_cam_result)


@app.route("/ip-camera/status", methods=["GET"])
def ip_camera_status():
    """Return whether the wireless camera is currently active."""
    return jsonify({"active": _ip_cam_active})


@app.route("/ip-camera/stream")
def ip_camera_stream():
    """
    MJPEG stream of the wireless camera feed with bounding boxes drawn.
    Use as <img src="http://localhost:5000/ip-camera/stream"> in the browser.
    """
    import time

    def generate():
        while _ip_cam_active:
            with _ip_cam_lock:
                frame = _ip_cam_frame
            if frame:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(0.05)   # ~20 fps cap, prevents busy-loop

    headers = {
        "Access-Control-Allow-Origin": "*",
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
    }
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame",
                    headers=headers)


@app.route("/health", methods=["GET"])
def health():
    """Quick liveness check."""
    return jsonify({
        "status":  "ok",
        "message": "Face Recognition API is running",
        "persons": len(engine.known_embeddings),
    })


@app.route("/recognize", methods=["POST"])
def recognize():
    """
    Detect and identify ALL faces in the submitted image.

    Accepts either:
      • JSON body  { "image": "<base64 data-URL or raw base64>" }
      • Multipart  form-field "image" (file upload)

    Returns JSON array, sorted by confidence descending (highest first):
      [{ "name", "confidence", "box", "detected", "message" }, ...]

    Empty array = no faces detected.
    """
    try:
        # --- Decode incoming image -------------------------------------------
        frame = None

        if request.is_json:
            data       = request.get_json(force=True)
            image_data = data.get("image", "")
            frame      = decode_base64_image(image_data)

        elif "image" in request.files:
            file_bytes = np.frombuffer(request.files["image"].read(), np.uint8)
            import cv2
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "No valid image received"}), 400

        # --- Run multi-face recognition -------------------------------------
        results  = engine.recognize_all(frame)
        messages = _load_messages()

        for result in results:
            name = result.get("name", "Unknown")

            # Attach greeting message
            if name in messages:
                result["message"] = messages[name]
            elif name != "Unknown":
                result["message"] = f"Welcome {name}!"
            else:
                result["message"] = messages.get("Unknown", "Unknown person detected")

            # Log known persons only
            if name not in ("Unknown",):
                _append_log(name, result.get("confidence", 0.0))

        return jsonify(results)

    except Exception as exc:
        logger.exception("Error in /recognize")
        return jsonify({"error": str(exc)}), 500


@app.route("/person-image/<name>", methods=["GET"])
def person_image(name: str):
    """
    Return the first image found in dataset/<name>/ as a base64 data URL.
    Returns 404 JSON if no image exists.
    """
    import base64
    import mimetypes
    person_dir = os.path.join(DATASET_DIR, name)
    if not os.path.isdir(person_dir):
        return jsonify({"error": "Not found"}), 404

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted([
        f for f in os.listdir(person_dir)
        if os.path.splitext(f)[1].lower() in valid_exts
    ])
    if not images:
        return jsonify({"error": "No image"}), 404

    path = os.path.join(person_dir, images[0])
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    with open(path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode("utf-8")

    return jsonify({"image": f"data:{mime};base64,{b64}"})


@app.route("/persons", methods=["GET"])
def list_persons():
    """Return metadata for every registered person."""
    try:
        return jsonify({"persons": engine.get_known_persons()})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/add-person", methods=["POST"])
def add_person():
    """
    Register a new person.

    Expects multipart/form-data with:
      • name   (text field)
      • images (one or more image files)
    """
    try:
        name = (request.form.get("name") or "").strip()
        if not name:
            return jsonify({"error": "Name is required"}), 400

        files = request.files.getlist("images")
        if not files:
            return jsonify({"error": "At least one image is required"}), 400

        # Save images to dataset/<name>/
        person_dir  = os.path.join(DATASET_DIR, name)
        os.makedirs(person_dir, exist_ok=True)

        saved = 0
        for idx, file in enumerate(files):
            if not file.filename:
                continue
            ext       = os.path.splitext(file.filename)[1].lower() or ".jpg"
            save_path = os.path.join(person_dir, f"img{idx + 1}{ext}")
            file.save(save_path)
            saved += 1

        if saved == 0:
            return jsonify({"error": "No valid image files received"}), 400

        # Build embeddings for this person
        ok = engine.add_person(name)
        if not ok:
            return jsonify({
                "error": (
                    f"Could not generate embeddings for '{name}'. "
                    "Make sure faces are clearly visible in the provided images."
                )
            }), 400

        # Seed a default custom message if none exists
        messages = _load_messages()
        if name not in messages:
            messages[name] = f"Welcome {name}!"
            _save_messages(messages)

        return jsonify({
            "message":      f"Successfully registered '{name}' with {saved} image(s).",
            "name":         name,
            "images_count": saved,
        })

    except Exception as exc:
        logger.exception("Error in /add-person")
        return jsonify({"error": str(exc)}), 500


@app.route("/delete-person/<name>", methods=["DELETE"])
def delete_person(name: str):
    """Remove a person from the system entirely."""
    try:
        ok = engine.delete_person(name)
        if ok:
            # Also remove their custom message
            messages = _load_messages()
            messages.pop(name, None)
            _save_messages(messages)
            return jsonify({"message": f"Successfully deleted '{name}'."})
        return jsonify({"error": f"Person '{name}' not found."}), 404
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/custom-messages", methods=["GET"])
def get_messages():
    """Return all custom greeting messages."""
    return jsonify(_load_messages())


@app.route("/custom-messages", methods=["POST"])
def update_messages():
    """Replace the entire custom-messages store."""
    try:
        data = request.get_json(force=True)
        if not isinstance(data, dict):
            return jsonify({"error": "Expected a JSON object"}), 400
        _save_messages(data)
        return jsonify({"message": "Custom messages saved."})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/face-log", methods=["GET"])
def face_log():
    """Return the last 100 recognition events, most-recent first."""
    if not os.path.exists(LOG_FILE):
        return jsonify([])
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as fh:
            logs = json.load(fh)
        return jsonify(list(reversed(logs[-100:])))
    except (json.JSONDecodeError, IOError):
        return jsonify([])


@app.route("/rebuild-embeddings", methods=["POST"])
def rebuild_embeddings():
    """Force-rebuild all embeddings from the dataset directory."""
    try:
        count = engine.rebuild_all_embeddings()
        return jsonify({
            "message": f"Rebuilt embeddings for {count} person(s).",
            "count":   count,
        })
    except Exception as exc:
        logger.exception("Error in /rebuild-embeddings")
        return jsonify({"error": str(exc)}), 500


# ── Entry point ───────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     logger.info("Starting Face Recognition API …")
#     logger.info("  Dataset:    %s", DATASET_DIR)
#     logger.info("  Embeddings: %s", EMBEDDINGS_DIR)

#     # Load (or build) embeddings before accepting requests
#     engine.load_embeddings()

#     app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)

if __name__ == "__main__":
    logger.info("Starting Face Recognition API …")
    logger.info("  Dataset:    %s", DATASET_DIR)
    logger.info("  Embeddings: %s", EMBEDDINGS_DIR)

    engine.load_embeddings()

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
