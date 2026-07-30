import os
import uuid
import json
import time
import threading
from datetime import timedelta
import cv2
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from flask_jwt_extended import (JWTManager, create_access_token,jwt_required, get_jwt_identity, decode_token)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from pathlib import Path
from FSM import frame_manager

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")   
ALLOWED_EXT = {"png", "jpg", "jpeg", "bmp", "webp"}
ALLOWED_VIDEO_EXT = {"mp4", "avi", "mov", "mkv"}

app = Flask(__name__)
app.config["JWT_SECRET_KEY"] = os.environ.get("JWT_SECRET_KEY", "change-this-secret-key")
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=6)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB max upload (videos are bigger)


VIDEO_JOBS = {}
VIDEO_JOBS_LOCK = threading.Lock()

VIDEO_JOB_CANCEL_EVENTS = {}

FIXED_VIDEO_JOB_ID = "123_video_job"

VIDEO_PROCESSING_LOCK = threading.Lock()

jwt = JWTManager(app)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ---------------------------------------------------------------------------
# "Users DB" -- replace with a real database (SQLite/Postgres) in production
# ---------------------------------------------------------------------------
USERS = {
    "admin": generate_password_hash("admin123"),
}

# ---------------------------------------------------------------------------
# Load the YOLO model ONCE at startup (not per-request -- important!)
# ---------------------------------------------------------------------------
print("Loading pothole detection model...")
model = YOLO(MODEL_PATH)
print("Model loaded.")
USERS_FILE = os.path.join(BASE_DIR, "app2_users.json")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def allowed_video_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_VIDEO_EXT

USERS_LOCK = threading.Lock()
def load_users():
    """Load saved password hashes, retaining the built-in admin account."""
    if not os.path.exists(USERS_FILE):
        return

    try:
        with open(USERS_FILE, "r", encoding="utf-8") as users_file:
            saved_users = json.load(users_file)
        if isinstance(saved_users, dict) and all(
            isinstance(username, str) and isinstance(password_hash, str)
            for username, password_hash in saved_users.items()
        ):
            USERS.update(saved_users)
    except (OSError, json.JSONDecodeError):
        # The service still starts with its default account if the file is bad.
        pass

def save_users():
    """Atomically save password hashes so accounts survive server restarts."""
    temporary_file = f"{USERS_FILE}.tmp"
    with open(temporary_file, "w", encoding="utf-8") as users_file:
        json.dump(USERS, users_file, indent=2)
    os.replace(temporary_file, USERS_FILE)

load_users()
# ---------------------------------------------------------------------------
# Auth endpoints
# ---------------------------------------------------------------------------
@app.route("/auth/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        return jsonify({"error": "username and password are required"}), 400

    with USERS_LOCK:
        password_hash = USERS.get(username)
    if not password_hash or not check_password_hash(password_hash, password):
        return jsonify({"error": "invalid username or password"}), 401

    return jsonify({
        "access_token": create_access_token(identity=username),
        "token_type": "Bearer",
    }), 200


@app.route("/auth/register", methods=["POST"])
def register():
    data = request.get_json(silent=True) or {}
    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        return jsonify({"error": "username and password are required"}), 400

    with USERS_LOCK:
        if username in USERS:
            return jsonify({"error": "user already exists"}), 409

        USERS[username] = generate_password_hash(password)
        try:
            save_users()
        except OSError as error:
            USERS.pop(username, None)
            return jsonify({"error": f"could not save user: {error}"}), 500

    return jsonify({"message": "user registered successfully"}), 201

# ---------------------------------------------------------------------------
# Protected endpoints
# ---------------------------------------------------------------------------
@app.route("/predict", methods=["POST"])
@jwt_required()
def predict():
    current_user = get_jwt_identity()

    if "image" not in request.files:
        return jsonify({"error": "no image file provided (use form field 'image')"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "empty filename"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": f"unsupported file type, allowed: {ALLOWED_EXT}"}), 400

    # Save the uploaded image with a unique name
    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file.save(filepath)

    conf_threshold = float(request.form.get("conf", 0.25))

    # Run inference
    results = model.predict(source=filepath, conf=conf_threshold, verbose=False)
    result = results[0]

    detections = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        detections.append({
            "class_id": cls_id,
            "class_name": model.names[cls_id],
            "confidence": round(float(box.conf[0]), 4),
            "bbox_xyxy": [round(v, 2) for v in box.xyxy[0].tolist()],
        })

    annotated = result.plot()  
    output_name = f"annotated_{unique_name}"
    output_path = os.path.join(OUTPUT_FOLDER, output_name)
    cv2.imwrite(output_path, annotated)

    return jsonify({
        "user": current_user,
        "num_potholes_detected": len(detections),
        "detections": detections,
        "annotated_image_url": f"/outputs/{output_name}"
    }), 200


def process_video_job(job_id, filepath, output_path, conf_threshold, frame_skip, cancel_event):
    try:
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            raise RuntimeError("could not open video file")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_idx = 0
        total_detections = 0
        max_potholes_in_frame = 0
        last_annotated_frame = None
        was_cancelled = False

        seen_track_ids = {}
        next_pothole_number = 1

        with VIDEO_PROCESSING_LOCK:
            while True:
                if cancel_event.is_set():
                    was_cancelled = True
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_skip == 0:
                    is_first_tracked_frame = (frame_idx == 0)
                    results = model.track(
                        source=frame,
                        conf=conf_threshold,
                        persist=True,
                        tracker="bytetrack.yaml",
                        verbose=False,
                    )
                    result = results[0]
                    num_in_frame = len(result.boxes)
                    total_detections += num_in_frame
                    max_potholes_in_frame = max(max_potholes_in_frame, num_in_frame)
                    annotated = frame.copy()
                    new_alerts_this_frame = []

                    if result.boxes.id is not None:
                        track_ids = result.boxes.id.int().tolist()
                        for box, track_id in zip(result.boxes, track_ids):
                            is_new = track_id not in seen_track_ids
                            if is_new:
                                pothole_number = next_pothole_number
                                seen_track_ids[track_id] = pothole_number
                                next_pothole_number += 1
                            else:
                                pothole_number = seen_track_ids[track_id]

                            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                            conf_val = float(box.conf[0])
                            label = f"Pothole #{pothole_number} {conf_val:.2f}"
                            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(annotated, label, (x1, max(y1 - 8, 0)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                            if is_new:
                                cls_id = int(box.cls[0])
                                new_alerts_this_frame.append({
                                    "pothole_id": pothole_number,
                                    "frame_index": frame_idx,
                                    "timestamp_sec": round(frame_idx / fps, 2),
                                    "confidence": round(conf_val, 4),
                                    "class_name": model.names[cls_id],
                                    "bbox_xyxy": [round(v, 2) for v in box.xyxy[0].tolist()],
                                })

                    last_annotated_frame = annotated
                    for alert in new_alerts_this_frame:
                        alert_frame_name = f"alert_{job_id}_pothole{alert['pothole_id']}.jpg"
                        alert_frame_path = os.path.join(OUTPUT_FOLDER, alert_frame_name)
                        cv2.imwrite(alert_frame_path, last_annotated_frame)
                        alert["alert_frame_url"] = f"/outputs/{alert_frame_name}"

                        with VIDEO_JOBS_LOCK:
                            VIDEO_JOBS[job_id]["alerts"].append(alert)
                            VIDEO_JOBS[job_id]["total_unique_potholes"] = next_pothole_number - 1

                writer.write(last_annotated_frame if last_annotated_frame is not None else frame)
                if frame_manager.has_subscribers(job_id):
                    frame_to_send = last_annotated_frame if last_annotated_frame is not None else frame
                    ok, buffer = cv2.imencode(".jpg", frame_to_send, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ok:
                        frame_manager.broadcast(job_id, buffer.tobytes())

                frame_idx += 1

                if total_frames > 0:
                    with VIDEO_JOBS_LOCK:
                        VIDEO_JOBS[job_id]["progress"] = round(frame_idx / total_frames * 100, 1)

        cap.release()
        writer.release()

        if was_cancelled:
            with VIDEO_JOBS_LOCK:
                VIDEO_JOBS[job_id].update({
                    "status": "cancelled",
                    "total_frames_processed": frame_idx,
                    "total_detections_across_video": total_detections,
                    "max_potholes_in_a_single_frame": max_potholes_in_frame,
                    "annotated_video_url": f"/outputs/{os.path.basename(output_path)}",
                })
        else:
            with VIDEO_JOBS_LOCK:
                VIDEO_JOBS[job_id].update({
                    "status": "completed",
                    "progress": 100.0,
                    "total_frames_processed": frame_idx,
                    "total_detections_across_video": total_detections,
                    "max_potholes_in_a_single_frame": max_potholes_in_frame,
                    "annotated_video_url": f"/outputs/{os.path.basename(output_path)}",
                })
    except Exception as e:
        with VIDEO_JOBS_LOCK:
            VIDEO_JOBS[job_id].update({"status": "failed", "error": str(e)})
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)
        VIDEO_JOB_CANCEL_EVENTS.pop(job_id, None)


@app.route("/videos", methods=["POST"])
@jwt_required()
def predict_video():
    current_user = get_jwt_identity()

    if "video" not in request.files:
        return jsonify({"error": "no video file provided (use form field 'video')"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "empty filename"}), 400
    if not allowed_video_file(file.filename):
        return jsonify({"error": f"unsupported file type, allowed: {ALLOWED_VIDEO_EXT}"}), 400

    conf_threshold = float(request.form.get("conf", 0.25))

    frame_skip = max(1, int(request.form.get("frame_skip", 1)))

    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file.save(filepath)

    output_name = f"annotated_{os.path.splitext(unique_name)[0]}.mp4"
    output_path = os.path.join(OUTPUT_FOLDER, output_name)

    # job_id = uuid.uuid4().hex
    job_id = FIXED_VIDEO_JOB_ID

    with VIDEO_JOBS_LOCK:
        VIDEO_JOBS[job_id] = {
            "status": "processing",
            "progress": 0.0,
            "user": current_user,
            "alerts": [],
            "total_unique_potholes": 0,
        }
    cancel_event = threading.Event()
    VIDEO_JOB_CANCEL_EVENTS[job_id] = cancel_event

    thread = threading.Thread(
        target=process_video_job,
        args=(job_id, filepath, output_path, conf_threshold, frame_skip, cancel_event),
        daemon=True,
    )
    thread.start()

    return jsonify({
        "message": "video submitted for processing",
        "job_id": job_id,
        "status_url": f"/video_status/{job_id}",
        "stop_url": f"/predict_video/{job_id}/stop"
    }), 202


@app.route("/videos/<job_id>/stop", methods=["POST"])
@jwt_required()
def stop_video_job(job_id):
    with VIDEO_JOBS_LOCK:
        job = VIDEO_JOBS.get(job_id)

    if not job:
        return jsonify({"error": "job not found"}), 404

    if job["status"] != "processing":
        return jsonify({"error": f"job is not running (status: {job['status']})"}), 409

    cancel_event = VIDEO_JOB_CANCEL_EVENTS.get(job_id)
    if not cancel_event:
        return jsonify({"error": "job is not currently running"}), 409

    cancel_event.set()
    return jsonify({"message": "stop requested", "job_id": job_id}), 202



@app.route("/jobs/<job_id>/events/stream", methods=["GET"])
@jwt_required()
def video_status_stream(job_id):
    def event_stream():
        last_sent = None
        last_alert_count = 0
        while True:
            with VIDEO_JOBS_LOCK:
                job = VIDEO_JOBS.get(job_id)
                job_snapshot = dict(job) if job else None

            if not job_snapshot:
                yield f"event: error\ndata: {json.dumps({'error': 'job not found'})}\n\n"
                return
            alerts = job_snapshot.get("alerts", [])
            if len(alerts) > last_alert_count:
                for alert in alerts[last_alert_count:]:
                    yield f"event: alert\ndata: {json.dumps(alert)}\n\n"
                last_alert_count = len(alerts)

            snapshot = json.dumps(job_snapshot, sort_keys=True)
            if snapshot != last_sent:
                yield f"event: progress\ndata: {json.dumps({'job_id': job_id, **job_snapshot})}\n\n"
                last_sent = snapshot

            if job_snapshot["status"] in ("completed", "failed", "cancelled"):
                return

            time.sleep(1)

    return Response(
        stream_with_context(event_stream()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable proxy buffering (e.g. nginx)
        },
    )

@app.route("/jobs/<job_id>", methods=["GET"])
@jwt_required()
def job_status(job_id):

    current_user = get_jwt_identity()

    with VIDEO_JOBS_LOCK:
        job = VIDEO_JOBS.get(job_id)

    if not job:
        return jsonify({
            "user": current_user,
            "job_id": job_id,
            "error": "job not found"
        }), 404

    return jsonify({
        "status": job.get("status", "unknown"),        # processing/completed/failed/cancelled
    }), 200

@app.route("/videos/<job_id>/stream", methods=["GET"])
@jwt_required()
def video_live(job_id):
    with VIDEO_JOBS_LOCK:
        job = VIDEO_JOBS.get(job_id)
    if not job:
        return jsonify({"error": "job not found"}), 404
    if job["status"] != "processing":
        return jsonify({"error": f"job is not live (status: {job['status']})"}), 409

    return Response(
        frame_manager.generate_mjpeg(job_id),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/outputs/<path:filename>", methods=["GET"])
@jwt_required()
def get_output_image(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)


@app.route("/videos/outputs", methods=["GET"])
@jwt_required()
def video_list():
    current_user = get_jwt_identity()
    output_dir = Path(OUTPUT_FOLDER)  

    if not output_dir.exists():
        return jsonify({
            "user": current_user,
            "count": 0,
            "videos": [],
            "error": "output folder does not exist"
        }), 200

    videos = []
    for path in output_dir.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower().lstrip(".") not in ALLOWED_VIDEO_EXT:
            continue

        stat = path.stat()
        videos.append({
            "filename": path.name,
            "url": f"/outputs/{path.name}",
            "size_bytes": stat.st_size,
            "modified_at": stat.st_mtime,
        })

    videos.sort(key=lambda v: v["modified_at"], reverse=True)

    return jsonify({
        "user": current_user,
        "count": len(videos),
        "videos": videos,
    }), 200
@app.route("/jobs/<job_id>/details", methods=["GET"])
@jwt_required()
def job_details(job_id):
    current_user = get_jwt_identity()

    with VIDEO_JOBS_LOCK:
        job = VIDEO_JOBS.get(job_id)

    if not job:
        return jsonify({
            "user": current_user,
            "job_id": job_id,
            "error": "job not found"
        }), 404

    # Build a safe copy of job data
    status = job.get("status", "unknown")
    progress = job.get("progress", 0.0)
    total_unique_potholes = job.get("total_unique_potholes", 0)
    alerts = job.get("alerts", [])

    details = {
        "user": current_user,
        "job_id": job_id,
        "status": status,
        "progress": progress,
        "total_unique_potholes": total_unique_potholes,
        "alerts_count": len(alerts),
        "output": {
            "annotated_video_url": job.get("annotated_video_url"),
            "total_frames_processed": job.get("total_frames_processed"),
            "total_detections_across_video": job.get("total_detections_across_video"),
            "max_potholes_in_a_single_frame": job.get("max_potholes_in_a_single_frame"),
        },
        "error": job.get("error"),  # only non-null if failed
    }

    if alerts:
        last_alerts = alerts[-3:]  # last 3 alerts
        details["alerts_summary"] = {
            "total_alerts": len(alerts),
            "last_alerts": [
                {
                    "pothole_id": a["pothole_id"],
                    "frame_index": a["frame_index"],
                    "timestamp_sec": a["timestamp_sec"],
                    "confidence": a["confidence"],
                    "class_name": a["class_name"],
                }
                for a in last_alerts
            ]
        }

    return jsonify(details), 200

@app.route("/jobs/<job_id>/results", methods=["GET"])
@jwt_required()
def job_result(job_id):
    """
    Return final detection results for a completed job.
    If the job is not completed, returns a 'not ready' response.
    """
    current_user = get_jwt_identity()

    with VIDEO_JOBS_LOCK:
        job = VIDEO_JOBS.get(job_id)

    if not job:
        return jsonify({
            "user": current_user,
            "job_id": job_id,
            "error": "job not found"
        }), 404

    status = job.get("status", "unknown")

    if status != "completed" :
        return jsonify({
            "user": current_user,
            "job_id": job_id,
            "status": status,
            "error": "results not ready yet"
        }), 202  # or 409 if you prefer

    alerts = job.get("alerts", [])

    pothole_map = {}
    for a in alerts:
        pid = a["pothole_id"]
        if pid not in pothole_map:
            pothole_map[pid] = {
                "pothole_id": pid,
                "first_seen": {
                    "frame_index": a["frame_index"],
                    "timestamp_sec": a["timestamp_sec"],
                    "confidence": a["confidence"],
                    "bbox_xyxy": a["bbox_xyxy"],
                    "alert_frame_url": a.get("alert_frame_url"),
                },
                "detection_count": 1,
                "last_confidence": a["confidence"],
            }
        else:
            pothole_map[pid]["detection_count"] += 1
            pothole_map[pid]["last_confidence"] = a["confidence"]

    potholes = list(pothole_map.values())
    potholes.sort(key=lambda p: p["pothole_id"])

    result = {
        "user": current_user,
        "job_id": job_id,
        "status": status,
        "summary": {
            "total_unique_potholes": job.get("total_unique_potholes", 0),
            "total_detections_across_video": job.get("total_detections_across_video", 0),
            "max_potholes_in_a_single_frame": job.get("max_potholes_in_a_single_frame", 0),
            "total_frames_processed": job.get("total_frames_processed", 0),
        },
        "potholes": potholes,
        "annotated_video_url": job.get("annotated_video_url"),
    }

    return jsonify(result), 200

# ---------------------------------------------------------------------------
# Health check (public)
# ---------------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)