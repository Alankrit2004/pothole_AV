import os
import re
import sys
import uuid
import json
import time
import logging
import subprocess
import threading
from datetime import timedelta
import cv2
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from flask_jwt_extended import (JWTManager, create_access_token,jwt_required, get_jwt_identity, decode_token)
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from pathlib import Path
from FSM import frame_manager

# ---------------------------------------------------------------------------
# Logging (auth success/failure + general audit trail)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("pothole_api")


# ---------------------------------------------------------------------------
# Standard response envelope: every JSON response (except streaming
# endpoints) uses {"status": "success"|"error", "message": ..., "data": ...}.
# HTTP status codes are kept as-is alongside this -- "status" here refers to
# the envelope's success/error field, not the HTTP status code.
#
# Use ONLY these helpers for JSON responses -- never call jsonify({...})
# directly in a route. That's what let the duplicate-key ("status" set
# twice in one dict), set-literal ({details} instead of {"data": details}),
# and typo ("errmessageor") bugs slip in: hand-built dicts have no
# structural guardrail against any of that.
# ---------------------------------------------------------------------------
def ok(data=None, message="", code=200):
    return jsonify({"status": "success", "message": message, "data": data}), code


def err(message, code=400, data=None):
    return jsonify({"status": "error", "message": message, "data": data}), code

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")
ALLOWED_EXT = {"png", "jpg", "jpeg", "bmp", "webp"}
ALLOWED_VIDEO_EXT = {"mp4", "avi", "mov", "mkv"}

# Fail fast: refuse to start with an unset/default JWT secret. This must
# happen before anything else touches Flask-JWT-Extended.
JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY")
if not JWT_SECRET_KEY:
    logger.error("JWT_SECRET_KEY is not set. Refusing to start. "
                  "Set a strong random value, e.g.: python -c \"import secrets; print(secrets.token_hex(32))\"")
    sys.exit(1)

app = Flask(__name__)
app.config["JWT_SECRET_KEY"] = JWT_SECRET_KEY
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=6)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB max upload (videos are bigger)

# ---------------------------------------------------------------------------
# CORS -- frontend is a separate app. Allowlist origins from env, never "*"
# (credentialed requests with Authorization headers can't use "*" anyway).
# ---------------------------------------------------------------------------
CORS_ORIGINS = [o.strip() for o in os.environ.get("CORS_ORIGINS", "").split(",") if o.strip()]
if not CORS_ORIGINS:
    logger.warning("CORS_ORIGINS is not set -- no frontend origin is allowlisted. "
                    "Set e.g. CORS_ORIGINS=https://app.example.com")
CORS(app, origins=CORS_ORIGINS, supports_credentials=True)

# ---------------------------------------------------------------------------
# Rate limiting -- protects /auth/login and /auth/register from brute force.
# Uses in-memory storage by default (fine for single-replica); point
# RATELIMIT_STORAGE_URI at redis:// for multi-replica deployments.
# ---------------------------------------------------------------------------
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    storage_uri=os.environ.get("RATELIMIT_STORAGE_URI", "memory://"),
    default_limits=[],
)

VIDEO_JOBS = {}
VIDEO_JOBS_LOCK = threading.Lock()

VIDEO_JOB_CANCEL_EVENTS = {}

# Maps output filename -> username that owns it (images, annotated videos,
# and alert frames). Used to authorize /outputs/<filename> per-user.
OUTPUT_OWNERS = {}
OUTPUT_OWNERS_LOCK = threading.Lock()

# Job time-to-live: jobs (and their cancel events) older than this are swept
# by the background retention thread so VIDEO_JOBS doesn't grow unbounded.
JOB_TTL_SECONDS = int(os.environ.get("JOB_TTL_SECONDS", str(24 * 3600)))
# Output files older than this are deleted by the retention thread.
OUTPUT_TTL_SECONDS = int(os.environ.get("OUTPUT_TTL_SECONDS", str(7 * 24 * 3600)))
RETENTION_SWEEP_INTERVAL_SECONDS = int(os.environ.get("RETENTION_SWEEP_INTERVAL_SECONDS", "3600"))

VIDEO_PROCESSING_LOCK = threading.Lock()

jwt = JWTManager(app)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ---------------------------------------------------------------------------
# "Users DB" -- replace with a real database (SQLite/Postgres) in production
# ---------------------------------------------------------------------------
# Bootstrap admin password comes from env so it can be rotated without a
# code change. Falls back to a random one-time password printed to the log
# if the operator hasn't set one, rather than shipping a known default.
_admin_password = os.environ.get("ADMIN_PASSWORD")
if not _admin_password:
    import secrets as _secrets
    _admin_password = _secrets.token_urlsafe(12)
    logger.warning("ADMIN_PASSWORD not set -- generated a one-time admin password: %s "
                    "(set ADMIN_PASSWORD env var to control this explicitly)", _admin_password)

USERS = {
    "admin": generate_password_hash(_admin_password),
}

PASSWORD_MIN_LENGTH = int(os.environ.get("PASSWORD_MIN_LENGTH", "10"))


def password_is_strong(password):
    """Minimal strength rules: since registration stays open, this is our
    main defense against throwaway/weak accounts. Returns (ok, reason)."""
    if len(password) < PASSWORD_MIN_LENGTH:
        return False, f"password must be at least {PASSWORD_MIN_LENGTH} characters"
    if not re.search(r"[A-Za-z]", password):
        return False, "password must contain at least one letter"
    if not re.search(r"[0-9]", password):
        return False, "password must contain at least one digit"
    return True, None


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
@limiter.limit("10 per minute")
def login():
    data = request.get_json(silent=True) or {}
    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        return err("username and password are required", 400)

    with USERS_LOCK:
        password_hash = USERS.get(username)
    if not password_hash or not check_password_hash(password_hash, password):
        logger.info("auth failure: login for user=%r from %s", username, get_remote_address())
        return err("invalid username or password", 401)

    logger.info("auth success: login for user=%r from %s", username, get_remote_address())
    return ok(
        data={"access_token": create_access_token(identity=username), "token_type": "Bearer"},
        message="logged in successfully",
    )


@app.route("/auth/register", methods=["POST"])
@limiter.limit("5 per minute")
def register():
    data = request.get_json(silent=True) or {}
    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        return err("username and password are required", 400)

    strong, reason = password_is_strong(password)
    if not strong:
        return err(reason, 400)

    with USERS_LOCK:
        if username in USERS:
            return err("user already exists", 409)

        USERS[username] = generate_password_hash(password)
        try:
            save_users()
        except OSError:
            USERS.pop(username, None)
            logger.exception("could not persist new user %r", username)
            return err("could not save user, please try again later", 500)

    logger.info("auth success: registered user=%r from %s", username, get_remote_address())
    return ok(data={"username": username}, message="user registered successfully", code=201)


@app.route("/auth/change-password", methods=["POST"])
@jwt_required()
@limiter.limit("5 per minute")
def change_password():
    """Lets any authenticated user (including admin) rotate their own
    password. This is what makes the admin password rotatable post-deploy
    without editing env vars and restarting."""
    current_user = get_jwt_identity()
    data = request.get_json(silent=True) or {}
    old_password = data.get("old_password")
    new_password = data.get("new_password")

    if not old_password or not new_password:
        return err("old_password and new_password are required", 400)

    strong, reason = password_is_strong(new_password)
    if not strong:
        return err(reason, 400)

    with USERS_LOCK:
        password_hash = USERS.get(current_user)
        if not password_hash or not check_password_hash(password_hash, old_password):
            logger.info("auth failure: change-password for user=%r from %s", current_user, get_remote_address())
            return err("old_password is incorrect", 401)

        USERS[current_user] = generate_password_hash(new_password)
        try:
            save_users()
        except OSError:
            logger.exception("could not persist password change for %r", current_user)
            return err("could not save new password, please try again later", 500)

    logger.info("auth success: password changed for user=%r from %s", current_user, get_remote_address())
    return ok(message="password updated successfully")

# ---------------------------------------------------------------------------
# Protected endpoints
# ---------------------------------------------------------------------------

def _remux_to_h264(output_path):
    """Fallback path when OpenCV's build has no H.264 encoder: re-encode the
    mp4v file to H.264 with ffmpeg so it's playable in-browser."""
    tmp_path = output_path + ".h264.mp4"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", output_path, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", tmp_path],
            check=True, capture_output=True, timeout=600,
        )
        os.replace(tmp_path, output_path)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        logger.exception("ffmpeg H.264 remux failed for %s -- output remains mp4v (not browser-playable)", output_path)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def process_video_job(job_id, filepath, output_path, conf_threshold, frame_skip, cancel_event, owner_user):
    with OUTPUT_OWNERS_LOCK:
        OUTPUT_OWNERS[os.path.basename(output_path)] = owner_user

    try:
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            raise RuntimeError("could not open video file")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Browsers can't play mp4v (MPEG-4 Part 2). Try H.264 (avc1) first;
        # if the installed OpenCV build lacks an H.264 encoder, fall back to
        # mp4v and remux to H.264 with ffmpeg after writing (openh264 in the
        # Docker image provides the actual encode either way).
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        needs_ffmpeg_remux = not writer.isOpened()
        if needs_ffmpeg_remux:
            writer.release()
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
                        with OUTPUT_OWNERS_LOCK:
                            OUTPUT_OWNERS[alert_frame_name] = owner_user

                        with VIDEO_JOBS_LOCK:
                            VIDEO_JOBS[job_id]["alerts"].append(alert)
                            VIDEO_JOBS[job_id]["total_unique_potholes"] = next_pothole_number - 1

                writer.write(last_annotated_frame if last_annotated_frame is not None else frame)
                if frame_manager.has_subscribers(job_id):
                    frame_to_send = last_annotated_frame if last_annotated_frame is not None else frame
                    encode_ok, buffer = cv2.imencode(".jpg", frame_to_send, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if encode_ok:
                        frame_manager.broadcast(job_id, buffer.tobytes())

                frame_idx += 1

                if total_frames > 0:
                    with VIDEO_JOBS_LOCK:
                        VIDEO_JOBS[job_id]["progress"] = round(frame_idx / total_frames * 100, 1)

        cap.release()
        writer.release()

        if needs_ffmpeg_remux:
            _remux_to_h264(output_path)

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
        return err("no video file provided (use form field 'video')", 400)

    file = request.files["video"]
    if file.filename == "":
        return err("empty filename", 400)
    if not allowed_video_file(file.filename):
        return err(f"unsupported file type, allowed: {ALLOWED_VIDEO_EXT}", 400)

    conf_threshold = float(request.form.get("conf", 0.25))

    frame_skip = max(1, int(request.form.get("frame_skip", 1)))

    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file.save(filepath)

    output_name = f"annotated_{os.path.splitext(unique_name)[0]}.mp4"
    output_path = os.path.join(OUTPUT_FOLDER, output_name)

    # Each upload gets its own job id so concurrent submissions (by the same
    # or different users) never clobber each other's job state.
    # job_id = uuid.uuid4().hex
    job_id = "123_FIX_JOB_ID"


    with VIDEO_JOBS_LOCK:
        VIDEO_JOBS[job_id] = {
            "status": "processing",
            "progress": 0.0,
            "user": current_user,
            "alerts": [],
            "total_unique_potholes": 0,
            "created_at": time.time(),
        }
    cancel_event = threading.Event()
    VIDEO_JOB_CANCEL_EVENTS[job_id] = cancel_event

    thread = threading.Thread(
        target=process_video_job,
        args=(job_id, filepath, output_path, conf_threshold, frame_skip, cancel_event, current_user),
        daemon=True,
    )
    thread.start()

    return ok(data={
        "job_id": job_id,
        "status_url": f"/jobs/{job_id}",
        "stop_url": f"/videos/{job_id}/stop"
    }, message="video submitted for processing", code=202)


def _require_job_owner(job, current_user):
    """Returns an error Response if current_user doesn't own the job, else None."""
    if job.get("user") != current_user:
        return err("job not found", 404)  # 404, not 403 -- don't reveal existence
    return None


@app.route("/videos/<job_id>/stop", methods=["POST"])
@jwt_required()
def stop_video_job(job_id):
    current_user = get_jwt_identity()
    with VIDEO_JOBS_LOCK:
        job = VIDEO_JOBS.get(job_id)

    if not job:
        return err("job not found", 404)
    owner_error = _require_job_owner(job, current_user)
    if owner_error:
        return owner_error

    if job["status"] != "processing":
        return err(f"job is not running (status: {job['status']})", 409)

    cancel_event = VIDEO_JOB_CANCEL_EVENTS.get(job_id)
    if not cancel_event:
        return err("job is not currently running", 409)

    cancel_event.set()
    return ok(data={"job_id": job_id}, message="stop requested", code=202)


@app.route("/jobs/<job_id>/events/stream", methods=["GET"])
@jwt_required()
def video_status_stream(job_id):
    current_user = get_jwt_identity()
    with VIDEO_JOBS_LOCK:
        job = VIDEO_JOBS.get(job_id)
    if not job:
        return err("job not found", 404)
    owner_error = _require_job_owner(job, current_user)
    if owner_error:
        return owner_error

    def event_stream():
        last_sent = None
        last_alert_count = 0
        while True:
            with VIDEO_JOBS_LOCK:
                job = VIDEO_JOBS.get(job_id)
                job_snapshot = dict(job) if job else None

            if not job_snapshot:
                yield f"event: error\ndata: {json.dumps({'status': 'error', 'message': 'job not found'})}\n\n"
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

@app.route("/jobs/<job_id>", methods=["GET"])  #Job status
@jwt_required()
def job_status(job_id):
    current_user = get_jwt_identity()

    with VIDEO_JOBS_LOCK:
        job = VIDEO_JOBS.get(job_id)

    if not job:
        return err("job not found", 404)
    owner_error = _require_job_owner(job, current_user)
    if owner_error:
        return owner_error

    return ok(
        data={"status": job.get("status", "unknown")},  # processing/completed/failed/cancelled
        message="job status retrieved",
    )

@app.route("/videos/<job_id>/stream", methods=["GET"])
@jwt_required()
def video_live(job_id):
    current_user = get_jwt_identity()
    with VIDEO_JOBS_LOCK:
        job = VIDEO_JOBS.get(job_id)
    if not job:
        return err("job not found", 404)
    owner_error = _require_job_owner(job, current_user)
    if owner_error:
        return owner_error
    if job["status"] != "processing":
        return err(f"job is not live (status: {job['status']})", 409)

    return Response(
        frame_manager.generate_mjpeg(job_id),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/videos/outputs", methods=["GET"])
@jwt_required()
def video_list():
    current_user = get_jwt_identity()
    output_dir = Path(OUTPUT_FOLDER)

    if not output_dir.exists():
        return ok(data={"count": 0, "videos": []}, message="output folder does not exist")

    videos = []
    for path in output_dir.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower().lstrip(".") not in ALLOWED_VIDEO_EXT:
            continue

        with OUTPUT_OWNERS_LOCK:
            owner = OUTPUT_OWNERS.get(path.name)
        if owner != current_user:
            continue

        stat = path.stat()
        videos.append({
            "filename": path.name,
            "url": f"/outputs/{path.name}",
            "size_bytes": stat.st_size,
            "modified_at": stat.st_mtime,
        })

    videos.sort(key=lambda v: v["modified_at"], reverse=True)

    return ok(data={"count": len(videos), "videos": videos}, message="video outputs retrieved")


@app.route("/jobs/<job_id>/details", methods=["GET"])
@jwt_required()
def job_details(job_id):
    current_user = get_jwt_identity()

    with VIDEO_JOBS_LOCK:
        job = VIDEO_JOBS.get(job_id)

    if not job:
        return err("job not found", 404)
    owner_error = _require_job_owner(job, current_user)
    if owner_error:
        return owner_error

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

    return ok(data=details, message="job details retrieved")

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
        return err("job not found", 404)
    owner_error = _require_job_owner(job, current_user)
    if owner_error:
        return owner_error

    status = job.get("status", "unknown")

    if status != "completed":
        return ok(data={"status": status}, message="results not ready yet", code=202)

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

    return ok(data=result, message="job results retrieved")

# ---------------------------------------------------------------------------
# Health check (public) -- also verifies the model actually loaded, not
# just that the process is up.
# ---------------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    model_ok = model is not None
    status_code = 200 if model_ok else 503
    if model_ok:
        return ok(data={"model_loaded": True}, message="ok")
    return err("degraded: model not loaded", status_code, data={"model_loaded": False})


# ---------------------------------------------------------------------------
# JWT error handlers -- Flask-JWT-Extended's defaults (e.g. {"msg": "Not
# enough segments"}) don't match our {status, message, data} envelope.
# Route them through err() too so every response the client ever sees has
# the same shape, including auth failures before a route even runs.
# ---------------------------------------------------------------------------
@jwt.unauthorized_loader
def _missing_token(reason):
    return err("authorization token is required", 401)


@jwt.invalid_token_loader
def _invalid_token(reason):
    return err("invalid authorization token", 422)


@jwt.expired_token_loader
def _expired_token(jwt_header, jwt_payload):
    return err("authorization token has expired", 401)


@jwt.revoked_token_loader
def _revoked_token(jwt_header, jwt_payload):
    return err("authorization token has been revoked", 401)


# ---------------------------------------------------------------------------
# Retention sweep -- runs in the background so VIDEO_JOBS/outputs/uploads
# don't grow unbounded on a long-lived process.
# ---------------------------------------------------------------------------
def _retention_sweep_loop():
    while True:
        time.sleep(RETENTION_SWEEP_INTERVAL_SECONDS)
        try:
            _sweep_stale_jobs()
            _sweep_stale_outputs()
        except Exception:
            logger.exception("retention sweep failed")


def _sweep_stale_jobs():
    now = time.time()
    with VIDEO_JOBS_LOCK:
        stale_ids = [
            job_id for job_id, job in VIDEO_JOBS.items()
            if job.get("status") in ("completed", "failed", "cancelled")
            and (now - job.get("created_at", now)) > JOB_TTL_SECONDS
        ]
        for job_id in stale_ids:
            VIDEO_JOBS.pop(job_id, None)
            VIDEO_JOB_CANCEL_EVENTS.pop(job_id, None)
    if stale_ids:
        logger.info("retention: swept %d stale video job(s)", len(stale_ids))


def _sweep_stale_outputs():
    now = time.time()
    removed = 0
    for folder in (OUTPUT_FOLDER, UPLOAD_FOLDER):
        for path in Path(folder).iterdir():
            if not path.is_file():
                continue
            try:
                age = now - path.stat().st_mtime
            except OSError:
                continue
            if age > OUTPUT_TTL_SECONDS:
                try:
                    path.unlink()
                    with OUTPUT_OWNERS_LOCK:
                        OUTPUT_OWNERS.pop(path.name, None)
                    removed += 1
                except OSError:
                    logger.exception("retention: failed to remove %s", path)
    if removed:
        logger.info("retention: removed %d stale output/upload file(s)", removed)


_retention_thread = threading.Thread(target=_retention_sweep_loop, daemon=True)
_retention_thread.start()


if __name__ == "__main__":
    # debug=True enables the Werkzeug debugger, which is a remote-code-
    # execution risk if this file is ever run directly in production.
    # Production runs via gunicorn (see Dockerfile/README); this path is
    # for local development only.
    app.run(host="0.0.0.0", port=5000, debug=False)