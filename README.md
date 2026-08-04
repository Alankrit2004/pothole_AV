# Pothole Detection API

Flask + YOLO (Ultralytics) service for pothole detection on images and video,
with JWT auth, per-user video jobs, live MJPEG preview, and SSE progress
events.

## Requirements

- `best.pt` (Ultralytics detect checkpoint, class `Potholes`) in the project
  root, named exactly `best.pt`.
- Docker, or Python 3.11+ with `ffmpeg` on PATH (used as a fallback H.264
  remux step if the OpenCV build has no built-in H.264 encoder).

## Configuration

Copy `.env.example` to `.env` and fill in real values. At minimum you must
set `JWT_SECRET_KEY` — the app refuses to start without it:

```bash
cp .env.example .env
python -c "import secrets; print(secrets.token_hex(32))"   # paste into JWT_SECRET_KEY
```

Set `CORS_ORIGINS` to your frontend's origin(s) (comma-separated). Set
`ADMIN_PASSWORD` to control the bootstrap admin password explicitly —
otherwise a random one-time password is generated and logged on first
startup, and can be rotated afterwards via `POST /auth/change-password`.

## Run with Docker (recommended)

```bash
docker compose up --build
```

This builds the image, mounts `uploads/` and `outputs/` as volumes so
files survive container restarts, and starts gunicorn with the
single-worker gthread config (required by the in-process job/queue design —
do not scale workers without moving `VIDEO_JOBS`/`FSM` state to shared
storage like Redis first).

## Run locally (development only)

```bash
pip install -r requirements.txt
export $(cat .env | xargs)
python app1.py
```

This runs Flask's dev server with `debug=False` (see note in `app1.py` on
why `debug=True` is never used outside local debugging). For anything
resembling production, use gunicorn:

```bash
gunicorn -w 1 --threads 8 -k gthread -b 0.0.0.0:5000 app1:app
```

## Response format

Every JSON response (except the two streaming endpoints below) uses the
same envelope:

```json
{
  "status": "success" | "error",
  "message": "human-readable summary",
  "data": { ... } | null
}
```

HTTP status codes are still meaningful and unchanged (`404`, `401`, `409`,
etc.) — `status` in the body is about the envelope, not the transport.

Auth failures (missing/malformed/expired token) are also normalized to this
shape via Flask-JWT-Extended error handlers, rather than the library's
default `{"msg": "..."}` responses.

## API reference

| Method | Endpoint | Auth | Notes |
|---|---|---|---|
| `POST` | `/auth/register` | none (rate-limited) | password strength enforced |
| `POST` | `/auth/login` | none (rate-limited) | returns `data.access_token` |
| `POST` | `/auth/change-password` | JWT (rate-limited) | rotate your own (or admin's) password |
| `POST` | `/predict` | JWT | single-image inference |
| `POST` | `/videos` | JWT | submit a video job, returns `data.job_id` |
| `GET` | `/jobs/<job_id>` | JWT, owner-only | short status |
| `GET` | `/jobs/<job_id>/details` | JWT, owner-only | detailed progress/alerts |
| `GET` | `/jobs/<job_id>/results` | JWT, owner-only | final results once completed |
| `GET` | `/jobs/<job_id>/events/stream` | JWT, owner-only | SSE progress stream (not enveloped — raw `event:`/`data:` lines) |
| `GET` | `/videos/<job_id>/stream` | JWT, owner-only | live MJPEG preview while processing (not enveloped — raw multipart stream) |
| `POST` | `/videos/<job_id>/stop` | JWT, owner-only | cancel a running job |
| `GET` | `/videos/outputs` | JWT | list your own completed video outputs |
| `GET` | `/outputs/<filename>` | JWT, owner-only | fetch an image/video/alert-frame you own |
| `GET` | `/health` | none | process + model-loaded check |

All job- and output-scoped endpoints return `404` (not `403`) for
resources you don't own, to avoid confirming another user's job/file
exists.

A ready-to-import Postman collection + environment
(`Pothole_API.postman_collection.json`,
`Pothole_API_local.postman_environment.json`) covers every endpoint above,
with Login and Submit Video auto-saving `access_token`/`job_id` for you.

## Known limitations (accepted for now, documented per the readiness review)

- JWT access tokens last 6 hours with no refresh or revocation list.
- `VIDEO_JOBS` and job ownership live in-process memory — single replica
  only; a restart drops in-flight/queued job state (completed video files
  on disk are unaffected).
- SSE/MJPEG streams authenticate via the standard `Authorization` header,
  which means browser `EventSource`/`<img>` tags can't hit them directly
  without a client that can set custom headers (e.g. `fetch` + a
  `ReadableStream`, or a signed short-lived query token if you add one).
- Rate limiting uses in-memory storage by default; set
  `RATELIMIT_STORAGE_URI=redis://...` if you run more than one replica.

## Retention

A background thread sweeps finished jobs older than `JOB_TTL_SECONDS` and
output/upload files older than `OUTPUT_TTL_SECONDS` (see `.env.example`).

## Frontend integration note

Every response field lives under `data` now (e.g. `data.access_token`,
`data.job_id`, `data.videos`), not at the top level. Make sure client code
reads `response.data.<field>`, not `response.<field>` — the earlier
unwrapped format is no longer returned by any endpoint.
