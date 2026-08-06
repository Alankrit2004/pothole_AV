# Pothole Detection API — GPU

Flask + YOLO (Ultralytics) service for pothole detection on images and video,
with JWT auth, per-user video jobs, live MJPEG preview, and SSE progress
events. This variant runs inference on an **NVIDIA GPU** (CUDA 12.8 + cuDNN).

## Requirements

- An NVIDIA GPU supported by CUDA 12.8 (see
  [NVIDIA support matrix](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities)).
  Tested against a GTX 1660 (Turing, sm_75).
- Host NVIDIA driver recent enough for CUDA 12.8. Check with `nvidia-smi`.
- **NVIDIA Container Toolkit** installed on the host (`nvidia-ctk`, the
  `nvidia` container runtime). This is the only GPU-related piece installed
  on the host.
- `best.pt` (Ultralytics detect checkpoint, class `Potholes`) in the project
  root, named exactly `best.pt`.

## Why your host stays clean

The Docker image is built from `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04`,
which contains **the complete CUDA runtime and cuDNN libraries inside the
container**. Nothing is installed on your host:

- No CUDA Toolkit, no cuDNN, no `nvcc`, no library or path changes on the
  host machine — so your other NVIDIA apps/drivers can't be disturbed.
- The host only exposes the **GPU driver** to the container through the
  NVIDIA Container Toolkit (read-only passthrough), the same way it would
  for any containerized GPU workload.

This is the "isolated environment" for CUDA/cuDNN development: every tool and
library you need for GPU inference lives in the image, versioned with it, and
goes away when you delete the image.

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

The app trusts `X-Forwarded-For`/`X-Forwarded-Proto` from a reverse proxy in
front of it (`ProxyFix`), so rate limits and auth logs see real client IPs.
Deploy it only behind a proxy you control that sets these headers.

## Run with Docker (recommended)

```bash
docker compose up --build
```

This builds the image (base ~2.9 GB + CUDA torch ~2.6 GB, so the first pull
takes a while), requests all GPUs via the `nvidia` runtime, mounts
`uploads/`/`outputs/` as named volumes, and starts gunicorn with the
single-worker gthread config (required by the in-process job/queue design —
do not scale workers without moving `VIDEO_JOBS`/`FSM` state to shared
storage like Redis first).

The service binds to **`127.0.0.1:5002`** by default (localhost-only, not on
the LAN) so it can coexist with other apps on the same host. Change the
`ports:` mapping in `docker-compose.yml` if 5002 is taken — the port is only
a host-side mapping; the app itself always listens on 5000 inside the
container.

Alternative, plain `docker run`:

```bash
docker build -t pothole-gpu .
docker run --gpus all -p 127.0.0.1:5002:5000 \
  --env-file .env \
  -v pothole_gpu_uploads:/app/uploads \
  -v pothole_gpu_outputs:/app/outputs \
  pothole-gpu
```

## Expose over Tailscale (optional)

Because the app binds to localhost, reach it from your tailnet with
`tailscale serve`, which proxies a tailnet port to a local port:

```bash
# Private tailnet only (any port you like):
tailscale serve --bg --https=5002 http://127.0.0.1:5002

# Public internet (Funnel) — only ports 443, 8443, 10000 are allowed:
tailscale funnel --bg --https=8443 http://127.0.0.1:5002
```

You can run multiple `tailscale serve`/`funnel` commands at once for
different local apps — each port on your tailnet address gets its own URL
(e.g. `https://<machine>:5002/`), so this does not clash with another app
already served on a different port. See `tailscale serve status` /
`tailscale funnel status` to list them.

## Verify the GPU is actually used

```bash
# Inside the container:
docker compose exec pothole-gpu python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expect: True NVIDIA GeForce GTX 1660
```

Also, `/health` returns `{"data":{"model_loaded":true}}` — the YOLO model is
loaded into CUDA memory at startup if a device is available.

## Run locally (development only)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # requires a local CUDA-capable environment
export $(cat .env | xargs)
python app.py
```

This runs Flask's dev server with `debug=True` — for local development only.
Never run this in production: the Werkzeug debugger is a remote-code-execution
risk. Production runs via gunicorn (see Dockerfile), which bypasses `__main__`:

```bash
gunicorn -w 1 --threads 8 -k gthread -b 0.0.0.0:5000 app:app
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
