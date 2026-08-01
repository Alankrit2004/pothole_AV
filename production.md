# Pothole API — Production Readiness Review & Plan

**Verdict (original):** NOT ready for production. The code runs, but there are blockers (some are one-liner fixes), plus structural/ops gaps.

## Decisions (confirmed with owner)

| Decision | Resolution |
|---|---|
| User registration | **Stays open** — anyone may register. Plan keeps `/auth/register` public, but must add password-strength rules to slow abuse. |
| Model weights | **`best.pt.zip` is correct.** Verified: it is a valid Ultralytics detect checkpoint (`task=detect`, class `Potholes`) and the `best/` subdirectory layout is *required* by PyTorch's archive format (a root-level `data.pkl` fails with "file in archive is not in a subdirectory"). `YOLO("best.pt")` loads it fine on torch 2.10 + ultralytics 8.2.103 (ultralytics patches `torch.load` to `weights_only=False`). The app needs it **named `best.pt`**, so the container must ship it as `best.pt` (currently present next to `Dockerfile`; `COPY best.pt ./best.pt` already works — no Dockerfile change needed). |
| Frontend | **Separate app** → API must enable CORS for the configured frontend origin(s) (`flask-cors`, origin from env var, not `*`). |
| Video codec | **H.264 required** — current `mp4v` output (`app1.py:204`) is not browser-playable. |

## Blockers

1. ~~Missing model file~~ — **Resolved** (see decisions). Keep `best.pt` in the build context; add it to the repo or document a `wget`/volume step.
2. **Admin password cannot be rotated** — `app1.py:50` hardcodes `admin123`; there is no change-password endpoint. Since registration is open, also enforce password strength (min length, not just a hash).
3. **Weak default JWT secret** — `app1.py:27`: `os.environ.get("JWT_SECRET_KEY", "change-this-secret-key")`. Fail fast: refuse to start if unset. Tokens last 6 h with no refresh/revocation — accept for now, but document.
4. **Fixed video job ID + no per-user authorization (IDOR)** — `app1.py:38,353`: every upload uses `FIXED_VIDEO_JOB_ID = "123_video_job"`, so concurrent video submissions clobber each other. `/jobs/<job_id>/*`, `/videos/<job_id>/stop`, streams and `/outputs/*` never check `current_user` owns the resource.
5. ~~No `.gitignore`~~ — **Resolved** (`.gitignore` added: excludes `app2_users.json` password hashes, `uploads/`, `outputs/`, caches, `.pt` files).

## High

6. **No rate limiting** on `/auth/login` and `/auth/register` → brute-force friendly.
7. **`VIDEO_JOBS` never cleaned up** (`app1.py:33`) → memory grows; jobs are in-memory only (single-replica, lost on restart).
8. **No disk cleanup** — `outputs/` (videos + alert frames) accumulate forever → disk exhaustion.
9. **Output videos are `mp4v`** — confirmed: switch to H.264 (`avc1`) encode or ffmpeg remux.
10. **`gunicorn` missing from `requirements.txt`** (only in Docker CMD); **torch version unpinned** (Dockerfile installs latest CPU wheel → drift risk; pin e.g. `torch==2.x.x+cpu`).
11. **Error responses leak internals** — e.g. `"could not save user: {error}"` (`app1.py:137`).
12. **CORS** — add `flask-cors`, allowlist frontend origin(s) from env (`CORS_ORIGINS`), keep credentials/token flows in mind.

## Medium

13. **SSE/MJPEG need `Authorization` header** (`app1.py:402,461`) — browser `EventSource`/`<img>` can't send headers → live streams need token-in-query (leaks into logs) or cookie-based auth.
14. **`app.run(debug=True)`** (`app1.py:657`) — Werkzeug debugger = RCE if run directly; gunicorn path is safe.
15. **No tests, no CI, no README/deploy docs** (single commit "Add files via upload").
16. **No auth-failure logging/audit trail**; `/health` only checks process up, not model loaded.
17. **No HTTPS/proxy config guidance, no compose file**, no volume story beyond the Dockerfile.

## What's already OK

- `secure_filename` + uuid-prefixed uploads, `send_from_directory` (no traversal), 200 MB cap
- Password hashing via Werkzeug (scrypt) with atomic file save
- Model loaded once at startup, threads + lock for shared state, `X-Accel-Buffering: no`
- Dockerfile: multi-stage, CPU torch, non-root user, healthcheck, single-worker gthread (correct for the in-process design)

## Remediation plan

| # | Fix
|---|-----|
| 1 | Generate job IDs per upload (`uuid4`); enforce `job["user"] == current_user` on all job/output/stream endpoints
| 2 | `ADMIN_PASSWORD` from env for bootstrap; add `/auth/change-password`; password strength rules (registration stays open)
| 3 | Fail-fast JWT secret (raise if unset); document in README/compose
| 4 | Add README with deploy steps
| 5 | Rate limit auth endpoints (`flask-limiter`); strip internal error details from responses
| 6 | Job TTL sweep on `VIDEO_JOBS`; retention job for `outputs/`
| 7 | H.264 encoding (`avc1` via ffmpeg/openh264) or remux step
| 8 | Add `gunicorn` + pinned `torch` (CPU) to `requirements.txt`
| 9 | CORS via `flask-cors`, origins from `CORS_ORIGINS` env
| 10 | Auth logging (success/failure + user), extended `/health` (model loaded)
| 11 | Optional: split `app1.py` monolith into auth / jobs / routes modules
