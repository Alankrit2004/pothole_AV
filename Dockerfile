# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Stage 1: build dependencies into a venv so the runtime stage doesn't carry
# build tooling (gcc, headers, pip cache) into the final image.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2: runtime image -- only what's needed to run the app.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

WORKDIR /app

# libgl1/libglib2.0-0: required by opencv-python-headless at import time.
# ffmpeg: used as the H.264 remux fallback in app1.py if the OpenCV build
# has no built-in H.264 encoder (see _remux_to_h264).
# curl: used by the container HEALTHCHECK below.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        ffmpeg \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Non-root user -- the app writes only to uploads/ and outputs/, both
# created and chowned below.
RUN useradd --create-home --uid 10001 appuser

COPY app1.py FSM.py ./
# best.pt must be present in the build context; see README "Requirements".
COPY best.pt ./best.pt

RUN mkdir -p uploads outputs \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Single worker, threaded (gthread): required by the in-process design --
# VIDEO_JOBS, OUTPUT_OWNERS, and the FSM frame broadcaster all live in
# process memory and are not shared across workers/replicas. Do not raise
# -w without first moving that state to shared storage (e.g. Redis).
CMD ["gunicorn", "-w", "1", "--threads", "8", "-k", "gthread", \
     "-b", "0.0.0.0:5000", "--timeout", "120", "app1:app"]
