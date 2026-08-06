# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# GPU build.
#
# Base image: nvidia/cuda 12.8 runtime with cuDNN preinstalled. ALL CUDA and
# cuDNN libraries live inside this image -- nothing is installed on the host,
# so this cannot interfere with other NVIDIA apps/drivers on the machine.
# The host only provides the NVIDIA GPU driver (through the NVIDIA Container
# Toolkit), which is passed into the container read-only at runtime.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Stage 1: build dependencies into a venv so the runtime stage doesn't carry
# build tooling (gcc, headers, pip cache) into the final image.
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 AS builder

WORKDIR /app

# python3/python3-venv: Ubuntu 24.04 ships Python 3.12; the venv is created
# here so the runtime stage can reuse it with matching interpreter paths.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-venv \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2: runtime image -- same CUDA base (so the venv's interpreter path
# matches), carrying only what's needed to run the app.
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 AS runtime

WORKDIR /app

# python3: provides /usr/bin/python3.12 that /opt/venv links against.
# libgl1/libglib2.0-0: required by opencv-python-headless at import time.
# ffmpeg: used as the H.264 remux fallback in app.py if the OpenCV build
# has no built-in H.264 encoder (see _remux_to_h264).
# curl: used by the container HEALTHCHECK below.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
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

COPY app.py FSM.py ./
# best.pt must be present in the build context; see README "Requirements".
COPY best.pt ./best.pt

RUN mkdir -p uploads outputs \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 5000

# start-period is generous: the container must initialize a CUDA context
# before the model is usable.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=5 \
    CMD curl -f http://localhost:5000/health || exit 1

# Single worker, threaded (gthread): required by the in-process design --
# VIDEO_JOBS, OUTPUT_OWNERS, and the FSM frame broadcaster all live in
# process memory and are not shared across workers/replicas. Do not raise
# -w without first moving that state to shared storage (e.g. Redis).
CMD ["gunicorn", "-w", "1", "--threads", "8", "-k", "gthread", \
     "-b", "0.0.0.0:5000", "--timeout", "120", "app:app"]
