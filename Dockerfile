# syntax=docker/dockerfile:1

###############################################################################
# Stage 1: build -- install Python deps
###############################################################################
FROM python:3.10-slim AS build

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

# System libraries required at runtime by OpenCV / Ultralytics.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgthread-2.0-0 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install the CPU build of PyTorch first to keep the image small (no CUDA).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

###############################################################################
# Stage 2: runtime -- minimal image with the app only
###############################################################################
FROM python:3.10-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgthread-2.0-0 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Reuse the packages installed in the build stage.
COPY --from=build /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=build /usr/local/bin /usr/local/bin

# Application code. best.pt must be present next to this Dockerfile
# (see README / build notes). Add it before building:
#   docker build -t pothole-api .
COPY FSM.py app1.py requirements.txt ./
COPY best.pt ./best.pt

RUN mkdir -p /app/uploads /app/outputs \
    && useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:5000/health', timeout=4).status==200 else 1)"

# Single worker is required: the app keeps jobs, locks and SSE/MJPEG streams
# in process memory (see app1.py). Threads handle concurrent requests; a long
# timeout keeps long-lived streams from being reaped.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "8", "--worker-class", "gthread", "--timeout", "6000", "--graceful-timeout", "30", "--access-logfile", "-", "--error-logfile", "-", "app1:app"]
