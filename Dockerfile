# MergeSE - container image
# Builds a CPU-only image. For GPU evaluations on the host, run with --gpus all
# and torch will pick CUDA automatically.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/data/hf \
    TRANSFORMERS_CACHE=/data/hf

# System deps (curl/git for HF Hub downloads, build for safetensors wheels on uncommon archs)
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates tini && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install --index-url https://download.pytorch.org/whl/cpu torch && \
    pip install -r requirements.txt

COPY . /app

RUN mkdir -p /data/hf /app/artifacts /app/uploads
VOLUME ["/data/hf", "/app/artifacts", "/app/uploads"]

EXPOSE 8765

# Report container health via the app's /api/health endpoint so orchestrators
# can detect and restart a wedged server. start-period covers the first-boot
# import of torch/transformers.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 CMD curl -fsS http://localhost:8765/api/health || exit 1

# tini reaps zombie processes from cancelled jobs
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["gunicorn", "-w", "2", "-k", "gthread", "--threads", "8", "-t", "0", \
     "-b", "0.0.0.0:8765", "server.app:app"]
