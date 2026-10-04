# =====================================================================
# rainnow - CPU image (pipeline + real-time service). Multi-arch: amd64 and arm64 (edge devices).
#
#   docker build --target serve    -t rainnow:serve .      # API only (default target)
#   docker build --target pipeline -t rainnow:pipeline .   # full pipeline incl. pysteps
#   docker buildx build --platform linux/arm64 --target serve -t rainnow:serve-arm64 .   # Raspberry Pi
#
# GPU training image for AAU AI-Lab: see Dockerfile.gpu and src/ailab/.
# Credentials are injected at runtime (.env / env vars); never baked into the image.
# =====================================================================
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    AWS_DEFAULT_REGION=eu-north-1

WORKDIR /app

# libgomp1: OpenMP runtime for LightGBM (h5py / pyproj wheels bundle HDF5 and PROJ)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-optional.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install -r requirements.txt

COPY src ./src
COPY README.md ./

# ---------------------------------------------------------------- pipeline
FROM base AS pipeline
RUN python -m pip install -r requirements-optional.txt || echo "pysteps not installed (optional)"
ENTRYPOINT ["python", "src/run.py"]
CMD ["--help"]

# ---------------------------------------------------------------- serve (default)
FROM base AS serve
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD curl -fs http://localhost:8000/health || exit 1
CMD ["uvicorn", "deploy.app:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
