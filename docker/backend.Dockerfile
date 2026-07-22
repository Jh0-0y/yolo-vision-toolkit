# CUDA runtime base — also runs fine on CPU-only hosts.
# Torch/CUDA already included; uv only adds the remaining deps.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# OpenCV runtime libs required by ultralytics
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY backend/pyproject.toml ./
RUN uv pip install --system -r pyproject.toml

COPY backend/ ./

ENV YVT_DATA_DIR=/data
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
