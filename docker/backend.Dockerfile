# CUDA runtime base — also runs fine on CPU-only hosts.
# Torch/CUDA already included; uv only adds the remaining deps.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# OpenCV runtime libs required by ultralytics + ffmpeg for H.264 video encoding
# (the Test page's object-tracking output is transcoded to browser-playable H.264).
# git: adaptive-crop 은 git 저장소에서 설치한다 (pyproject 참고).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY backend/pyproject.toml ./
# adaptive-crop 은 비공개 저장소라 빌드 시 GitHub 인증이 필요하다. 토큰을 설치 URL에
# 박으면 .dist-info/direct_url.json 에 그대로 남으므로(이미지에 토큰 유출) insteadOf 로
# 주입하고 설정 파일은 같은 레이어에서 지운다:
#   GH_PAT=... docker buildx build --secret id=gh_pat,env=GH_PAT .
# (fine-grained PAT 의 Contents: Read 권한이면 충분하다)
RUN --mount=type=secret,id=gh_pat \
    if [ -f /run/secrets/gh_pat ]; then \
      git config --global \
        url."https://x-access-token:$(cat /run/secrets/gh_pat)@github.com/".insteadOf \
        "https://github.com/"; \
    fi && \
    uv pip install --system -r pyproject.toml && \
    rm -f /root/.gitconfig

COPY backend/ ./

ENV YVT_DATA_DIR=/data
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
