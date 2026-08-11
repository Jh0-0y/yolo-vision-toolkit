# CUDA runtime base — also runs fine on CPU-only hosts.
# Torch/CUDA already included; uv only adds the remaining deps.
#
# 태그를 고를 때 파이썬 버전을 먼저 본다. pytorch 공식 이미지는 2.8.0 이하가 conda
# 기반 Python 3.11, **2.10.0부터 system Python 3.12**다. 이 프로젝트는 3.12 기준이고
# (backend/pyproject.toml의 requires-python) 의존성 중 adaptive-crop이 >=3.12를
# 강제하므로, 3.11 베이스에서는 설치 자체가 거부된다.
FROM pytorch/pytorch:2.10.0-cuda12.6-cudnn9-runtime

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
#
# --break-system-packages: Ubuntu 24.04 의 시스템 파이썬은 PEP 668 로 "외부 관리"
# 표시가 붙어 있어 이 플래그 없이는 설치가 거부된다(exit 2). 컨테이너라 보호할
# 배포판 파이썬 환경이 따로 없고, 베이스의 torch 도 같은 site-packages 에 있어서
# 여기 설치해야 uv 가 그 torch 를 이미 있는 것으로 인식한다 — 안 그러면 CUDA 휠을
# 통째로 다시 받는다.
RUN --mount=type=secret,id=gh_pat \
    if [ -f /run/secrets/gh_pat ]; then \
      git config --global \
        url."https://x-access-token:$(cat /run/secrets/gh_pat)@github.com/".insteadOf \
        "https://github.com/"; \
    fi && \
    uv pip install --system --break-system-packages -r pyproject.toml && \
    rm -f /root/.gitconfig

COPY backend/ ./

ENV YVT_DATA_DIR=/data
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
