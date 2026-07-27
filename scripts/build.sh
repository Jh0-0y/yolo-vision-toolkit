#!/usr/bin/env bash
# 내 PC(맥 포함)에서 backend/frontend 이미지를 굽고 GHCR 에 올린다.
# 실행 대상은 윈도우(WSL2, amd64) 이므로 맥(arm64)에서 빌드해도 amd64 로 고정한다
# — 안 하면 arm64 이미지가 나와 윈도우에서 실행되지 않는다. (첫 빌드는 에뮬레이션이라 다소 느림)
#
# 사전 준비(1회): GHCR 로그인 — PAT 는 write:packages 권한 필요
#   echo <GITHUB_TOKEN> | docker login ghcr.io -u jh0-0y --password-stdin
#
# 사용:
#   scripts/build.sh              # TAG=latest 로 빌드+푸시
#   TAG=v1.2.3 scripts/build.sh   # 특정 버전으로 빌드+푸시
set -euo pipefail

cd "$(dirname "$0")/.."   # 리포 루트에서 실행 (build context = .)

OWNER=jh0-0y
TAG="${TAG:-latest}"
PLATFORM=linux/amd64

BACKEND="ghcr.io/${OWNER}/yvt-backend:${TAG}"
FRONTEND="ghcr.io/${OWNER}/yvt-frontend:${TAG}"

echo "▶ backend  → ${BACKEND}"
docker buildx build --platform "${PLATFORM}" \
  -f docker/backend.Dockerfile -t "${BACKEND}" --push .

echo "▶ frontend → ${FRONTEND}"
docker buildx build --platform "${PLATFORM}" \
  -f docker/frontend.Dockerfile -t "${FRONTEND}" --push .

echo "✔ pushed:"
echo "    ${BACKEND}"
echo "    ${FRONTEND}"
