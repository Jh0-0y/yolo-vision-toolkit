#!/usr/bin/env bash
# 내 PC(맥 포함)에서 backend/frontend 이미지를 굽고 GHCR 에 올린다.
# 실행 대상은 윈도우(WSL2, amd64) 이므로 맥(arm64)에서 빌드해도 amd64 로 고정한다
# — 안 하면 arm64 이미지가 나와 윈도우에서 실행되지 않는다. (첫 빌드는 에뮬레이션이라 다소 느림)
#
# 사전 준비(1회): GHCR 로그인 — PAT 는 write:packages 권한 필요
#   echo <GITHUB_TOKEN> | docker login ghcr.io -u jh0-0y --password-stdin
#
# 백엔드 빌드에는 GH_PAT 가 필요하다 — 크롭 좌표 계산에 쓰는 adaptive-crop 이
# 비공개 저장소라, 빌드 중 git 이 인증할 수 있어야 한다. --secret 으로 넘기므로
# 토큰이 이미지 레이어에 남지 않는다 (URL 에 박으면 direct_url.json 에 남는다).
# 필요한 권한은 그 저장소 읽기(fine-grained PAT 의 Contents: Read)뿐이다.
#
# 사용:
#   GH_PAT=<token> scripts/build.sh              # TAG=latest 로 빌드+푸시
#   GH_PAT=<token> TAG=v1.2.3 scripts/build.sh   # 특정 버전으로 빌드+푸시
set -euo pipefail

cd "$(dirname "$0")/.."   # 리포 루트에서 실행 (build context = .)

OWNER=jh0-0y
TAG="${TAG:-latest}"
PLATFORM=linux/amd64

BACKEND="ghcr.io/${OWNER}/yvt-backend:${TAG}"
FRONTEND="ghcr.io/${OWNER}/yvt-frontend:${TAG}"

# 없으면 여기서 멈춘다 — 빌드를 태운 뒤 pip 설치 단계에서 실패하는 것보다 낫다.
if [ -z "${GH_PAT:-}" ]; then
  echo "✖ GH_PAT 가 없습니다. adaptive-crop(비공개 저장소) 설치에 필요합니다." >&2
  echo "  GH_PAT=<github_token> TAG=${TAG} scripts/build.sh" >&2
  exit 1
fi

echo "▶ backend  → ${BACKEND}"
docker buildx build --platform "${PLATFORM}" --secret id=gh_pat,env=GH_PAT \
  -f docker/backend.Dockerfile -t "${BACKEND}" --push .

echo "▶ frontend → ${FRONTEND}"
docker buildx build --platform "${PLATFORM}" \
  -f docker/frontend.Dockerfile -t "${FRONTEND}" --push .

echo "✔ pushed:"
echo "    ${BACKEND}"
echo "    ${FRONTEND}"
