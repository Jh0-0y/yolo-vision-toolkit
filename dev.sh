#!/usr/bin/env bash
# 백엔드(uvicorn:8010) + 프론트엔드(vite:5173) 동시 실행
# Ctrl+C 한 번으로 둘 다(자식 프로세스까지) 종료됩니다.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

pids=()

# 프로세스 트리(자식 포함) 재귀 종료 — uv→python, npm→vite 처럼
# 래퍼가 신호를 자식에게 전달하지 않는 경우까지 확실히 정리한다.
kill_tree() {
  local pid=$1
  for child in $(pgrep -P "$pid" 2>/dev/null); do
    kill_tree "$child"
  done
  kill -TERM "$pid" 2>/dev/null || true
}

cleanup() {
  trap - INT TERM EXIT
  echo ""
  echo "[dev] 종료 중..."
  for pid in "${pids[@]}"; do
    kill_tree "$pid"
  done
  wait 2>/dev/null || true
  echo "[dev] 종료 완료."
}
trap cleanup INT TERM EXIT

echo "[dev] 백엔드 시작 → http://localhost:8010"
( cd "$ROOT/backend" && exec uv run uvicorn app.main:app --reload --port 8010 ) &
pids+=($!)

echo "[dev] 프론트엔드 시작 → http://localhost:5173"
( cd "$ROOT/frontend" && exec npm run dev ) &
pids+=($!)

echo "[dev] 실행 중 (종료: Ctrl+C)"
# 어느 하나라도 죽으면 전체 종료.
# (`wait -n`은 bash 4.3+ 전용 — macOS 기본 bash 3.2 호환을 위해 폴링으로 대체)
while true; do
  for pid in "${pids[@]}"; do
    kill -0 "$pid" 2>/dev/null || exit 0
  done
  sleep 1
done
