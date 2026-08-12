"""Test 플레이그라운드 — 학습된 모델을 데이터셋을 건드리지 않고 돌려 보는 곳.

네 계열이 `/predict` 아래 함께 산다. 한 파일이었을 때는 603줄이었고 계열끼리
아무 관계가 없어 서로를 가렸다. 계열별로 나누고 여기서 다시 묶는다 — **경로와
OpenAPI 태그는 나누기 전과 같다.**

    inference   이미지 한 장 추론 · 상주 모델         POST ""  · /residents
    annotate    영상 오버레이 렌더 · 크롭 컷           /annotate · /crop-cut
    live        검출 1회 + 튜닝마다 좌표 재계산        /live
    compare     테스트셋으로 모델 채점                 /compare

공통 조각(모델 경로 풀기 · SSE · 검출기 파싱)은 `common.py` 에 있다.
"""

from fastapi import APIRouter

from app.api.v1.endpoints.predict import annotate, compare, inference, live

# 접두사·태그는 각 하위 라우터가 들고 있다 — POST "" (경로가 빈 라우트)는
# 접두사 없는 라우터에 실을 수 없기 때문이다. 여기서는 묶기만 한다.
router = APIRouter()

for _sub in (inference.router, annotate.router, live.router, compare.router):
    router.include_router(_sub)
