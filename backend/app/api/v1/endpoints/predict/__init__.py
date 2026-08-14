"""Test 플레이그라운드 — 학습된 모델을 데이터셋을 건드리지 않고 돌려 보는 곳.

두 계열이 `/predict` 아래 함께 산다. 계열별로 나누고 여기서 다시 묶는다 —
**경로와 OpenAPI 태그는 나누기 전과 같다.**

    inference   이미지 한 장 추론 · 상주 모델         POST ""  · /residents
    compare     테스트셋으로 모델 채점                 /compare

크롭은 여기 없다. 연구실(`endpoints/lab_crops.py`)이 시작부터 산출물까지 한 곳에서
맡는다 — 옛 annotate·live 계열은 그리로 옮겨 가며 사라졌다.

공통 조각(모델 경로 풀기 · SSE · 검출기 파싱)은 `common.py` 에 있다.
"""

from fastapi import APIRouter

from app.api.v1.endpoints.predict import compare, inference

# 접두사·태그는 각 하위 라우터가 들고 있다 — POST "" (경로가 빈 라우트)는
# 접두사 없는 라우터에 실을 수 없기 때문이다. 여기서는 묶기만 한다.
router = APIRouter()

for _sub in (inference.router, compare.router):
    router.include_router(_sub)
