"""검출 — 모델을 돌려 박스를 얻고, 합치고, 정답과 맞춰본다.

    predict    이미지 한 장 추론 (여러 모델 -> 융합된 박스)
    labeling   이미지 폴더 오토라벨링 (앙상블 + 라벨 파일 쓰기)
    ensemble   여러 모델의 박스를 Weighted Box Fusion 으로 병합
    evaluate   검출을 정답 라벨과 맞춰 P/R/F1 · AP 계산

`ensemble` 이 바닥이다 — `predict`·`labeling` 은 융합에, `evaluate` 는 IoU 에
기댄다. 평가를 별도 패키지로 떼면 그 의존이 패키지 경계를 넘어가므로, "검출
결과를 다루는 계산"으로 묶어 한 패키지에 둔다.

**torch·ultralytics 는 함수 안에서 import 한다.** API 프로세스가 이 모듈을
읽는 것만으로 프레임워크가 올라오면 프로세스 분리가 무의미해진다.

디바이스는 이미 해석된 문자열을 **인자로 받는다** — 설정을 모른다.
"""

from lib.detect import ensemble, evaluate, labeling, predict

__all__ = ["ensemble", "evaluate", "labeling", "predict"]
