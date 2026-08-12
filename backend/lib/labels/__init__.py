"""라벨과 클래스 — YOLO 라벨 파일 형식, 프로젝트 라벨 저장소, 클래스 목록.

    io         라벨 .txt · box meta · data.yaml 읽고 쓰기 (좌표 변환 포함)
    store      프로젝트 디렉터리 기준 박스 · 검수 상태
    classes    프로젝트 클래스 목록 CRUD (추가 · 이름변경 · 삭제)
    registry   여러 모델의 클래스 이름을 하나로 합치기 (앙상블 · 비교)

`io` 가 바닥이다 — `store` 와 `classes` 가 그 위에 프로젝트 규약(labels/{stem}.txt ·
reviewed.json · classes.json)을 얹는다. `registry` 는 파일을 모른다.

쓰기는 전부 원자적이다(`io.atomic_write_text`) — 학습이 반쯤 쓰인 라벨 파일을
읽는 일이 없어야 하기 때문이다.
"""

from lib.labels import classes, io, registry, store

__all__ = ["classes", "io", "registry", "store"]
