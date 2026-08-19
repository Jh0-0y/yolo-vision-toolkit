"""학습에 쓰이는 파일 조작 — data.yaml 찾기, 산출물 읽기, 빠른 디스크 스테이징.

    dataset   학습 트리에서 data.yaml 찾기 · 정규화
    results   학습이 남긴 산출물 읽기 (results.csv · per-class 지표)
    staging   학습 데이터셋을 빠른 디스크(SSD)로 복사하고 치우기

**언제** 할지는 서비스 계층(train_manager)이 정하고, 여기는 **어떻게** 할지만
안다. DB 도 프로세스도 모른다.

학습이 먹을 트리를 **만드는** 것은 `lib/labels/dataset_export.materialize` 다 —
데이터셋에서 train/val 을 하드링크로 펼쳐 `runs/{run_id}/dataset/` 에 놓는다.

학습 실행(ultralytics 호출) 자체는 아직 `app/workers/train_runner.py` 에 있다.
"""

from lib.train import dataset, results, staging

__all__ = ["dataset", "results", "staging"]
