"""학습에 쓰이는 파일 조작 — 업로드 데이터셋 수명, 빠른 디스크 스테이징.

    uploads   업로드된 데이터셋 폴더 찾기 · 자동삭제 플래그 · 삭제
    staging   학습 데이터셋을 빠른 디스크(SSD)로 복사하고 치우기

**언제** 할지는 서비스 계층(train_manager)이 정하고, 여기는 **어떻게** 할지만
안다. DB 도 프로세스도 모른다.

학습 실행(ultralytics 호출) 자체는 아직 `app/workers/train_runner.py` 에 있다.
"""

from lib.train import staging, uploads

__all__ = ["staging", "uploads"]
