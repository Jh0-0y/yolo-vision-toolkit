"""긴 잡 — 연구실 크롭 런과 모델 채점.

둘 다 오래 도는 GPU 잡(영상 전체 인코딩, 라벨셋 전체 추론)이라 상주 추론 워커에
맞지 않는다 — `infer_manager` 의 180초 유휴 수거기가 몇 분짜리 잡을 도중에 죽인다.
그래서 이 매니저가 **자기 `ProcessPoolExecutor(max_workers=1)`** 를 소유한다(수거기
없음, DB 기록 없음). 진행률은 `jobs_dir/{job_id}/progress.jsonl` 로 흐른다.

크롭 런의 **산출물**은 여기가 아니라 연구실 아래 런 디렉터리에 남는다 — 자리와
수명은 `services/lab_crop_runs.py` 를 본다(TTL 없음, 지우는 것은 사용자뿐). 벤치마크
런도 같은 성질이다 — 자리와 수명은 `services/benchmarks.py` 를 본다.
"""

from __future__ import annotations

import multiprocessing
import threading
from concurrent.futures import Future, ProcessPoolExecutor

from app.core.config import settings
from infra import jobs


class TestJobManager:
    """Owns two single-worker pools so a long video job and a
    model-comparison/analysis job don't block each other (they used to share one
    worker). Each pool is spawn-based, has no idle reaper, and does no DB work."""

    def __init__(self) -> None:
        self._executors: dict[str, ProcessPoolExecutor] = {}
        self._lock = threading.Lock()
        self._futures: dict[str, Future] = {}

    def _get_executor(self, kind: str) -> ProcessPoolExecutor:
        with self._lock:
            executor = self._executors.get(kind)
            # 자식이 급사(segfault·언피클 실패 등)하면 풀이 영구 broken이 된다 —
            # 서버 재시작 없이 새 풀로 갈아끼워 다음 잡부터 정상 동작하게 한다.
            if executor is not None and getattr(executor, "_broken", False):
                executor.shutdown(wait=False, cancel_futures=True)
                executor = None
            if executor is None:
                executor = ProcessPoolExecutor(
                    max_workers=1, mp_context=multiprocessing.get_context("spawn")
                )
                self._executors[kind] = executor
            return executor

    def _prepare(self, job_id: str):
        return jobs.at(settings.jobs_dir, job_id).prepare()

    def submit_lab_crop(self, job_id: str, cfg: dict) -> None:
        """연구실 크롭 런 — 검출 + 렌더가 한 잡 안에 있다. 무거운 GPU 잡이
        모델 채점과 부딪히지 않도록 전용 "video" 풀을 쓴다."""
        from app.workers import lab_crop_worker

        self._prepare(job_id)
        future = self._get_executor("video").submit(
            lab_crop_worker.run_lab_crop, job_id, cfg, str(settings.jobs_dir)
        )
        self._futures[job_id] = future

    def submit_compare(self, job_id: str, cfg: dict) -> None:
        from app.workers import compare_worker

        self._prepare(job_id)
        future = self._get_executor("eval").submit(
            compare_worker.run_compare, job_id, cfg, str(settings.jobs_dir)
        )
        self._futures[job_id] = future

    def is_active(self, job_id: str) -> bool:
        future = self._futures.get(job_id)
        return future is not None and not future.done()

    def cancel(self, job_id: str) -> None:
        jobs.at(settings.jobs_dir, job_id).request_cancel()

    def shutdown(self) -> None:
        with self._lock:
            for ex in self._executors.values():
                ex.shutdown(wait=False, cancel_futures=True)
            self._executors = {}


test_job_manager = TestJobManager()
