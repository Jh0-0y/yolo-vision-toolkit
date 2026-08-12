"""VideoManager: thread-pool runner for video frame extraction.

Extraction is CPU/IO (cv2 decode releases the GIL), so it runs on a small
ThreadPoolExecutor — deliberately NOT the GPU-serialized ProcessPoolExecutor in
``runner.py``. This keeps frame extraction from blocking labeling/training.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from app.core.config import settings
from app.domain.video import ExtractParams, extract_frames
from infra import jobs


def task_dir(video_id: str) -> Path:
    """Progress/sentinel dir; reuses jobs_dir so the shared SSE reader finds it."""
    return settings.jobs_dir / video_id


class VideoManager:
    def __init__(self) -> None:
        self._executor: ThreadPoolExecutor | None = None
        self._futures: dict[str, Future] = {}

    def _get_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=2, thread_name_prefix="video-extract"
            )
        return self._executor

    def submit(
        self,
        video_id: str,
        video_path: Path,
        raw_dir: Path,
        stem: str,
        params: ExtractParams,
    ) -> None:
        # 재추출이면 이전 이벤트가 섞이지 않게 진행률을 비우고 시작한다
        job = jobs.at(settings.jobs_dir, video_id).reset()

        future = self._get_executor().submit(
            extract_frames,
            video_path,
            raw_dir,
            stem,
            params,
            job.emit,
            job.cancel_path,
        )
        self._futures[video_id] = future
        future.add_done_callback(lambda _f: self._futures.pop(video_id, None))

    def cancel(self, video_id: str) -> bool:
        future = self._futures.get(video_id)
        if future is None:
            return False
        if future.cancel():  # still queued — never started
            return True
        jobs.at(settings.jobs_dir, video_id).request_cancel()  # running: signal the worker
        return True

    def is_active(self, video_id: str) -> bool:
        future = self._futures.get(video_id)
        return future is not None and not future.done()


video_manager = VideoManager()
