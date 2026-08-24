"""VideoManager: thread-pool runner for video frame extraction.

Extraction is CPU/IO (cv2 decode releases the GIL), so it runs on a small
ThreadPoolExecutor — deliberately NOT the GPU-serialized ProcessPoolExecutor in
``runner.py``. This keeps frame extraction from blocking labeling/training.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from app.core.config import settings
from infra import jobs
from lib.media.extract import ExtractParams, extract_frames


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
        *,
        delete_source: bool = False,
        on_done: Callable[[dict | None], None] | None = None,
    ) -> None:
        """프레임을 뽑는다. `delete_source` 면 끝난 뒤 원본 영상을 지운다.

        데이터셋으로 가져올 때가 그렇다 — 영상은 프레임을 얻는 수단일 뿐이라
        보관하지 않는다. 실패하든 취소하든 지운다(반쯤 올라온 파일이 남지 않게).

        `on_done` 은 끝난 뒤 결과(실패하면 None)를 받는다. **몇 장이 나왔는지는
        끝나야 알기 때문에** 출처 기록을 그때 채운다.
        """
        # 재추출이면 이전 이벤트가 섞이지 않게 진행률을 비우고 시작한다
        job = jobs.at(settings.jobs_dir, video_id).reset()

        def _run() -> dict:
            try:
                return extract_frames(
                    video_path, raw_dir, stem, params, job.emit, job.cancel_path
                )
            finally:
                if delete_source:
                    video_path.unlink(missing_ok=True)

        def _finish(f) -> None:
            self._futures.pop(video_id, None)
            if on_done is None:
                return
            try:
                on_done(f.result())
            except Exception:
                # 추출이 죽었어도 기록은 남겨야 한다 — 이름이 이미 잡혔기 때문이다
                on_done(None)

        future = self._get_executor().submit(_run)
        self._futures[video_id] = future
        future.add_done_callback(_finish)

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
