"""TileManager: 타일링 잡 실행기 — 스레드 풀.

타일링은 crop + JPEG 인코딩이라 torch 를 쓰지 않는다. GPU 프로세스 풀
(`label_manager`)에 넣으면 오토라벨링·학습 뒤에 줄을 서게 되는데, GPU 를 기다릴
이유가 없다. 영상 추출이 같은 판단으로 스레드 풀을 쓴다(`video_manager`).

**DB 행을 만들지 않는다.** 상태는 `progress.jsonl` 에서 파생한다 — 크롭 런과 같은
길이고, 이어붙일 두 번째 진실을 만들지 않는다.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from app.core.config import settings
from app.services import datasets
from infra import jobs
from lib.labels.dataset_tile import TileCancelled, TileDatasetParams, materialize


class TileManager:
    def __init__(self) -> None:
        self._executor: ThreadPoolExecutor | None = None
        self._futures: dict[str, Future] = {}

    def _get_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="tiling")
        return self._executor

    def submit(
        self,
        tile_id: str,
        src_dir: Path,
        out_dir: Path,
        reviewed: set[str],
        params: TileDatasetParams,
        *,
        on_done: Callable[[dict | None], None] | None = None,
    ) -> None:
        """`tile_id` 는 **파생 데이터셋 id 를 그대로 쓴다** — 크롭 런이 `crop_id` 를
        잡 id 로 쓰는 것과 같다. 산출물과 진행률이 같은 이름을 갖는다."""
        job = jobs.at(settings.jobs_dir, tile_id).prepare()

        def _run() -> dict:
            try:
                result = materialize(
                    dataset_dir=src_dir,
                    out_dir=out_dir,
                    reviewed=reviewed,
                    params=params,
                    emit=job.emit,
                    cancel_path=job.cancel_path,
                )
            except TileCancelled:
                job.emit({"phase": "cancelled"})
                return {"status": "cancelled"}
            except Exception as e:
                job.emit({"phase": "error", "msg": str(e)})
                raise
            job.emit({"phase": "done", **result})
            return {"status": "done", **result}

        def _finish(f: Future) -> None:
            self._futures.pop(tile_id, None)
            if on_done is None:
                return
            try:
                on_done(f.result())
            except Exception:
                # 잡이 죽어도 기록은 남겨야 한다 — 반쯤 만들어진 데이터셋이 있다
                on_done(None)

        future = self._get_executor().submit(_run)
        self._futures[tile_id] = future
        future.add_done_callback(_finish)

    def cancel(self, tile_id: str) -> bool:
        future = self._futures.get(tile_id)
        if future is None:
            return False
        if future.cancel():  # still queued — never started
            jobs.at(settings.jobs_dir, tile_id).emit({"phase": "cancelled"})
            return True
        jobs.at(settings.jobs_dir, tile_id).request_cancel()  # running: signal it
        return True

    def is_active(self, tile_id: str) -> bool:
        future = self._futures.get(tile_id)
        return future is not None and not future.done()


tile_manager = TileManager()


def reconcile_on_boot() -> None:
    """스레드 풀은 API 프로세스가 소유하므로 재시작하면 돌던 타일링은 죽는다.
    출처 기록이 `running` 인 채 남으면 영원히 만드는 중으로 보인다."""
    root = settings.projects_dir
    if not root.exists():
        return
    for sources_path in root.glob("*/datasets/*/sources.json"):
        dataset_id = sources_path.parent.name
        project_id = sources_path.parents[2].name
        for source in datasets.read_sources(project_id, dataset_id):
            if source.get("kind") != "tiling" or source.get("status") != "running":
                continue
            datasets.update_source(project_id, dataset_id, source["id"], status="error")
            jobs.at(settings.jobs_dir, dataset_id).ensure().emit(
                {"phase": "error", "msg": "Interrupted by a server restart"}
            )
