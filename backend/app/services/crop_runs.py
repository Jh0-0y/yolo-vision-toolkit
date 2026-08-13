"""크롭 랩 산출물(crop run)의 파일 규약 한 곳.

랩 결과는 DB 가 아니라 `projects/{project_id}/crops/{crop_id}/` 아래 파일로만
남는다. **상태는 어디에도 저장하지 않는다** — `jobs/{crop_id}/progress.jsonl` 의
마지막 이벤트가 곧 상태다. 그래서 목록이 곧 진행 현황이고, 브라우저에 무엇을
기억시켜 둘 필요가 없다.

시작하는 쪽(annotate 엔드포인트) · 읽는 쪽(crops 엔드포인트) · 치우는 쪽(sweep)이
전부 이 모듈만 본다. `crop_id` 는 잡 id 로도 그대로 쓴다.
"""

from __future__ import annotations

import json
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import settings
from infra import jobs
from lib.labels.io import atomic_write_text

# 렌더 영상은 곧 사라질 임시물이라 이 나이를 넘으면 지운다. run.json 과 crop.json
# 은 만료 대상이 아니다 — 목록에는 "영상 만료" 로 남는다.
VIDEO_TTL_SEC = 3600

META_NAME = "run.json"
CROP_NAME = "crop.json"
VIDEO_NAME = "out.mp4"
SOURCE_STEM = "source"


def new_id() -> str:
    return f"c_{uuid.uuid4().hex[:10]}"


def valid_id(crop_id: str) -> bool:
    """경로 조작 차단 — id 는 우리가 만든 `c_<hex>` 형태다."""
    return bool(crop_id) and "/" not in crop_id and "\\" not in crop_id and not crop_id.startswith(".")


def runs_dir(project_id: str) -> Path:
    return settings.projects_dir / project_id / "crops"


def run_dir(project_id: str, crop_id: str) -> Path:
    return runs_dir(project_id) / crop_id


# ---------- 메타 ----------


def create(project_id: str, kind: str, source_name: str, run_settings: dict) -> str:
    """잡을 던지기 **전에** 자리를 잡고 메타를 쓴다. 시작 시점에 쓰기 때문에
    실패한 시도도 목록에 남는다 — 어떤 설정이 실패했는지가 랩에서는 값지다."""
    crop_id = new_id()
    meta = {
        "id": crop_id,
        "name": source_name or crop_id,
        "kind": kind,  # "json" = 좌표만 | "cut" = 추론 없는 크롭 컷
        "source_name": source_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "settings": run_settings,
    }
    run_dir(project_id, crop_id).mkdir(parents=True, exist_ok=True)
    write_meta(project_id, crop_id, meta)
    return crop_id


def write_meta(project_id: str, crop_id: str, meta: dict) -> None:
    atomic_write_text(
        run_dir(project_id, crop_id) / META_NAME, json.dumps(meta, ensure_ascii=False)
    )


def read_meta(project_id: str, crop_id: str) -> dict | None:
    return _read_json(run_dir(project_id, crop_id) / META_NAME)


def _read_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


# ---------- 상태 ----------


def status_of(crop_id: str) -> tuple[str, str | None]:
    """`(status, error)` — 진행률 파일이 유일한 근거다."""
    return jobs.at(settings.jobs_dir, crop_id).status()


def to_out(project_id: str, crop_id: str, meta: dict) -> dict:
    """목록·상세 응답 한 건. 커버리지 요약은 crop.json 안에 이미 있으므로
    따로 저장하지 않고 여기서 읽는다 — 워커는 이 파일 규약을 몰라도 된다."""
    d = run_dir(project_id, crop_id)
    status, error = status_of(crop_id)
    crop = _read_json(d / CROP_NAME)
    summary = crop.get("summary") if crop else None
    return {
        "id": crop_id,
        "name": meta.get("name") or crop_id,
        "kind": meta.get("kind", "json"),
        "source_name": meta.get("source_name", ""),
        "created_at": meta.get("created_at", ""),
        "settings": meta.get("settings", {}),
        "status": status,
        "error": error,
        "summary": summary if isinstance(summary, dict) else None,
        "has_crop_json": crop is not None,
        "has_video": (d / VIDEO_NAME).exists(),
        "video_expired": bool(meta.get("video_expired")),
    }


def list_runs(project_id: str) -> list[dict]:
    root = runs_dir(project_id)
    out: list[dict] = []
    if not root.exists():
        return out
    for meta_path in root.glob(f"*/{META_NAME}"):
        meta = _read_json(meta_path)
        if meta is None:
            continue
        out.append(to_out(project_id, meta_path.parent.name, meta))
    out.sort(key=lambda m: m["created_at"], reverse=True)
    return out


def delete(project_id: str, crop_id: str) -> None:
    shutil.rmtree(run_dir(project_id, crop_id), ignore_errors=True)
    shutil.rmtree(settings.jobs_dir / crop_id, ignore_errors=True)


# ---------- 정리 ----------


def sweep_expired() -> None:
    """끝난 런의 원본 영상을 지우고, TTL 을 넘긴 렌더 영상을 만료시킨다.

    도는 중인 런은 건드리지 않는다 — 워커가 쓰고 있는 파일이다.
    """
    root = settings.projects_dir
    if not root.exists():
        return
    now = time.time()
    for d in root.glob(f"*/crops/*/{META_NAME}"):
        run = d.parent
        project_id = run.parent.parent.name
        crop_id = run.name
        status, _ = status_of(crop_id)
        if status == "running":
            continue
        # 원본은 잡이 끝나면 쓸모가 없다 — 가장 큰 파일이라 바로 지운다.
        for src in run.glob(f"{SOURCE_STEM}.*"):
            src.unlink(missing_ok=True)
        video = run / VIDEO_NAME
        if not video.exists():
            continue
        try:
            aged = now - video.stat().st_mtime > VIDEO_TTL_SEC
        except OSError:
            continue
        if not aged:
            continue
        video.unlink(missing_ok=True)
        meta = read_meta(project_id, crop_id)
        if meta is not None and not meta.get("video_expired"):
            meta["video_expired"] = True
            write_meta(project_id, crop_id, meta)


def reconcile_on_boot() -> None:
    """워커 풀은 API 프로세스가 소유하므로 재시작하면 돌던 런은 죽는다.
    상태의 유일한 근거가 progress.jsonl 이니, 여기에 종료 이벤트를 남겨
    영원히 'running' 으로 보이는 런이 없게 한다."""
    root = settings.projects_dir
    if not root.exists():
        return
    for d in root.glob(f"*/crops/*/{META_NAME}"):
        crop_id = d.parent.name
        status, _ = status_of(crop_id)
        if status != "running":
            continue
        jobs.at(settings.jobs_dir, crop_id).ensure().emit(
            {"phase": "error", "msg": "Interrupted by a server restart"}
        )
