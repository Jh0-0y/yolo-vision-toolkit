"""연구실 파일 규약 한 곳 — 영상 아카이브와 크롭 런이 사는 자리.

연구실은 라벨도 클래스도 쓰지 않아 학습실 프로젝트와 나란한 최상위 자리를 갖는다.
이름·프리셋만 DB(`LabProject`)에 있고 나머지는 전부 파일이다.

```
labs/{lab_id}/
├─ lab.json                 사람이 읽는 표시(이름) — 진실은 DB 행이다
├─ videos/{video_id}.mp4    원본 + {video_id}.json 사이드카
└─ crops/{crop_id}/         크롭 런 (2단계)
```
"""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import settings
from lib.labels.io import atomic_write_text

VIDEO_META_SUFFIX = ".json"


def new_video_id() -> str:
    return f"vid_{uuid.uuid4().hex[:10]}"


def valid_id(value: str) -> bool:
    """경로 조작 차단 — id 는 우리가 만든 `<prefix>_<hex>` 형태다."""
    return bool(value) and "/" not in value and "\\" not in value and not value.startswith(".")


# ---------- 자리 ----------


def lab_dir(lab_id: str) -> Path:
    return settings.lab_dir(lab_id)


def videos_dir(lab_id: str) -> Path:
    return lab_dir(lab_id) / "videos"


def crops_dir(lab_id: str) -> Path:
    return lab_dir(lab_id) / "crops"


def ensure_dirs(lab_id: str) -> None:
    videos_dir(lab_id).mkdir(parents=True, exist_ok=True)
    crops_dir(lab_id).mkdir(parents=True, exist_ok=True)


def write_marker(lab_id: str, name: str) -> None:
    """`lab.json` — 디렉터리만 보고도 어느 연구실인지 알게 하는 표시."""
    atomic_write_text(
        lab_dir(lab_id) / "lab.json",
        json.dumps({"id": lab_id, "name": name}, ensure_ascii=False),
    )


def delete_lab(lab_id: str) -> None:
    shutil.rmtree(lab_dir(lab_id), ignore_errors=True)


# ---------- 영상 ----------


def video_meta_path(lab_id: str, video_id: str) -> Path:
    return videos_dir(lab_id) / f"{video_id}{VIDEO_META_SUFFIX}"


def read_video_meta(lab_id: str, video_id: str) -> dict | None:
    try:
        data = json.loads(video_meta_path(lab_id, video_id).read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def write_video_meta(lab_id: str, video_id: str, meta: dict) -> None:
    atomic_write_text(
        video_meta_path(lab_id, video_id), json.dumps(meta, ensure_ascii=False)
    )


def video_file(lab_id: str, video_id: str) -> Path | None:
    """원본 경로. 사이드카의 확장자를 쓴다 — 컨테이너가 여러 가지라 추측하지 않는다."""
    meta = read_video_meta(lab_id, video_id)
    if meta is None:
        return None
    path = videos_dir(lab_id) / f"{video_id}{meta.get('ext', '')}"
    return path if path.exists() else None


def probe_meta(path: Path) -> dict:
    """규격을 읽어 사이드카에 넣을 값으로 만든다. 열지 못하면 0 으로 둔다 —
    목록에 못 뜨는 것보다 낫고, 실제 실패는 크롭을 돌릴 때 드러난다."""
    from lib.video import probe as video_probe  # cv2 를 끌고 오므로 지연 import

    try:
        meta = video_probe(path)
    except (ValueError, OSError):
        return {"width": 0, "height": 0, "fps": 0.0, "duration_ms": 0}
    return {
        "width": meta.width,
        "height": meta.height,
        "fps": meta.fps,
        "duration_ms": meta.duration_ms,
    }


def new_video_meta(video_id: str, filename: str, ext: str, probed: dict) -> dict:
    return {
        "id": video_id,
        "name": filename,  # 이름변경 전까지는 파일명이 곧 이름이다
        "filename": filename,
        "ext": ext,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **probed,
    }


def list_videos(lab_id: str) -> list[dict]:
    root = videos_dir(lab_id)
    if not root.exists():
        return []
    out: list[dict] = []
    for meta_path in root.glob(f"*{VIDEO_META_SUFFIX}"):
        meta = read_video_meta(lab_id, meta_path.stem)
        if meta is None:
            continue
        path = root / f"{meta_path.stem}{meta.get('ext', '')}"
        meta["size_bytes"] = path.stat().st_size if path.exists() else 0
        out.append(meta)
    out.sort(key=lambda m: m.get("created_at", ""), reverse=True)
    return out


def delete_video(lab_id: str, video_id: str) -> None:
    """원본과 사이드카만 지운다. 크롭 런은 원본을 참조하지 않고 자기 사본을
    갖기 때문에 여기서 건드릴 것이 없다."""
    meta = read_video_meta(lab_id, video_id)
    if meta is not None:
        (videos_dir(lab_id) / f"{video_id}{meta.get('ext', '')}").unlink(missing_ok=True)
    video_meta_path(lab_id, video_id).unlink(missing_ok=True)


def count_runs_for_video(lab_id: str, video_id: str) -> int:
    """이 영상으로 돌린 크롭 런 수. 런 메타가 `source_video_id` 를 들고 있다."""
    root = crops_dir(lab_id)
    if not root.exists():
        return 0
    count = 0
    for run_meta in root.glob("*/run.json"):
        try:
            data = json.loads(run_meta.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and data.get("source_video_id") == video_id:
            count += 1
    return count


def count_runs(lab_id: str) -> int:
    root = crops_dir(lab_id)
    return len(list(root.glob("*/run.json"))) if root.exists() else 0
