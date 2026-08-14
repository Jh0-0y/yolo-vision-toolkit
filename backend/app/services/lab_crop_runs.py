"""연구실 크롭 런의 파일 규약 한 곳.

런 하나는 디렉터리 하나다. **상태는 어디에도 저장하지 않는다** —
`jobs/{crop_id}/progress.jsonl` 의 마지막 이벤트가 곧 상태다.

```
labs/{lab_id}/crops/{crop_id}/
├─ run.json      설정 스냅샷 (모델 목록 · 노브 · 비율 · 출력 옵션 · 원본 영상)
├─ crop.json     좌표
├─ wide.mp4      가로 영상 — 그리기 옵션이 있으면 오버레이 렌더본, 없으면 원본 사본
└─ crop.mp4      세로 크롭 영상
```

세 산출물은 **전부 필수**다. 선택이 아니다.

`wide.mp4` 는 원본과 완전히 분리된다 — 아카이브에서 영상을 지워도 런은 온전하다.
다만 그리기 옵션이 없을 때 원본을 통째로 복사하면 런마다 원본만 한 용량이 드므로
**하드링크로 만들고 실패하면 복사로 폴백**한다(같은 볼륨이면 링크가 되고, 원본을
지워도 링크는 살아 있어 의미는 그대로다).

TTL 은 없다. 지우는 것은 사용자뿐이다.
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import settings
from app.services import lab_store
from infra import jobs
from lib.crop.geometry import DEFAULT_CROP_H, DEFAULT_CROP_W
from lib.labels.io import atomic_write_text

META_NAME = "run.json"
CROP_NAME = "crop.json"
WIDE_NAME = "wide.mp4"
CUT_NAME = "crop.mp4"

# 그리기 도구 — 켜진 것이 하나라도 있으면 wide.mp4 를 렌더한다.
DRAW_TOGGLES = (
    "obj_boxes",  # 추론 박스
    "player_trails",  # 선수 궤적
    "ball_trail",  # 공 궤적
    "target_highlight",  # 타깃 (선택된 공·소유선수)
    "crop_box",  # 크롭 박스
    "dead_zone",  # 데드존
    "center_line",  # 중앙선
)


def new_id() -> str:
    return f"c_{uuid.uuid4().hex[:10]}"


def valid_id(crop_id: str) -> bool:
    """경로 조작 차단 — id 는 우리가 만든 `c_<hex>` 형태다."""
    return lab_store.valid_id(crop_id)


# ---------- 자리 ----------


def runs_dir(lab_id: str) -> Path:
    return lab_store.crops_dir(lab_id)


def run_dir(lab_id: str, crop_id: str) -> Path:
    return runs_dir(lab_id) / crop_id


def artifact(lab_id: str, crop_id: str, name: str) -> Path:
    return run_dir(lab_id, crop_id) / name


# ---------- 메타 ----------


def wants_overlay(toggles: dict) -> bool:
    """그리기 도구가 하나라도 켜져 있나. 하나도 없으면 wide.mp4 는 원본 그대로다."""
    return any(bool(toggles.get(k)) for k in DRAW_TOGGLES)


def create(lab_id: str, name: str, run_settings: dict) -> str:
    """잡을 던지기 **전에** 자리를 잡고 메타를 쓴다. 시작 시점에 쓰기 때문에
    실패한 시도도 목록에 남는다 — 어떤 설정이 실패했는지가 연구실에서는 값지다."""
    crop_id = new_id()
    meta = {
        "id": crop_id,
        "name": name or crop_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        # 런이 실제로 쓴 값의 스냅샷 — 연구실 프리셋을 나중에 바꿔도 이 기록은 안 흔들린다
        **run_settings,
    }
    run_dir(lab_id, crop_id).mkdir(parents=True, exist_ok=True)
    write_meta(lab_id, crop_id, meta)
    return crop_id


def write_meta(lab_id: str, crop_id: str, meta: dict) -> None:
    atomic_write_text(
        artifact(lab_id, crop_id, META_NAME), json.dumps(meta, ensure_ascii=False)
    )


def read_meta(lab_id: str, crop_id: str) -> dict | None:
    return _read_json(artifact(lab_id, crop_id, META_NAME))


def _read_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


# ---------- 원본 사본 ----------


def link_or_copy(src: Path, dst: Path) -> str:
    """원본을 런 안으로 들인다. 하드링크가 되면 링크, 안 되면 복사.

    돌려주는 값은 실제로 한 일(`"link"` · `"copy"`)이다 — 런 메타에 남겨 두면
    나중에 용량을 설명할 수 있다. 링크든 복사든 **원본을 지워도 런은 온전하다.**
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.unlink(missing_ok=True)
    try:
        os.link(src, dst)
        return "link"
    except OSError:
        # 다른 볼륨이거나 파일시스템이 하드링크를 안 받는다 — 용량을 쓰더라도 복사한다
        shutil.copy2(src, dst)
        return "copy"


# ---------- 상태 ----------


def status_of(crop_id: str) -> tuple[str, str | None]:
    """`(status, error)` — 진행률 파일이 유일한 근거다."""
    return jobs.at(settings.jobs_dir, crop_id).status()


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def wide_kind(lab_id: str, crop_id: str, meta: dict) -> str:
    """가로 영상이 **지금** 무엇인지 — `render` · `link` · `copy`. 저장하지 않고 파생한다.

    그리기 옵션이 하나라도 켜져 있었으면 렌더본이다. 아니면 원본 사본인데, 원본과
    자리를 나눠 쓰는 중인지는 링크 수(`st_nlink > 1`)가 이미 알고 있다.

    **만들어진 방식이 아니라 현재 상태다.** 하드링크로 만든 사본도 아카이브에서
    원본을 지우면 `copy` 로 읽힌다 — 실제로 그 시점부터 홀로 디스크를 차지하므로
    맞는 표현이고, 파일 자체는 온전하다.
    """
    wide = artifact(lab_id, crop_id, WIDE_NAME)
    if not wide.exists():
        return ""
    if wants_overlay(meta.get("toggles") or {}):
        return "render"
    try:
        return "link" if wide.stat().st_nlink > 1 else "copy"
    except OSError:
        return ""


def to_out(lab_id: str, crop_id: str, meta: dict) -> dict:
    """목록·상세 응답 한 건. 커버리지 요약은 crop.json 안에 이미 있으므로
    따로 저장하지 않고 여기서 읽는다 — 워커는 이 파일 규약을 몰라도 된다."""
    status, error = status_of(crop_id)
    crop = _read_json(artifact(lab_id, crop_id, CROP_NAME))
    summary = crop.get("summary") if crop else None
    # 실제로 쓰인 창은 crop.json 이 이미 들고 있다 — 요청값이 소스보다 커서 clamp
    # 됐는지가 여기서 드러난다. 따로 저장하지 않는다.
    applied = (crop or {}).get("crop") or {}
    wide = artifact(lab_id, crop_id, WIDE_NAME)
    cut = artifact(lab_id, crop_id, CUT_NAME)
    return {
        "id": crop_id,
        "name": meta.get("name") or crop_id,
        "created_at": meta.get("created_at", ""),
        "source_video_id": meta.get("source_video_id", ""),
        "source_name": meta.get("source_name", ""),
        "models": meta.get("models", []),
        "overrides": meta.get("overrides", {}),
        "crop_w": meta.get("crop_w", DEFAULT_CROP_W),
        "crop_h": meta.get("crop_h", DEFAULT_CROP_H),
        "applied_w": applied.get("width", 0),
        "applied_h": applied.get("height", 0),
        "toggles": meta.get("toggles", {}),
        "wide_kind": wide_kind(lab_id, crop_id, meta),  # render | link | copy
        "status": status,
        "error": error,
        "summary": summary if isinstance(summary, dict) else None,
        "has_crop_json": crop is not None,
        "has_wide": wide.exists(),
        "has_cut": cut.exists(),
        "size_bytes": _size(wide) + _size(cut) + _size(artifact(lab_id, crop_id, CROP_NAME)),
    }


def list_runs(lab_id: str) -> list[dict]:
    root = runs_dir(lab_id)
    if not root.exists():
        return []
    out: list[dict] = []
    for meta_path in root.glob(f"*/{META_NAME}"):
        meta = _read_json(meta_path)
        if meta is None:
            continue
        out.append(to_out(lab_id, meta_path.parent.name, meta))
    out.sort(key=lambda m: m["created_at"], reverse=True)
    return out


def delete(lab_id: str, crop_id: str) -> None:
    shutil.rmtree(run_dir(lab_id, crop_id), ignore_errors=True)
    shutil.rmtree(settings.jobs_dir / crop_id, ignore_errors=True)


def reconcile_on_boot() -> None:
    """워커 풀은 API 프로세스가 소유하므로 재시작하면 돌던 런은 죽는다.
    상태의 유일한 근거가 progress.jsonl 이니, 여기에 종료 이벤트를 남겨
    영원히 'running' 으로 보이는 런이 없게 한다."""
    root = settings.labs_dir
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
