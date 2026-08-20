"""벤치마크 런의 파일 규약 한 곳.

런 하나는 디렉터리 하나다. **상태는 어디에도 저장하지 않는다** —
`jobs/{bench_id}/progress.jsonl` 의 마지막 이벤트가 곧 상태다. 크롭 런
(`services/lab_crop_runs.py`)과 같은 길이고, 이어붙일 두 번째 진실을 만들지 않는다.

```
projects/{project_id}/benchmarks/{bench_id}/
├─ run.json               설정 스냅샷 — 잡을 **던지기 전에** 쓴다
├─ result.json            채점 결과
├─ images_manifest.json   오버레이 이미지 색인
└─ dataset/               펼친 test 트리 — 하드링크라 바이트가 늘지 않는다
```

`dataset/` 을 런과 함께 남기는 이유: 오버레이 이미지가 그 트리를 가리키므로, 남겨
두어야 과거 벤치마크를 열었을 때 박스를 계속 볼 수 있다. 하드링크라 원본 데이터셋을
나중에 고쳐도 이 런의 기록은 흔들리지 않는다 — 학습 런이 갖는 성질과 같다.

**TTL 은 없다.** 지우는 것은 사용자뿐이다.
"""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import settings
from infra import jobs
from lib.labels.io import atomic_write_text

META_NAME = "run.json"
RESULT_NAME = "result.json"
MANIFEST_NAME = "images_manifest.json"
DATASET_DIR = "dataset"


def new_id() -> str:
    return f"b_{uuid.uuid4().hex[:10]}"


def valid_id(bench_id: str) -> bool:
    """경로 조작 차단 — id 는 우리가 만든 `b_<hex>` 형태다."""
    return (
        bool(bench_id)
        and "/" not in bench_id
        and "\\" not in bench_id
        and not bench_id.startswith(".")
    )


# ---------- 자리 ----------


def runs_dir(project_id: str) -> Path:
    return settings.projects_dir / project_id / "benchmarks"


def run_dir(project_id: str, bench_id: str) -> Path:
    return runs_dir(project_id) / bench_id


def artifact(project_id: str, bench_id: str, name: str) -> Path:
    return run_dir(project_id, bench_id) / name


# ---------- 메타 ----------


def _read_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def read_meta(project_id: str, bench_id: str) -> dict | None:
    return _read_json(artifact(project_id, bench_id, META_NAME))


def write_meta(project_id: str, bench_id: str, meta: dict) -> None:
    atomic_write_text(
        artifact(project_id, bench_id, META_NAME),
        json.dumps(meta, ensure_ascii=False),
    )


def create(project_id: str, run_settings: dict) -> str:
    """잡을 던지기 **전에** 자리를 잡고 메타를 쓴다. 시작 시점에 쓰기 때문에
    실패한 시도도 목록에 남는다 — 어떤 설정이 실패했는지가 값지다."""
    bench_id = new_id()
    meta = {
        "id": bench_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **run_settings,
    }
    run_dir(project_id, bench_id).mkdir(parents=True, exist_ok=True)
    write_meta(project_id, bench_id, meta)
    return bench_id


# ---------- 상태 ----------


def status_of(bench_id: str) -> tuple[str, str | None]:
    """`(status, error)` — 진행률 파일이 유일한 근거다."""
    return jobs.at(settings.jobs_dir, bench_id).status()


def to_out(project_id: str, bench_id: str, meta: dict) -> dict:
    status, error = status_of(bench_id)
    return {
        "id": bench_id,
        "created_at": meta.get("created_at", ""),
        "dataset_name": meta.get("dataset_name", ""),
        "dataset": meta.get("dataset", ""),
        "entries": meta.get("entries", 0),
        "conf": meta.get("conf", 0.0),
        "iou": meta.get("iou", 0.0),
        "status": status,
        "error": error,
    }


def list_runs(project_id: str) -> list[dict]:
    root = runs_dir(project_id)
    if not root.exists():
        return []
    out: list[dict] = []
    for meta_path in root.glob(f"*/{META_NAME}"):
        meta = _read_json(meta_path)
        if meta is None:
            continue
        out.append(to_out(project_id, meta_path.parent.name, meta))
    out.sort(key=lambda m: m["created_at"], reverse=True)
    return out


def delete(project_id: str, bench_id: str) -> None:
    """런 하나를 통째로 지운다 — 산출물도 진행률도 이 안에만 있다."""
    shutil.rmtree(run_dir(project_id, bench_id), ignore_errors=True)
    shutil.rmtree(settings.jobs_dir / bench_id, ignore_errors=True)


def reconcile_on_boot() -> None:
    """워커 풀은 API 프로세스가 소유하므로 재시작하면 돌던 런은 죽는다.
    상태의 유일한 근거가 progress.jsonl 이니, 여기에 종료 이벤트를 남겨
    영원히 'running' 으로 보이는 런이 없게 한다."""
    root = settings.projects_dir
    if not root.exists():
        return
    for meta_path in root.glob(f"*/benchmarks/*/{META_NAME}"):
        bench_id = meta_path.parent.name
        status, _ = status_of(bench_id)
        if status != "running":
            continue
        jobs.at(settings.jobs_dir, bench_id).ensure().emit(
            {"phase": "error", "msg": "Interrupted by a server restart"}
        )
