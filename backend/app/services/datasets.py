"""데이터셋의 파일 규약 한 곳.

데이터셋 하나가 **자기 것을 전부 갖는다** — 이미지·라벨·클래스·검수 상태·분할.
데이터셋끼리는 아무것도 공유하지 않는다. 같은 영상으로 둘을 만들면 프레임을 두 번
뽑고, 라벨도 각각 그린다. 그 대신 서로를 신경 쓸 일이 없다.

```
projects/{project_id}/datasets/{dataset_id}/
├─ dataset.json     이름 · 생성일
├─ raw/ · thumbs/   이미지
├─ labels/          {stem}.txt (+ {stem}.meta.json — 박스별 score·status)
├─ classes.json     이 데이터셋만의 클래스
├─ reviewed.json    {stem: true}
└─ splits.json      {stem: "train"|"val"|"test"}
```

**수치는 저장하지 않는다.** 목록이 보여주는 미검수·검수완료·train/val/test 는 전부
파일에서 그때 세어 만든다 — 크롭 런이 상태를 progress.jsonl 에서 파생하는 것과 같다.
저장해 두면 반드시 실제와 어긋나는 순간이 온다.

이미지의 흐름은 한 방향이다.

    가져오기 → 미검수 → (검수) → 검수완료·미할당 → (비율 split) → train / val / test
"""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import settings
from lib.formats import IMAGE_EXTS
from lib.labels.io import atomic_write_text

META_NAME = "dataset.json"
CLASSES_NAME = "classes.json"
REVIEWED_NAME = "reviewed.json"
SPLITS_NAME = "splits.json"

SPLITS = ("train", "val", "test")


def new_id() -> str:
    return f"ds_{uuid.uuid4().hex[:10]}"


def valid_id(value: str) -> bool:
    """경로 조작 차단 — id 는 우리가 만든 `ds_<hex>` 형태다."""
    return bool(value) and "/" not in value and "\\" not in value and not value.startswith(".")


# ---------- 자리 ----------


def datasets_dir(project_id: str) -> Path:
    return settings.projects_dir / project_id / "datasets"


def dataset_dir(project_id: str, dataset_id: str) -> Path:
    return datasets_dir(project_id) / dataset_id


def raw_dir(project_id: str, dataset_id: str) -> Path:
    return dataset_dir(project_id, dataset_id) / "raw"


def thumbs_dir(project_id: str, dataset_id: str) -> Path:
    return dataset_dir(project_id, dataset_id) / "thumbs"


def labels_dir(project_id: str, dataset_id: str) -> Path:
    return dataset_dir(project_id, dataset_id) / "labels"


def ensure_dirs(project_id: str, dataset_id: str) -> None:
    for d in (
        raw_dir(project_id, dataset_id),
        thumbs_dir(project_id, dataset_id),
        labels_dir(project_id, dataset_id),
    ):
        d.mkdir(parents=True, exist_ok=True)


# ---------- 메타 ----------


def _read_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def read_meta(project_id: str, dataset_id: str) -> dict | None:
    return _read_json(dataset_dir(project_id, dataset_id) / META_NAME)


def write_meta(project_id: str, dataset_id: str, meta: dict) -> None:
    atomic_write_text(
        dataset_dir(project_id, dataset_id) / META_NAME,
        json.dumps(meta, ensure_ascii=False, indent=2),
    )


def create(project_id: str, name: str) -> str:
    dataset_id = new_id()
    ensure_dirs(project_id, dataset_id)
    write_meta(
        project_id,
        dataset_id,
        {
            "id": dataset_id,
            "name": name,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return dataset_id


def rename(project_id: str, dataset_id: str, name: str) -> dict:
    meta = read_meta(project_id, dataset_id) or {"id": dataset_id}
    meta["name"] = name
    write_meta(project_id, dataset_id, meta)
    return meta


def delete(project_id: str, dataset_id: str) -> None:
    """데이터셋 하나를 통째로 지운다 — 이미지도 라벨도 이 안에만 있다."""
    shutil.rmtree(dataset_dir(project_id, dataset_id), ignore_errors=True)


# ---------- 검수 · 분할 ----------


def read_reviewed(project_id: str, dataset_id: str) -> set[str]:
    data = _read_json(dataset_dir(project_id, dataset_id) / REVIEWED_NAME) or {}
    return {stem for stem, flag in data.items() if flag}


def write_reviewed(project_id: str, dataset_id: str, stems: set[str]) -> None:
    atomic_write_text(
        dataset_dir(project_id, dataset_id) / REVIEWED_NAME,
        json.dumps({stem: True for stem in sorted(stems)}, ensure_ascii=False, indent=2),
    )


def read_splits(project_id: str, dataset_id: str) -> dict[str, str]:
    """`{stem: "train"|"val"|"test"}`. 모르는 값은 버린다."""
    data = _read_json(dataset_dir(project_id, dataset_id) / SPLITS_NAME) or {}
    return {k: v for k, v in data.items() if v in SPLITS}


def write_splits(project_id: str, dataset_id: str, splits: dict[str, str]) -> None:
    atomic_write_text(
        dataset_dir(project_id, dataset_id) / SPLITS_NAME,
        json.dumps(dict(sorted(splits.items())), ensure_ascii=False, indent=2),
    )


# ---------- 수치 ----------


def image_stems(project_id: str, dataset_id: str) -> set[str]:
    raw = raw_dir(project_id, dataset_id)
    if not raw.exists():
        return set()
    return {p.stem for p in raw.iterdir() if p.suffix.lower() in IMAGE_EXTS}


def counts(project_id: str, dataset_id: str) -> dict:
    """목록·상단에 보이는 수치. **저장하지 않고 그때 센다.**

    분할은 검수완료인 것만 인정한다 — 검수를 취소하면 그 이미지는 분할에서도 빠진
    것으로 보인다(splits.json 을 굳이 고치러 다니지 않는다).
    """
    stems = image_stems(project_id, dataset_id)
    reviewed = stems & read_reviewed(project_id, dataset_id)
    splits = read_splits(project_id, dataset_id)

    by_split = {s: 0 for s in SPLITS}
    assigned = 0
    for stem in reviewed:
        split = splits.get(stem)
        if split in by_split:
            by_split[split] += 1
            assigned += 1

    return {
        "images": len(stems),
        "unreviewed": len(stems) - len(reviewed),
        "reviewed": len(reviewed),
        "unassigned": len(reviewed) - assigned,
        **by_split,
    }


def to_out(project_id: str, dataset_id: str, meta: dict) -> dict:
    return {
        "id": dataset_id,
        "name": meta.get("name") or dataset_id,
        "created_at": meta.get("created_at", ""),
        **counts(project_id, dataset_id),
    }


def list_datasets(project_id: str) -> list[dict]:
    root = datasets_dir(project_id)
    if not root.exists():
        return []
    out: list[dict] = []
    for meta_path in root.glob(f"*/{META_NAME}"):
        meta = _read_json(meta_path)
        if meta is None:
            continue
        out.append(to_out(project_id, meta_path.parent.name, meta))
    out.sort(key=lambda d: d["created_at"], reverse=True)
    return out
