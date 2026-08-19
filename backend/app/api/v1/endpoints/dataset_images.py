"""데이터셋 안의 이미지 — 목록 · 검수 · 삭제.

이미지는 한 방향으로 흐른다.

    미검수 → (검수) → 검수완료·미할당 → (비율 split) → train / val / test

그래서 `reviewed` 가 1급 필터고, `split` 은 검수완료 안의 갈래다. 나머지(labeled ·
cls · q · sort)는 그 안에서 좁히는 데 쓴다.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from app.core.config import settings
from app.db import get_session
from app.models import Project
from app.schemas.dataset import DatasetImagesOut, DeleteImagesIn, ReviewedIn
from app.services import datasets
from lib.formats import IMAGE_EXTS
from lib.labels.store import label_classes, label_path, read_boxes

router = APIRouter(prefix="/projects/{project_id}/datasets/{dataset_id}/images", tags=["datasets"])


def _require_dataset(session: Session, project_id: str, dataset_id: str) -> Path:
    if session.get(Project, project_id) is None:
        raise HTTPException(404, "Project not found")
    if not datasets.valid_id(dataset_id):
        raise HTTPException(422, "Invalid dataset id")
    if datasets.read_meta(project_id, dataset_id) is None:
        raise HTTPException(404, "Dataset not found")
    return datasets.dataset_dir(project_id, dataset_id)


def _raw_images(ddir: Path) -> list[Path]:
    raw = ddir / "raw"
    if not raw.exists():
        return []
    return [p for p in raw.iterdir() if p.suffix.lower() in IMAGE_EXTS and not p.name.startswith(".")]


def _labeled_stems(ddir: Path) -> set[str]:
    labels = ddir / "labels"
    if not labels.exists():
        return set()
    return {p.stem for p in labels.glob("*.txt")}


@router.get("", response_model=DatasetImagesOut)
def list_dataset_images(
    project_id: str,
    dataset_id: str,
    reviewed: bool | None = None,
    split: str | None = None,  # train | val | test | none(미할당)
    labeled: bool | None = None,
    cls: int | None = None,
    q: str = "",
    sort: str = "created",
    order: str = "desc",
    page: int = 1,
    size: int = 60,
    names_only: bool = False,
    session: Session = Depends(get_session),
):
    ddir = _require_dataset(session, project_id, dataset_id)
    if sort not in ("created", "name"):
        raise HTTPException(422, f"Unknown sort: {sort}")
    if split is not None and split not in (*datasets.SPLITS, "none"):
        raise HTTPException(422, f"Unknown split: {split}")

    labeled_stems = _labeled_stems(ddir)
    reviewed_stems = datasets.read_reviewed(project_id, dataset_id)
    splits = datasets.read_splits(project_id, dataset_id)

    images = _raw_images(ddir)
    if q:
        needle = q.lower()
        images = [p for p in images if needle in p.name.lower()]
    if reviewed is not None:
        images = [p for p in images if (p.stem in reviewed_stems) == reviewed]
    if split is not None:
        # 분할은 검수완료인 것만 인정한다 — 검수를 취소하면 미할당으로 보인다
        def _split_of(stem: str) -> str:
            return splits.get(stem, "none") if stem in reviewed_stems else "none"

        images = [p for p in images if _split_of(p.stem) == split]
    if labeled is not None:
        images = [p for p in images if (p.stem in labeled_stems) == labeled]
    if cls is not None:
        if cls == -1:  # "클래스 없음" — 라벨은 있는데 박스가 없는 네거티브
            images = [
                p for p in images
                if p.stem in labeled_stems and not label_classes(ddir, p.stem)
            ]
        else:
            images = [
                p for p in images
                if p.stem in labeled_stems and cls in label_classes(ddir, p.stem)
            ]

    reverse = order == "desc"
    if sort == "created":
        entries = sorted(
            ((p, p.stat().st_mtime) for p in images),
            key=lambda e: (e[1], e[0].name),
            reverse=reverse,
        )
    else:
        entries = sorted(
            ((p, p.stat().st_mtime) for p in images),
            key=lambda e: e[0].name.lower(),
            reverse=reverse,
        )

    total = len(entries)
    if names_only:
        return {"total": total, "names": [p.name for p, _ in entries]}

    base = f"{settings.api_prefix}/files/projects/{project_id}/datasets/{dataset_id}"
    start = (page - 1) * size
    items = [
        {
            "name": p.name,
            "stem": p.stem,
            "thumb": f"{base}/thumbs/{p.name}",
            "url": f"{base}/raw/{p.name}",
            "labeled": p.stem in labeled_stems,
            "reviewed": p.stem in reviewed_stems,
            "split": splits.get(p.stem) if p.stem in reviewed_stems else None,
            "boxes": read_boxes(ddir, p.stem) if p.stem in labeled_stems else [],
            "created_at": mtime,
        }
        for p, mtime in entries[start : start + size]
    ]
    return {"total": total, "page": page, "size": size, "items": items}


@router.put("/{stem}/reviewed")
def set_image_reviewed(
    project_id: str,
    dataset_id: str,
    stem: str,
    body: ReviewedIn,
    session: Session = Depends(get_session),
):
    """검수 표시를 켜고 끈다 — 이 데이터셋 안에서만 의미가 있다."""
    ddir = _require_dataset(session, project_id, dataset_id)
    if "/" in stem or "\\" in stem or stem.startswith("."):
        raise HTTPException(422, "Invalid filename")
    if not any(p.stem == stem for p in _raw_images(ddir)):
        raise HTTPException(404, "Image not found")

    stems = datasets.read_reviewed(project_id, dataset_id)
    if body.reviewed:
        stems.add(stem)
    else:
        stems.discard(stem)
    datasets.write_reviewed(project_id, dataset_id, stems)
    return {"ok": True, "reviewed": body.reviewed}


@router.delete("")
def delete_dataset_images(
    project_id: str,
    dataset_id: str,
    body: DeleteImagesIn,
    session: Session = Depends(get_session),
):
    """이미지와 거기 딸린 것을 전부 지운다 — 썸네일 · 라벨 · **박스 메타 사이드카** ·
    검수 표시 · 분할 배정."""
    ddir = _require_dataset(session, project_id, dataset_id)
    reviewed = datasets.read_reviewed(project_id, dataset_id)
    splits = datasets.read_splits(project_id, dataset_id)

    deleted = 0
    for name in body.names:
        base = name.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        if base.startswith(".") or Path(base).suffix.lower() not in IMAGE_EXTS:
            continue
        raw = ddir / "raw" / base
        if not raw.exists():
            continue
        stem = raw.stem
        raw.unlink()
        (ddir / "thumbs" / f"{stem}.jpg").unlink(missing_ok=True)
        label = label_path(ddir, stem)
        label.unlink(missing_ok=True)
        # 사이드카를 빼먹으면 이미지 없는 .meta.json 이 쌓인다 (실제로 1만 개가 쌓였다)
        label.with_suffix(".meta.json").unlink(missing_ok=True)
        reviewed.discard(stem)
        splits.pop(stem, None)
        deleted += 1

    datasets.write_reviewed(project_id, dataset_id, reviewed)
    datasets.write_splits(project_id, dataset_id, splits)
    return {"deleted": deleted}
