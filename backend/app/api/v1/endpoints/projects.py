"""Projects: CRUD, image ingest, unified image listing (filter/sort/search), stats."""

from __future__ import annotations

import json
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path

import yaml
from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from sqlmodel import Session, select

from app.core.config import settings
from app.db import get_session
from app.domain.tiling import TilingParams, clip_boxes_to_tile, tile_grid, tile_stem
from app.ml.labeling import IMAGE_EXTS
from app.models import ModelEntry, Project, TrainRun, iso_utc
from app.schemas.project import (
    DeleteImagesIn,
    ProjectCreate,
    ProjectOut,
    ReviewedIn,
    StatsOut,
)
from lib.labels.io import atomic_write_text, read_label_file, write_label_file
from lib.labels.registry import ClassRegistry
from lib.labels.store import (
    label_classes,
    label_path,
    read_boxes,
    read_reviewed,
    set_reviewed,
    write_reviewed,
)

router = APIRouter(prefix="/projects", tags=["projects"])

PROJECT_SUBDIRS = ("raw", "thumbs", "videos", "labels", "exports")


def _project_dir(project_id: str):
    return settings.projects_dir / project_id


def _get_project(session: Session, project_id: str) -> Project:
    project = session.get(Project, project_id)
    if project is None:
        raise HTTPException(404, "Project not found")
    return project


def _raw_images(pdir: Path) -> list[Path]:
    raw = pdir / "raw"
    if not raw.exists():
        return []
    return [p for p in raw.iterdir() if p.suffix.lower() in IMAGE_EXTS]


def _labeled_stems(pdir: Path) -> set[str]:
    labels = pdir / "labels"
    if not labels.exists():
        return set()
    return {p.stem for p in labels.glob("*.txt")}


@router.get("", response_model=list[ProjectOut])
def list_projects(session: Session = Depends(get_session)):
    projects = session.exec(select(Project).order_by(Project.created_at.desc())).all()
    return [ProjectOut(id=p.id, name=p.name, created_at=iso_utc(p.created_at)) for p in projects]


@router.post("", response_model=ProjectOut, status_code=201)
def create_project(req: ProjectCreate, session: Session = Depends(get_session)):
    project = Project(name=req.name)
    pdir = _project_dir(project.id)
    for sub in PROJECT_SUBDIRS:
        (pdir / sub).mkdir(parents=True, exist_ok=True)
    (pdir / "project.json").write_text(
        json.dumps({"id": project.id, "name": project.name}, ensure_ascii=False)
    )
    session.add(project)
    session.commit()
    return ProjectOut(id=project.id, name=project.name, created_at=iso_utc(project.created_at))


@router.delete("/{project_id}", status_code=204)
def delete_project(project_id: str, session: Session = Depends(get_session)):
    project = _get_project(session, project_id)
    # cascade: this project's scoped models & training runs. Their files live
    # under the project dir (removed by the rmtree below); here we drop the DB
    # rows and each run's progress log in the shared jobs pool.
    for run in session.exec(select(TrainRun).where(TrainRun.project_id == project_id)).all():
        shutil.rmtree(settings.jobs_dir / run.id, ignore_errors=True)
        session.delete(run)
    for model in session.exec(select(ModelEntry).where(ModelEntry.project_id == project_id)).all():
        session.delete(model)
    session.delete(project)
    session.commit()
    shutil.rmtree(_project_dir(project_id), ignore_errors=True)


@router.post("/{project_id}/images", status_code=201)
async def upload_images(
    project_id: str,
    files: list[UploadFile],
    session: Session = Depends(get_session),
):
    _get_project(session, project_id)
    raw_dir = _project_dir(project_id) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    added = 0
    skipped = 0
    for file in files:
        name = (file.filename or "").rsplit("/", 1)[-1]
        suffix = ("." + name.rsplit(".", 1)[-1].lower()) if "." in name else ""
        if suffix == ".zip":
            tmp = raw_dir / f".upload_{name}"
            with open(tmp, "wb") as f:
                while chunk := await file.read(1 << 20):
                    f.write(chunk)
            try:
                with zipfile.ZipFile(tmp) as zf:
                    for info in zf.infolist():
                        base = info.filename.rsplit("/", 1)[-1]
                        ext = ("." + base.rsplit(".", 1)[-1].lower()) if "." in base else ""
                        if info.is_dir() or ext not in IMAGE_EXTS or base.startswith("."):
                            skipped += 1
                            continue
                        with zf.open(info) as src, open(raw_dir / base, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                        added += 1
            except zipfile.BadZipFile:
                raise HTTPException(422, f"Corrupted zip file: {name}")
            finally:
                tmp.unlink(missing_ok=True)
        elif suffix in IMAGE_EXTS:
            with open(raw_dir / name, "wb") as f:
                while chunk := await file.read(1 << 20):
                    f.write(chunk)
            added += 1
        else:
            skipped += 1
    return {"added": added, "skipped": skipped}


def _read_yaml_names(extract: Path) -> dict[int, str]:
    """Read the class names from a YOLO dataset's data.yaml (list or dict form)."""
    yaml_path = None
    for candidate in ("data.yaml", "data.yml"):
        hits = sorted(extract.rglob(candidate))
        if hits:
            yaml_path = hits[0]
            break
    if yaml_path is None:
        raise HTTPException(422, "data.yaml not found in the zip (a YOLO-format dataset is required)")
    data = yaml.safe_load(yaml_path.read_text()) or {}
    raw = data.get("names")
    if isinstance(raw, dict):
        return {int(k): str(v) for k, v in raw.items()}
    if isinstance(raw, list):
        return {i: str(v) for i, v in enumerate(raw)}
    return {}


def _remap_label_text(path: Path, mapping: dict[int, int]) -> str:
    """Rewrite a YOLO label file's class ids through ``mapping`` (local -> project)."""
    lines: list[str] = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            old = int(parts[0])
        except ValueError:
            continue
        parts[0] = str(mapping.get(old, old))
        lines.append(" ".join(parts))
    return "\n".join(lines) + ("\n" if lines else "")


def _import_tiled(
    img: Path,
    lbl: Path | None,
    mapping: dict[int, int],
    raw_dir: Path,
    labels_dir: Path,
    tiling: TilingParams,
) -> tuple[int, int]:
    """이미지 한 장을 타일로 쪼개 저장한다. (저장한 타일 수, 라벨 쓴 타일 수) 반환.

    라벨은 자동 변환: 타일에 들어온 박스의 가시 비율이 min_visibility 이상이면
    타일 경계로 클립해 유지, 미만이면 삭제. drop_empty면 (라벨 있는 원본에서)
    박스가 하나도 안 남은 타일은 저장하지 않는다.
    """
    import cv2

    frame = cv2.imread(str(img))
    if frame is None:
        return 0, 0
    img_h, img_w = frame.shape[:2]

    boxes: list[tuple[int, tuple[float, float, float, float]]] = []
    if lbl is not None:
        boxes = [(mapping.get(c, c), xyxy) for c, xyxy in read_label_file(lbl)]

    saved = labeled = 0
    for col, row, tx, ty in tile_grid(img_w, img_h, tiling):
        tile_boxes = clip_boxes_to_tile(boxes, img_w, img_h, tx, ty, tiling)
        if lbl is not None and tiling.drop_empty and not tile_boxes:
            continue
        stem = tile_stem(img.stem, col, row)
        crop = frame[ty : ty + tiling.tile_size, tx : tx + tiling.tile_size]
        cv2.imwrite(str(raw_dir / f"{stem}.jpg"), crop)
        saved += 1
        if lbl is not None:
            write_label_file(labels_dir / f"{stem}.txt", tile_boxes)
            labeled += 1
    return saved, labeled


def _import_labeled_zip(tmp_zip: Path, pdir: Path, tiling: TilingParams | None = None) -> dict:
    """Import a YOLO-format dataset zip (images + labels + data.yaml) into a
    project's raw/ + labels/, merging its classes into classes.json and
    remapping label class ids so they stay valid project-wide.

    tiling이 주어지면 각 이미지를 타일로 쪼개 저장한다 (라벨 자동 클립/삭제)."""
    with tempfile.TemporaryDirectory() as td:
        extract = Path(td)
        try:
            with zipfile.ZipFile(tmp_zip) as zf:
                zf.extractall(extract)
        except zipfile.BadZipFile:
            raise HTTPException(422, "Corrupted zip file")

        names = _read_yaml_names(extract)

        classes_path = pdir / "classes.json"
        if classes_path.exists():
            registry = ClassRegistry.from_dict(json.loads(classes_path.read_text()))
        else:
            registry = ClassRegistry()
        mapping = registry.add_model("import", names) if names else {}

        # YOLO puts labels under a `labels/` directory; key them by image stem.
        label_files: dict[str, Path] = {}
        for p in extract.rglob("*.txt"):
            if any(part.lower() == "labels" for part in p.parts):
                label_files.setdefault(p.stem, p)

        raw_dir = pdir / "raw"
        labels_dir = pdir / "labels"
        raw_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        added_images = 0
        added_labels = 0
        for img in extract.rglob("*"):
            if img.suffix.lower() not in IMAGE_EXTS or img.name.startswith("."):
                continue
            lbl = label_files.get(img.stem)
            if tiling is not None:
                saved, labeled = _import_tiled(img, lbl, mapping, raw_dir, labels_dir, tiling)
                added_images += saved
                added_labels += labeled
                continue
            shutil.copyfile(img, raw_dir / img.name)
            added_images += 1
            if lbl is not None:
                atomic_write_text(labels_dir / f"{img.stem}.txt", _remap_label_text(lbl, mapping))
                added_labels += 1

        atomic_write_text(classes_path, json.dumps(registry.to_dict(), indent=2))
        return {
            "images": added_images,
            "labeled": added_labels,
            "classes": len(registry.classes),
        }


@router.post("/{project_id}/dataset-zip", status_code=201)
async def import_dataset_zip(
    project_id: str,
    file: UploadFile,
    tiling: bool = Form(False),  # 타일로 쪼개 학습 데이터셋 생성
    tile_size: int = Form(640),
    stride: int = Form(480),  # 겹침 = tile_size - stride
    min_visibility: float = Form(0.6),  # 60% 이상 보여야 라벨 유지 (프로젝트 기본)
    drop_empty: bool = Form(True),  # 박스 없는 타일 제외
    session: Session = Depends(get_session),
):
    """Ingest an already-labeled YOLO dataset zip so it shows up in the dataset view.

    tiling=True면 각 이미지를 타일(tile_size/stride)로 쪼개고 라벨을 자동
    변환한다 — 가시 비율 min_visibility 이상이면 클립 유지, 미만이면 삭제."""
    _get_project(session, project_id)
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(422, "Only .zip files are accepted")
    params: TilingParams | None = None
    if tiling:
        params = TilingParams(
            tile_size=tile_size, stride=stride,
            min_visibility=min_visibility, drop_empty=drop_empty,
        )
        errors = params.validate()
        if errors:
            raise HTTPException(422, f"Invalid tiling params: {errors}")
    pdir = _project_dir(project_id)
    pdir.mkdir(parents=True, exist_ok=True)
    tmp_zip = pdir / f".import_{uuid.uuid4().hex[:8]}.zip"
    try:
        with open(tmp_zip, "wb") as f:
            while chunk := await file.read(1 << 20):
                f.write(chunk)
        return await run_in_threadpool(_import_labeled_zip, tmp_zip, pdir, params)
    finally:
        tmp_zip.unlink(missing_ok=True)


@router.get("/{project_id}/images")
def list_images(
    project_id: str,
    labeled: bool | None = None,
    reviewed: bool | None = None,
    cls: int | None = None,
    sort: str = "created",
    order: str = "desc",
    q: str = "",
    page: int = 1,
    size: int = 60,
    names_only: bool = False,
    session: Session = Depends(get_session),
):
    _get_project(session, project_id)
    pdir = _project_dir(project_id)
    if sort not in ("created", "name"):
        raise HTTPException(422, f"Unknown sort: {sort}")

    labeled_stems = _labeled_stems(pdir)
    reviewed_stems = read_reviewed(pdir)

    images = _raw_images(pdir)
    if q:
        needle = q.lower()
        images = [p for p in images if needle in p.name.lower()]
    if labeled is not None:
        images = [p for p in images if (p.stem in labeled_stems) == labeled]
    if reviewed is not None:
        images = [p for p in images if (p.stem in reviewed_stems) == reviewed]
    if cls is not None:
        if cls == -1:  # "no classes": labeled but empty (negative)
            images = [
                p
                for p in images
                if p.stem in labeled_stems and not label_classes(pdir, p.stem)
            ]
        else:
            images = [
                p
                for p in images
                if p.stem in labeled_stems and cls in label_classes(pdir, p.stem)
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

    start = (page - 1) * size
    page_entries = entries[start : start + size]
    items = [
        {
            "name": p.name,
            "stem": p.stem,
            "thumb": f"{settings.api_prefix}/files/projects/{project_id}/thumbs/{p.name}",
            "url": f"{settings.api_prefix}/files/projects/{project_id}/raw/{p.name}",
            "labeled": p.stem in labeled_stems,
            "reviewed": p.stem in reviewed_stems,
            "boxes": read_boxes(pdir, p.stem) if p.stem in labeled_stems else [],
            "created_at": mtime,
        }
        for p, mtime in page_entries
    ]
    return {"total": total, "page": page, "size": size, "items": items}


@router.delete("/{project_id}/images")
def delete_images(
    project_id: str,
    body: DeleteImagesIn,
    session: Session = Depends(get_session),
):
    """Delete raw images and everything attached to them (thumb, label, reviewed flag)."""
    _get_project(session, project_id)
    pdir = _project_dir(project_id)
    reviewed = read_reviewed(pdir)
    deleted = 0
    for name in body.names:
        base = name.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        if base.startswith(".") or Path(base).suffix.lower() not in IMAGE_EXTS:
            continue
        raw = pdir / "raw" / base
        if not raw.exists():
            continue
        stem = raw.stem
        raw.unlink()
        (pdir / "thumbs" / f"{stem}.jpg").unlink(missing_ok=True)
        label_path(pdir, stem).unlink(missing_ok=True)
        reviewed.discard(stem)
        deleted += 1
    write_reviewed(pdir, reviewed)
    return {"deleted": deleted}


@router.put("/{project_id}/images/{stem}/reviewed")
def put_reviewed(
    project_id: str,
    stem: str,
    body: ReviewedIn,
    session: Session = Depends(get_session),
):
    _get_project(session, project_id)
    if "/" in stem or "\\" in stem or stem.startswith("."):
        raise HTTPException(422, "Invalid filename")
    pdir = _project_dir(project_id)
    if not any(p.stem == stem for p in _raw_images(pdir)):
        raise HTTPException(404, "Image not found")
    set_reviewed(pdir, stem, body.reviewed)
    return {"ok": True, "reviewed": body.reviewed}


@router.get("/{project_id}/stats", response_model=StatsOut)
def project_stats(project_id: str, session: Session = Depends(get_session)):
    _get_project(session, project_id)
    pdir = _project_dir(project_id)

    classes: list[dict] = []
    classes_path = pdir / "classes.json"
    if classes_path.exists():
        classes = json.loads(classes_path.read_text()).get("classes", [])

    stems = {p.stem for p in _raw_images(pdir)}
    labeled = stems & _labeled_stems(pdir)
    reviewed = stems & read_reviewed(pdir)
    return StatsOut(
        images=len(stems),
        labeled=len(labeled),
        reviewed=len(reviewed),
        classes=classes,
    )
