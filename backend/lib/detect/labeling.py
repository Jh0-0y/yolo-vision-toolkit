"""Ensemble labeling pipeline: N models over an image folder.

Writes labels/{stem}.txt for every processed image (empty file = negative).
`cfg.conf` is a hard filter (boxes below it are discarded), and each class can
be capped at a maximum box count per image. Review status is never set here —
"reviewed" means a human confirmed the data, which auto-labeling can't decide.

This module is the worker-process entrypoint — ultralytics/torch are imported
lazily so the API process never touches CUDA.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from lib.detect.ensemble import Detection, FusedBox, merge_detections
from lib.detect.tiled import TiledParams, collect, tiles_for
from lib.formats import IMAGE_EXTS
from lib.labels.io import atomic_write_text, write_box_meta, write_label_file
from lib.labels.registry import ClassRegistry, normalize

# Models run at this confidence floor so weighted-box fusion sees low-confidence
# candidates; the user's `cfg.conf` is then applied as a hard filter on the
# fused boxes (see run_labeling), not merely as a review boundary.
DETECT_FLOOR = 0.05

# Per-class cap for classes the user didn't specify — effectively unlimited
# (ultralytics' own default max_det is 300 total per image).
DEFAULT_MAX_PER_CLASS = 300

ProgressFn = Callable[[dict], None]
CancelFn = Callable[[], bool]


class JobCancelled(Exception):
    pass


@dataclass
class DetectorSpec:
    """모델 하나와 **그 모델의 추론 방식.**

    엔트리마다 방식이 다를 수 있다 — 선수 모델은 풀 프레임으로 크게 보고, 공 모델은
    타일로 잘게 보는 식이다. 어느 쪽이든 결과는 같은 `Detection` 목록이라 아래층은
    구분하지 않는다.
    """

    model_path: Path
    model_id: str
    mode: str = "full"  # full | tiled
    # 추론 해상도. **두 방식 모두 쓴다** — full 은 원본을, tiled 는 타일 하나를
    # 이 크기로 맞춰 넣는다. 타일보다 크게 잡으면 확대되어 작은 객체가 살아난다
    # (640 타일 안의 20px 공을 imgsz=1280 으로 보면 40px 이 된다).
    imgsz: int = 640
    tile_size: int = 640  # tiled 일 때 자르는 크기
    stride: int = 480
    merge_iou: float = 0.5  # 타일 경계 중복 병합 (모델 간 병합과 별개다)
    border_margin_px: int = 4


@dataclass
class LabelJobConfig:
    detectors: list[DetectorSpec]
    images_dir: Path
    out_dir: Path
    conf: float = 0.4
    iou_wbf: float = 0.55
    device: str = "cpu"  # 이미 해석된 장치 문자열 — "auto" 를 넘기지 않는다
    batch_size: int = 16
    # restrict to these file names (None = all images in images_dir)
    image_names: list[str] | None = None
    # class name -> max boxes per image (unspecified names use DEFAULT_MAX_PER_CLASS)
    max_boxes_per_class: dict[str, int] | None = None
    job_id: str | None = None


@dataclass
class LabelJobResult:
    total: int = 0
    labeled: int = 0
    boxes: int = 0
    stems: list[str] = field(default_factory=list)
    registry: dict = field(default_factory=dict)


def list_images(images_dir: Path) -> list[Path]:
    return sorted(
        p for p in images_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS and p.is_file()
    )


def _load_registry(out_dir: Path) -> ClassRegistry:
    """Seed from the existing classes.json so re-labeling never renumbers ids."""
    path = out_dir / "classes.json"
    if not path.exists():
        return ClassRegistry()
    try:
        return ClassRegistry.from_dict(json.loads(path.read_text()))
    except (json.JSONDecodeError, OSError, ValueError):
        return ClassRegistry()


def _cap_per_class(
    boxes: list[FusedBox], limit_by_cls: dict[int, int], default: int
) -> list[FusedBox]:
    """Keep only the top-N highest-scoring boxes of each class (N per class)."""
    counts: dict[int, int] = defaultdict(int)
    kept: list[FusedBox] = []
    for fb in sorted(boxes, key=lambda b: b.score, reverse=True):
        limit = limit_by_cls.get(fb.cls, default)
        if counts[fb.cls] < limit:
            counts[fb.cls] += 1
            kept.append(fb)
    return kept


def _detect_tiled(
    model,
    img_path: Path,
    spec: DetectorSpec,
    mapping: dict[int, int],
    device: str,
    floor: float,
    max_det: int,
) -> list[Detection]:
    """이미지 한 장을 타일로 잘라 **한 번의 배치**로 추론한다.

    좌표 규칙과 경계 처리는 `lib/detect/tiled.py` 에 있다 — 여기서는 자르고 모델에
    넣는 일만 한다. 그래서 그쪽은 모델 없이 테스트된다.
    """
    import cv2

    frame = cv2.imread(str(img_path))
    if frame is None:
        return []
    img_h, img_w = frame.shape[:2]
    params = TiledParams(
        tile_size=spec.tile_size,
        stride=spec.stride,
        merge_iou=spec.merge_iou,
        border_margin_px=spec.border_margin_px,
    )
    tiles = tiles_for(img_w, img_h, params)
    crops = [frame[ty : ty + spec.tile_size, tx : tx + spec.tile_size] for tx, ty in tiles]

    results = model.predict(
        crops,
        conf=floor,
        imgsz=spec.imgsz,
        device=device,
        max_det=max_det,
        verbose=False,
    )

    tile_boxes: list[tuple[int, int, tuple[float, float, float, float], int, float]] = []
    for (tx, ty), r in zip(tiles, results):
        if r.boxes is None:
            continue
        xyxy = r.boxes.xyxy.cpu().numpy()  # 타일 안 픽셀 좌표
        clss = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
        for k in range(len(clss)):
            tile_boxes.append(
                (tx, ty, tuple(float(v) for v in xyxy[k]), int(clss[k]), float(confs[k]))
            )
    return collect(tile_boxes, img_w, img_h, params, spec.model_id, mapping)


def run_labeling(
    cfg: LabelJobConfig,
    progress: ProgressFn | None = None,
    cancel_check: CancelFn | None = None,
) -> LabelJobResult:
    from ultralytics import YOLO

    device = cfg.device

    models: list[tuple[DetectorSpec, "YOLO", dict[int, int]]] = []
    registry = _load_registry(cfg.out_dir)
    for spec in cfg.detectors:
        model = YOLO(str(spec.model_path))
        mapping = registry.add_model(spec.model_id, dict(model.names))
        models.append((spec, model, mapping))

    class_sources = registry.class_sources()

    # Resolve per-class caps (keyed by class name) to global class ids. Names are
    # matched with the same normalization the registry uses for de-duplication.
    raw_limits = {normalize(k): int(v) for k, v in (cfg.max_boxes_per_class or {}).items()}
    limit_by_cls = {
        cid: raw_limits.get(normalize(name), DEFAULT_MAX_PER_CLASS)
        for cid, name in registry.names.items()
    }
    # Open the per-model NMS cap wide enough that a high per-class limit isn't
    # pre-empted by ultralytics' default (max_det=300, counted across all classes).
    max_det = max(DEFAULT_MAX_PER_CLASS, sum(limit_by_cls.values()) or DEFAULT_MAX_PER_CLASS)

    floor = min(cfg.conf, DETECT_FLOOR)

    images = list_images(cfg.images_dir)
    if cfg.image_names is not None:
        wanted = set(cfg.image_names)
        images = [p for p in images if p.name in wanted]
    result = LabelJobResult(total=len(images), registry=registry.to_dict())

    labels_dir = cfg.out_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(cfg.out_dir / "classes.json", json.dumps(registry.to_dict(), indent=2))

    done = 0
    for start in range(0, len(images), cfg.batch_size):
        if cancel_check and cancel_check():
            if progress:
                progress({"phase": "cancelled", "done": done, "total": result.total})
            raise JobCancelled(f"cancelled at {done}/{result.total}")
        batch = images[start : start + cfg.batch_size]
        # detections per image index in this batch
        per_image: list[list[Detection]] = [[] for _ in batch]

        for spec, model, mapping in models:
            if spec.mode == "tiled":
                for i, img_path in enumerate(batch):
                    per_image[i].extend(
                        _detect_tiled(model, img_path, spec, mapping, device, floor, max_det)
                    )
                continue

            # full — 배치로 한 번에. ultralytics 가 letterbox 로 줄여 추론하고
            # 박스를 **원본 정규화 좌표**로 되돌려 준다.
            results = model.predict(
                [str(p) for p in batch],
                conf=floor,
                imgsz=spec.imgsz,
                device=device,
                max_det=max_det,
                verbose=False,
            )
            for i, r in enumerate(results):
                if r.boxes is None:
                    continue
                xyxyn = r.boxes.xyxyn.cpu().numpy()
                clss = r.boxes.cls.cpu().numpy().astype(int)
                confs = r.boxes.conf.cpu().numpy()
                for k in range(len(clss)):
                    mapped = mapping.get(int(clss[k]))
                    if mapped is None:
                        continue
                    per_image[i].append(
                        Detection(
                            cls=mapped,
                            xyxy=tuple(float(v) for v in xyxyn[k]),  # type: ignore[arg-type]
                            score=float(confs[k]),
                            model_id=spec.model_id,
                        )
                    )

        for i, img_path in enumerate(batch):
            fused = merge_detections(per_image[i], class_sources, iou_thr=cfg.iou_wbf)
            # conf is a hard filter on the fused boxes; then cap each class at its
            # configured maximum, keeping the highest-scoring boxes.
            fused = [fb for fb in fused if fb.score >= cfg.conf]
            fused = _cap_per_class(fused, limit_by_cls, DEFAULT_MAX_PER_CLASS)
            label_file = labels_dir / f"{img_path.stem}.txt"
            write_label_file(label_file, [(fb.cls, fb.xyxy) for fb in fused])
            # score is recorded for display; review status is left to the user.
            write_box_meta(
                label_file,
                [{"score": fb.score, "status": None} for fb in fused],
            )
            result.labeled += 1
            result.boxes += len(fused)
            result.stems.append(img_path.stem)
            done += 1

        if progress:
            progress(
                {
                    "phase": "inference",
                    "done": done,
                    "total": result.total,
                    "labeled": result.labeled,
                    "boxes": result.boxes,
                }
            )

    if progress:
        progress({"phase": "done", "done": done, "total": result.total})
    return result
