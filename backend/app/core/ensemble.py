"""Per-class weighted box fusion across N models, with source tracking.

Unlike the ensemble-boxes package, this implementation returns cluster
membership per fused box so decision rules can use agree_count (how many
distinct models contributed to a box).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Detection:
    cls: int  # global class id
    xyxy: tuple[float, float, float, float]  # normalized [0,1]
    score: float
    model_id: str


@dataclass
class FusedBox:
    cls: int
    xyxy: tuple[float, float, float, float]
    score: float
    sources: list[tuple[str, float]] = field(default_factory=list)  # (model_id, score)

    @property
    def agree_count(self) -> int:
        return len({m for m, _ in self.sources})


def iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _weighted_merge(members: list[Detection]) -> FusedBox:
    total = sum(d.score for d in members)
    coords = tuple(
        sum(d.xyxy[i] * d.score for d in members) / total for i in range(4)
    )
    # WBF-style score: mean of contributing scores
    score = total / len(members)
    return FusedBox(
        cls=members[0].cls,
        xyxy=coords,  # type: ignore[arg-type]
        score=score,
        sources=[(d.model_id, d.score) for d in members],
    )


def fuse_class(dets: list[Detection], iou_thr: float = 0.55) -> list[FusedBox]:
    """Greedy IoU clustering + score-weighted coordinate average for one class."""
    clusters: list[list[Detection]] = []
    fused: list[FusedBox] = []
    for det in sorted(dets, key=lambda d: d.score, reverse=True):
        placed = False
        for i, fb in enumerate(fused):
            if iou(det.xyxy, fb.xyxy) >= iou_thr:
                clusters[i].append(det)
                fused[i] = _weighted_merge(clusters[i])
                placed = True
                break
        if not placed:
            clusters.append([det])
            fused.append(_weighted_merge([det]))
    return fused


def merge_detections(
    dets: list[Detection],
    class_sources: dict[int, list[str]],
    iou_thr: float = 0.55,
) -> list[FusedBox]:
    """Merge one image's detections from all models.

    Classes owned by a single model pass through untouched (the model already
    applied NMS); classes owned by 2+ models go through weighted box fusion.
    """
    by_cls: dict[int, list[Detection]] = {}
    for d in dets:
        by_cls.setdefault(d.cls, []).append(d)

    out: list[FusedBox] = []
    for cls, cls_dets in by_cls.items():
        if len(class_sources.get(cls, [])) <= 1:
            out.extend(
                FusedBox(cls=d.cls, xyxy=d.xyxy, score=d.score, sources=[(d.model_id, d.score)])
                for d in cls_dets
            )
        else:
            out.extend(fuse_class(cls_dets, iou_thr=iou_thr))
    return out
