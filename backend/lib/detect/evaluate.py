"""Prediction-vs-ground-truth matching + detection metrics.

Pure module (no torch, no DB): given one image's ground-truth boxes and a
model's predictions, greedily match by IoU (same class) to count TP/FP/FN, then
aggregate per-class Precision/Recall/F1 across images.

Reuses `ml.ensemble.iou`. Box coords are normalized xyxy in [0,1].
"""

from __future__ import annotations

from lib.detect.ensemble import iou


def match_frame(gt: list[dict], pred: list[dict], iou_thr: float = 0.5) -> dict:
    """Match one image. `gt` items: {cls, xyxy_n}. `pred` items: {cls, xyxyn, score}.

    Greedy: predictions high→low score claim the best unused same-class GT with
    IoU ≥ thr (TP), else FP. Unclaimed GT are FN. Returns per-class + total tallies.
    """
    gt_used = [False] * len(gt)
    per_class: dict[int, dict[str, int]] = {}

    def bump(cls: int, key: str) -> None:
        per_class.setdefault(cls, {"tp": 0, "fp": 0, "fn": 0})[key] += 1

    for p in sorted(pred, key=lambda x: -x["score"]):
        pb = tuple(p["xyxyn"])
        best_iou, best_j = iou_thr, -1
        for j, g in enumerate(gt):
            if gt_used[j] or g["cls"] != p["cls"]:
                continue
            v = iou(pb, tuple(g["xyxy_n"]))
            if v >= best_iou:
                best_iou, best_j = v, j
        if best_j >= 0:
            gt_used[best_j] = True
            bump(p["cls"], "tp")
        else:
            bump(p["cls"], "fp")

    for j, g in enumerate(gt):
        if not gt_used[j]:
            bump(g["cls"], "fn")

    tp = sum(c["tp"] for c in per_class.values())
    fp = sum(c["fp"] for c in per_class.values())
    fn = sum(c["fn"] for c in per_class.values())
    return {"per_class": per_class, "tp": tp, "fp": fp, "fn": fn}


# COCO-style IoU sweep for mAP@0.5:0.95 (0.50, 0.55, …, 0.95).
IOU_THRESHOLDS = [round(0.5 + 0.05 * i, 2) for i in range(10)]


def match_for_ap(gt: list[dict], pred: list[dict], iou_thr: float) -> list[tuple[int, float, bool]]:
    """Greedy-match one image at one IoU threshold for AP accumulation.

    Same matching rule as ``match_frame`` (predictions high→low score claim the
    best unused same-class GT with IoU ≥ thr), but instead of counts it returns
    one ``(cls, score, is_tp)`` row per prediction. GT totals are counted by the
    caller (they don't depend on the IoU threshold).
    """
    gt_used = [False] * len(gt)
    rows: list[tuple[int, float, bool]] = []
    for p in sorted(pred, key=lambda x: -x["score"]):
        pb = tuple(p["xyxyn"])
        best_iou, best_j = iou_thr, -1
        for j, g in enumerate(gt):
            if gt_used[j] or g["cls"] != p["cls"]:
                continue
            v = iou(pb, tuple(g["xyxy_n"]))
            if v >= best_iou:
                best_iou, best_j = v, j
        if best_j >= 0:
            gt_used[best_j] = True
            rows.append((p["cls"], p["score"], True))
        else:
            rows.append((p["cls"], p["score"], False))
    return rows


def average_precision(flags: list[tuple[float, bool]], n_gt: int) -> float:
    """AP for one class at one IoU threshold via 101-point interpolation (COCO).

    ``flags`` are ``(score, is_tp)`` detections (any order) accumulated across all
    images of the class; ``n_gt`` is the class's total ground-truth count (> 0).
    """
    if n_gt <= 0:
        return 0.0
    tp = fp = 0
    points: list[tuple[float, float]] = []  # (recall, precision) as score decreases
    for _, is_tp in sorted(flags, key=lambda x: -x[0]):
        if is_tp:
            tp += 1
        else:
            fp += 1
        points.append((tp / n_gt, tp / (tp + fp)))
    # 101-point interpolation: for each recall level, best precision at recall ≥ r
    total = 0.0
    for i in range(101):
        r = i / 100
        p_max = 0.0
        for rec, prec in points:
            if rec >= r and prec > p_max:
                p_max = prec
        total += p_max
    return total / 101


def map_from_accumulated(
    acc: dict[float, dict[int, list[tuple[float, bool]]]], gt_by_cls: dict[int, int]
) -> dict:
    """Aggregate accumulated detections into mAP metrics.

    ``acc[iou_thr][cls]`` is the list of ``(score, is_tp)`` for that class at that
    IoU threshold, gathered over every image. Only classes with GT (> 0) count —
    predicted-but-undefined classes (e.g. the ``-1`` sentinel) have no GT and are
    excluded from the class mean, exactly like Ultralytics.

    Returns ``{map50, map, per_class: {cls: {ap50, ap}}}``.
    """
    classes = [c for c, n in gt_by_cls.items() if n > 0]
    per_class: dict[int, dict[str, float]] = {}
    map_per_iou: dict[float, float] = {}

    for t in IOU_THRESHOLDS:
        aps = {c: average_precision(acc.get(t, {}).get(c, []), gt_by_cls[c]) for c in classes}
        map_per_iou[t] = sum(aps.values()) / len(aps) if aps else 0.0

    for c in classes:
        ap_over_iou = [average_precision(acc.get(t, {}).get(c, []), gt_by_cls[c]) for t in IOU_THRESHOLDS]
        per_class[c] = {
            "ap50": round(ap_over_iou[0], 4),
            "ap": round(sum(ap_over_iou) / len(ap_over_iou), 4),
        }

    map50 = map_per_iou.get(0.5, 0.0)
    mean_map = sum(map_per_iou.values()) / len(map_per_iou) if map_per_iou else 0.0
    return {"map50": round(map50, 4), "map": round(mean_map, 4), "per_class": per_class}


def accumulate(total: dict[int, dict[str, int]], frame: dict[int, dict[str, int]]) -> None:
    """Merge one frame's per-class tallies into a running total (in place)."""
    for cls, c in frame.items():
        acc = total.setdefault(cls, {"tp": 0, "fp": 0, "fn": 0})
        acc["tp"] += c["tp"]
        acc["fp"] += c["fp"]
        acc["fn"] += c["fn"]


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return round(prec, 4), round(rec, 4), round(f1, 4)


def aggregate(total: dict[int, dict[str, int]], names: dict[int, str]) -> dict:
    """Per-class rows + overall (micro-averaged) metrics."""
    rows = []
    for cls in sorted(total):
        c = total[cls]
        prec, rec, f1 = _prf(c["tp"], c["fp"], c["fn"])
        rows.append({
            "cls": cls,
            "name": names.get(cls, str(cls)),
            "tp": c["tp"], "fp": c["fp"], "fn": c["fn"],
            "gt": c["tp"] + c["fn"], "pred": c["tp"] + c["fp"],
            "precision": prec, "recall": rec, "f1": f1,
        })
    tp = sum(c["tp"] for c in total.values())
    fp = sum(c["fp"] for c in total.values())
    fn = sum(c["fn"] for c in total.values())
    prec, rec, f1 = _prf(tp, fp, fn)
    overall = {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1}
    return {"per_class": rows, "overall": overall}
