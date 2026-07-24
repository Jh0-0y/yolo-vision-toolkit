"""Prediction-vs-ground-truth matching + detection metrics.

Pure module (no torch, no DB): given one image's ground-truth boxes and a
model's predictions, greedily match by IoU (same class) to count TP/FP/FN, then
aggregate per-class Precision/Recall/F1 across images.

Reuses `ml.ensemble.iou`. Box coords are normalized xyxy in [0,1].
"""

from __future__ import annotations

from app.ml.ensemble import iou


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
