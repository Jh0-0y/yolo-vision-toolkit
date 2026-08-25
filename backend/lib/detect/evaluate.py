"""Prediction-vs-ground-truth matching + detection metrics.

Pure module (no torch, no DB): given one image's ground-truth boxes and a
model's predictions, greedily match by IoU (same class) to count TP/FP/FN, then
aggregate per-class Precision/Recall/F1 across images.

Reuses `ml.ensemble.iou`. Box coords are normalized xyxy in [0,1].
"""

from __future__ import annotations

import numpy as np

from lib.detect.ensemble import iou

# 채점 누적기는 예측 하나마다 한 줄씩 쌓여 이미지 수에 비례해 커진다. 파이썬 튜플로
# 들면 한 줄에 65 B 지만 배열이면 점수 4 B + 판정 1 B 다 — 그래서 이 모듈의 계산은
# 배열이 기본이고, 리스트를 받는 기존 진입점은 그 위의 얇은 어댑터로 남는다.
# (numpy 는 torch·ultralytics·cv2 와 달리 가볍고 계층 규칙의 금지 목록에도 없다.)


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


def match_for_ap_indexed(
    gt: list[dict], pred: list[dict], iou_thr: float
) -> list[tuple[int, float, int, int]]:
    """`match_for_ap` 과 같은 규칙이되, **어느 정답에 붙었는지와 몇 번째 예측인지**까지 돌려준다.

    마지막 원소는 호출자가 넘긴 `pred` 리스트에서의 색인이다. 이것이 없으면 호출자가
    같은 정렬을 다시 만들어 zip 해야 하는데, 그러면 여기 정렬 규칙이 조금만 바뀌어도
    점수·클래스·박스가 **조용히 서로 다른 예측에 붙는다.** 색인을 주면 그 결합이 사라진다.

    크기별 AP 의 COCO 규칙("다른 구간 정답에 붙은 예측은 무시")을 지키려면 참/거짓만으로는
    부족하다 — 그 예측이 *어느* 정답을 claim 했는지 알아야 구간 밖인지 판단할 수 있다.
    매칭을 한 번만 하고 크기별 AP·동작점 스냅샷·전체 AP 를 모두 여기서 파생시킨다.
    """
    gt_used = [False] * len(gt)
    rows: list[tuple[int, float, int, int]] = []
    for pi, p in sorted(enumerate(pred), key=lambda x: -x[1]["score"]):
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
        rows.append((p["cls"], p["score"], best_j, pi))
    return rows


SIZE_BUCKETS: tuple[str, ...] = ("small", "medium", "large")

# COCO 의 넓이 경계 — 픽셀 제곱이다.
_SMALL_MAX = 32 * 32
_MEDIUM_MAX = 96 * 96


def size_of(xyxy_n: list[float], img_w: int, img_h: int) -> str:
    """박스가 어느 크기 구간인가. **원본 프레임 픽셀**로 잰다.

    정규화 좌표로 재면 이미지 크기에 따라 같은 객체가 다른 구간이 되어, 타일 모델과
    풀 프레임 모델을 같은 잣대로 비교할 수 없다.
    """
    x1, y1, x2, y2 = xyxy_n
    area = max(0.0, (x2 - x1) * img_w) * max(0.0, (y2 - y1) * img_h)
    if area < _SMALL_MAX:
        return "small"
    if area < _MEDIUM_MAX:
        return "medium"
    return "large"


def _ranked(scores: np.ndarray, is_tp: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """점수 내림차순 랭킹에서의 (정렬된 점수, 누적 재현율, 누적 정밀도).

    AP·PR 곡선·F1 곡선이 전부 이 하나에서 나온다 — 같은 재료를 두 번 만들지 않는다.
    동점 처리는 `sorted(flags, key=lambda x: -x[0])` 과 같아야 하므로 안정 정렬을 쓴다.
    """
    order = np.argsort(-scores, kind="stable")
    tp = np.cumsum(is_tp[order], dtype=np.int64)
    rank = np.arange(1, tp.size + 1, dtype=np.int64)
    return scores[order], tp, rank


def _average_precision_arrays(scores: np.ndarray, is_tp: np.ndarray, n_gt: int) -> float:
    """AP for one class at one IoU threshold via 101-point interpolation (COCO).

    배열 핵심. 정밀도를 뒤에서부터 누적 최대로 만들면 그것이 곧 "재현율 r 이상에서의
    최고 정밀도" 포락선이고, 재현율은 단조 증가하므로 격자 101 점은 `searchsorted`
    한 번으로 찍힌다. 합은 **101 개를 순서대로** 더한다 — `np.sum` 의 쌍대합은 마지막
    자리가 흔들릴 수 있고, 이 값은 기존 테스트가 pin 하고 있다.
    """
    if n_gt <= 0 or scores.size == 0:
        return 0.0
    _, tp, rank = _ranked(scores, is_tp)
    rec = tp / n_gt
    prec = tp / rank
    envelope = np.maximum.accumulate(prec[::-1])[::-1]
    grid = np.arange(101, dtype=np.float64) / 100
    at = np.searchsorted(rec, grid, side="left")
    picked = np.where(at < rec.size, envelope[np.minimum(at, rec.size - 1)], 0.0)
    total = 0.0
    for p in picked.tolist():
        total += p
    return total / 101


def _as_flag_arrays(flags: list[tuple[float, bool]]) -> tuple[np.ndarray, np.ndarray]:
    """`(score, is_tp)` 리스트를 배열 둘로. **float64 를 유지한다** — 리스트 진입점은
    호출자가 준 정밀도 그대로 답해야 한다(워커는 애초에 float32 인 점수를 직접 넘긴다)."""
    n = len(flags)
    scores = np.fromiter((f[0] for f in flags), np.float64, n)
    is_tp = np.fromiter((f[1] for f in flags), bool, n)
    return scores, is_tp


def average_precision(flags: list[tuple[float, bool]], n_gt: int) -> float:
    """AP for one class at one IoU threshold via 101-point interpolation (COCO).

    ``flags`` are ``(score, is_tp)`` detections (any order) accumulated across all
    images of the class; ``n_gt`` is the class's total ground-truth count (> 0).

    리스트를 받는 기존 진입점 — `_average_precision_arrays` 를 부르는 어댑터다.
    """
    return _average_precision_arrays(*_as_flag_arrays(flags), n_gt)


def _thin(points: list, max_points: int, keep: tuple[int, ...] = ()) -> list:
    """점을 고르게 솎되 **양 끝과 `keep` 의 점은 반드시 남긴다.**

    `keep` 은 곡선의 최댓값처럼 "이 점이 사라지면 라벨과 그림이 어긋나는" 자리다.
    개수를 늘리지 않으려고 더하지 않고 **가장 가까운 표본을 그것으로 바꾼다.**
    """
    if max_points <= 2 or len(points) <= max_points:
        return points
    step = (len(points) - 1) / (max_points - 1)
    idx = {int(round(i * step)) for i in range(max_points)} | {0, len(points) - 1}
    ends = {0, len(points) - 1}
    for k in keep:
        if k in idx:
            continue
        inner = idx - ends
        if inner:
            idx.discard(min(inner, key=lambda j: abs(j - k)))
        idx.add(k)
    return [points[i] for i in sorted(idx)]


def curves_from_flags(
    flags: list[tuple[float, bool]], n_gt: int, max_points: int = 200
) -> dict:
    """PR 곡선과 F1–conf 곡선. `average_precision` 과 **같은 재료**를 쓴다.

    점수 내림차순으로 훑으며 누적 TP/FP 를 적으면 각 지점이 곧 "그 점수를 conf 로
    잡았을 때"의 정밀도·재현율이다. 그래서 곡선은 추론을 다시 돌리지 않고 나온다.

    정답이 없으면(`n_gt <= 0`) 곡선을 만들지 않는다 — 0 짜리 곡선은 "성능이 0"으로
    읽히지만 실제로는 "잴 것이 없다"이다.
    """
    return _curves_from_arrays(*_as_flag_arrays(flags), n_gt, max_points)


def _curves_from_arrays(
    scores: np.ndarray, is_tp: np.ndarray, n_gt: int, max_points: int = 200
) -> dict:
    """배열 핵심. 재현율·정밀도·F1 은 한 번에 벡터로 내고, 점을 담고 최고 F1 을 고르는
    일만 순서대로 훑는다.

    훑는 부분을 남긴 것은 게을러서가 아니다 — 최고점 판정이 `f1 > 직전 최고의 **반올림된**
    값` 이라 앞의 결과가 뒤의 판정을 바꾼다. 벡터로 옮기면 4 자리 경계에서 고르는 점이
    달라질 수 있어, 그 자리에서는 원래 규칙을 그대로 둔다. 반올림도 파이썬 `round` 를
    쓴다(`np.round` 는 10 의 거듭제곱을 곱했다 나누므로 마지막 자리가 갈릴 수 있다).
    """
    if n_gt <= 0 or scores.size == 0:
        return {"pr": [], "f1_conf": [], "best_f1": None}

    ranked, tp, rank = _ranked(scores, is_tp)
    rec = tp / n_gt
    prec = tp / rank
    denom = prec + rec
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(denom > 0, 2 * prec * rec / denom, 0.0)

    pr: list[list[float]] = []
    f1c: list[list[float]] = []
    best = {"value": -1.0, "conf": 0.0}
    best_i = -1
    for n, (score, r, p, f) in enumerate(
        zip(ranked.tolist(), rec.tolist(), prec.tolist(), f1.tolist())
    ):
        pr.append([round(r, 4), round(p, 4)])
        f1c.append([round(score, 4), round(f, 4)])
        if f > best["value"]:
            best = {"value": round(f, 4), "conf": round(score, 4)}
            best_i = n

    keep = (best_i,) if best_i >= 0 else ()
    return {
        "pr": _thin(pr, max_points, keep),
        "f1_conf": _thin(f1c, max_points, keep),
        "best_f1": best,
    }


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


def _map_from_arrays(
    scores: np.ndarray, classes: np.ndarray, correct: np.ndarray, gt_by_cls: dict[int, int]
) -> dict:
    """`map_from_accumulated` 과 같은 답을, 예측 하나가 한 줄인 배열에서 낸다.

    `correct` 는 `bool[N, len(IOU_THRESHOLDS)]` — 열 하나가 IoU 임계 하나다. 임계마다
    같은 예측을 따로 쌓지 않고 열을 골라 쓰므로, 누적기가 열 배로 붓지 않는다.
    """
    classes_with_gt = [c for c, n in gt_by_cls.items() if n > 0]
    per_class: dict[int, dict[str, float]] = {}
    map_per_iou: list[float] = []

    ap_by_cls: dict[int, list[float]] = {}
    for c in classes_with_gt:
        pick = classes == c
        s = scores[pick]
        hit = correct[pick]
        ap_by_cls[c] = [
            _average_precision_arrays(s, hit[:, j], gt_by_cls[c])
            for j in range(len(IOU_THRESHOLDS))
        ]

    for j in range(len(IOU_THRESHOLDS)):
        aps = [ap_by_cls[c][j] for c in classes_with_gt]
        map_per_iou.append(sum(aps) / len(aps) if aps else 0.0)

    for c in classes_with_gt:
        ap_over_iou = ap_by_cls[c]
        per_class[c] = {
            "ap50": round(ap_over_iou[0], 4),
            "ap": round(sum(ap_over_iou) / len(ap_over_iou), 4),
        }

    map50 = map_per_iou[0] if map_per_iou else 0.0
    mean_map = sum(map_per_iou) / len(map_per_iou) if map_per_iou else 0.0
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


def build_cls_map(
    model_names: dict[int, str], ds_by_norm: dict[str, int], other: int = -1
) -> dict[int, int]:
    """모델의 클래스 id → 데이터셋의 클래스 id. **모르는 것은 버리지 않고 `other` 로.**

    타일 추론(`lib/detect/tiled.py:collect`)은 이 매핑에 없는 클래스를 조용히 버린다.
    그런데 채점에서 데이터셋에 없는 클래스 예측은 **오검출로 세야** 한다 — 안 그러면
    어휘 밖 오검출만 사라져 그 모델이 실제보다 정확해 보인다. 그래서 모든 모델
    클래스를 빠짐없이 담고, 대조에 실패한 것은 `other`(기본 -1)로 보낸다.

    이름 대조는 `lib/labels/registry.normalize` 로 정규화해서 한다.
    """
    from lib.labels.registry import normalize

    return {
        int(cid): ds_by_norm.get(normalize(name), other)
        for cid, name in model_names.items()
    }


def match_any_class(
    gt: list[dict], pred: list[dict], iou_thr: float = 0.5
) -> tuple[list[tuple[int, int, float]], list[int], list[tuple[int, float]]]:
    """혼동행렬용 매칭 — **클래스가 달라도 짝짓는다.**

    "공을 선수로 봤다"를 보이는 것이 혼동행렬의 존재 이유인데, 같은 클래스끼리만 붙이는
    채점용 매칭으로는 그 사실이 만들어지지 않는다.

    점수를 함께 돌려주므로 호출자는 이 매칭을 **한 번만** 하고, 동작점마다
    `점수 ≥ conf` 로 거르면 된다.
    """
    gt_used = [False] * len(gt)
    matched: list[tuple[int, int, float]] = []
    spurious: list[tuple[int, float]] = []
    for p in sorted(pred, key=lambda x: -x["score"]):
        pb = tuple(p["xyxyn"])
        best_iou, best_j = iou_thr, -1
        for j, g in enumerate(gt):
            if gt_used[j]:
                continue
            v = iou(pb, tuple(g["xyxy_n"]))
            if v >= best_iou:
                best_iou, best_j = v, j
        if best_j >= 0:
            gt_used[best_j] = True
            matched.append((gt[best_j]["cls"], p["cls"], p["score"]))
        else:
            spurious.append((p["cls"], p["score"]))
    missed = [g["cls"] for j, g in enumerate(gt) if not gt_used[j]]
    return matched, missed, spurious


def confusion_at(
    conf: float,
    matched: list[tuple[int, int, float]],
    missed: list[int],
    spurious: list[tuple[int, float]],
    class_ids: list[int],
    names: dict[int, str],
) -> dict:
    """한 동작점의 혼동행렬. 행이 실제, 열이 예측이고 양쪽 끝에 `background` 가 붙는다.

    `conf` 아래로 잘린 짝은 **놓침으로 강등된다** — 예측이 사라졌을 뿐 정답은 그대로
    거기 있기 때문이다. 그래서 행 합(실제 개수)은 conf 와 무관하게 일정하다.

    `background × background` 는 뜻이 없어 비운다(`None`). 다만 **값이 있으면
    비우지 않는다** — 호출자가 예측에 나올 수 있는 클래스를 전부 `class_ids` 에
    넣으면 이 값은 늘 0 이지만, 계약을 어긴 경우 그 자리에 쌓인 헛것의 오검출을
    지워서는 안 된다.
    """
    m_gt = np.fromiter((m[0] for m in matched), np.int32, len(matched))
    m_pred = np.fromiter((m[1] for m in matched), np.int32, len(matched))
    m_score = np.fromiter((m[2] for m in matched), np.float64, len(matched))
    missed_arr = np.fromiter(missed, np.int32, len(missed))
    s_pred = np.fromiter((s[0] for s in spurious), np.int32, len(spurious))
    s_score = np.fromiter((s[1] for s in spurious), np.float64, len(spurious))
    return _confusion_from_arrays(
        conf, m_gt, m_pred, m_score, missed_arr, s_pred, s_score, class_ids, names
    )


def _id_lookup(class_ids: list[int], *arrays: np.ndarray) -> tuple[int, np.ndarray]:
    """클래스 id → `class_ids` 안의 자리를 찾는 표. 목록에 없는 id 는 -1 로 답한다.

    id 가 작은 정수(데이터셋 클래스 + 어휘 밖 -1)라서 표 하나로 훑을 수 있다 —
    수백만 줄을 파이썬 `dict.get` 으로 하나씩 뒤지지 않기 위한 것이다.
    """
    lows = [*class_ids, *(int(a.min()) for a in arrays if a.size)] or [0]
    highs = [*class_ids, *(int(a.max()) for a in arrays if a.size)] or [0]
    lo, hi = min(lows), max(highs)
    table = np.full(hi - lo + 1, -1, dtype=np.int64)
    for i, c in enumerate(class_ids):
        table[c - lo] = i
    return lo, table


def _confusion_from_arrays(
    conf: float,
    m_gt: np.ndarray,
    m_pred: np.ndarray,
    m_score: np.ndarray,
    missed: np.ndarray,
    s_pred: np.ndarray,
    s_score: np.ndarray,
    class_ids: list[int],
    names: dict[int, str],
) -> dict:
    """배열 핵심. 규칙은 `confusion_at` 의 docstring 그대로다.

    `conf` 는 **점수와 같은 정밀도로** 비교한다 — 점수를 float32 로 들고 있으면 4 자리로
    반올림된 값과 격자의 conf 가 딱 맞아떨어지는 자리에서 float64 로 재면 어긋난다.
    """
    n = len(class_ids)
    bg = n  # background 의 자리
    rows = np.zeros((n + 1, n + 1), dtype=np.int64)
    lo, table = _id_lookup(class_ids, m_gt, m_pred, missed, s_pred)

    if m_gt.size:
        r = table[m_gt - lo]
        c = table[m_pred - lo]
        c = np.where(c < 0, bg, c)  # 데이터셋에 없는 예측 클래스는 background 열로
        target = np.where(m_score >= m_score.dtype.type(conf), c, bg)  # 잘린 짝 → 놓침
        ok = r >= 0  # 데이터셋에 없는 정답 클래스는 있을 수 없다
        flat = np.bincount((r[ok] * (n + 1) + target[ok]), minlength=(n + 1) ** 2)
        rows += flat.reshape(n + 1, n + 1)
    if missed.size:
        r = table[missed - lo]
        rows[:, bg] += np.bincount(r[r >= 0], minlength=n + 1)
    if s_pred.size:
        keep = s_score >= s_score.dtype.type(conf)
        c = table[s_pred[keep] - lo]
        rows[bg] += np.bincount(np.where(c < 0, bg, c), minlength=n + 1)

    out = rows.tolist()
    # background × background 은 셀 수 있는 것이 아니라 비운다.
    # 다만 **값이 있으면 비우지 않는다** — 그 자리에 쌓였다는 것은 `class_ids` 에 없는
    # 클래스의 헛것이 있었다는 뜻이고, 그것을 지우면 오검출을 숨기는 셈이 된다.
    # (호출자가 예측 가능한 클래스를 전부 `class_ids` 에 넣으면 이 값은 늘 0 이다.)
    if not out[bg][bg]:
        out[bg][bg] = None
    return {"labels": [names.get(c, str(c)) for c in class_ids] + ["background"], "rows": out}


# 화면의 conf 슬라이더가 밟는 단계. 촘촘히 할수록 파일이 붓고, 성기면 최적점을 놓친다.
CONF_STEPS: tuple[float, ...] = tuple(round(0.05 * i, 2) for i in range(1, 20))


def counts_at(
    flags_by_cls: dict[int, list[tuple[float, bool]]],
    gt_by_cls: dict[int, int],
    conf: float,
) -> dict[int, dict[str, int]]:
    """한 동작점의 클래스별 TP/FP/FN. 결과는 그대로 `aggregate()` 에 넣는다.

    `flags_by_cls` 는 **호출자가 고른 매칭 IoU** 에서 누적한 `(score, is_tp)` 랭킹이다.
    mAP 를 만들 때 이미 쌓아 둔 것을 그대로 쓰므로 동작점마다 매칭을 다시 할 필요가 없다 —
    벤치마크는 그 런에 설정된 IoU 의 것을 넘겨, 대표 숫자와 같은 잣대가 되게 한다.

    FN 은 세지 않고 뺀다: 정답 수는 conf 와 무관하게 고정이므로 `정답 − TP` 가 곧 놓침이다.
    """
    total = sum(len(rows) for rows in flags_by_cls.values())
    scores = np.empty(total, np.float64)
    classes = np.empty(total, np.int32)
    is_tp = np.empty(total, bool)
    at = 0
    for cls, rows in flags_by_cls.items():
        for score, tp in rows:
            scores[at], classes[at], is_tp[at] = score, cls, tp
            at += 1
    out = _counts_from_arrays(scores, classes, is_tp, gt_by_cls, conf)
    # 줄이 하나도 없는 클래스 키도 기존 진입점은 표에 남겼다 — 어댑터가 그것까지 지킨다.
    for cls in flags_by_cls:
        out.setdefault(cls, {"tp": 0, "fp": 0, "fn": max(0, gt_by_cls.get(cls, 0))})
    return out


def _counts_from_arrays(
    scores: np.ndarray,
    classes: np.ndarray,
    is_tp: np.ndarray,
    gt_by_cls: dict[int, int],
    conf: float,
) -> dict[int, dict[str, int]]:
    """배열 핵심. 규칙은 `counts_at` 의 docstring 그대로다.

    클래스 id 는 작은 정수라 `bincount` 한 번으로 클래스별 TP·FP 가 동시에 나온다 —
    동작점이 스무 개라 클래스마다 전체를 다시 훑으면 그만큼 곱절이 된다.

    `conf` 는 **점수와 같은 정밀도로** 비교한다(`_confusion_from_arrays` 와 같은 이유).
    """
    counts: dict[int, tuple[int, int]] = {}
    if scores.size:
        keep = scores >= scores.dtype.type(conf)
        lo = int(classes.min())
        bins = int(classes.max()) - lo + 1
        shifted = (classes - lo).astype(np.intp)
        tp_counts = np.bincount(shifted[keep & is_tp], minlength=bins)
        fp_counts = np.bincount(shifted[keep & ~is_tp], minlength=bins)
        for cls in np.unique(classes).tolist():
            counts[int(cls)] = (int(tp_counts[cls - lo]), int(fp_counts[cls - lo]))

    out: dict[int, dict[str, int]] = {}
    for cls in set(counts) | set(gt_by_cls):
        tp, fp = counts.get(cls, (0, 0))
        out[cls] = {"tp": tp, "fp": fp, "fn": max(0, gt_by_cls.get(cls, 0) - tp)}
    return out
