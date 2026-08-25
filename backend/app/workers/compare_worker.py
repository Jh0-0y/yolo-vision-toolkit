"""Benchmark worker: score one or more detector entries against a dataset's
`test` split — the API materializes it (images + labels + data.yaml) into the
benchmark run's own `dataset/` directory before dispatching this job, and
passes that path as cfg["dataset_dir"]. Each entry is scored separately (a
model id paired with its own inference mode — full frame or tiled), so they
can be compared. Runs in a child process (torch/ultralytics). Progress →
jobs_dir/{job_id}/progress.jsonl; result and images_manifest.json →
cfg["out_dir"] (the benchmark run's own directory, not the job directory).

Metrics per entry:
  - Precision / Recall / F1 + TP/FP/FN at the chosen display confidence
    (micro-averaged overall, and per class).
  - mAP@0.5 and mAP@0.5:0.95 + per-class AP, computed COCO-style over the FULL
    prediction set (every box down to the detector floor), independent of the
    display confidence.
  - ap50 / ap75 and per-object-size AP@0.5 (`by_size`: small / medium / large
    by COCO area in ORIGINAL frame pixels; a bucket with no GT is omitted).
  - PR and F1-vs-confidence curves (`curves`) plus one snapshot per confidence
    step (`operating_points`: overall + per-class counts and a confusion matrix
    at each of `CONF_STEPS`) — all derived from the same accumulated ranking, so
    nothing is re-inferred.
  - Wall-clock inference speed (`speed`) and weight-file facts (`model`).

Full-frame predictions are mapped to the dataset's class ids by NAME
(normalized); tiled predictions are already mapped to dataset ids by
`lib/detect/tiled.py:collect` and must not be name-matched again. Either way,
a predicted class that isn't in the dataset is kept as a false positive
(id -1) rather than dropped. GT is read from the dataset's YOLO label files.

Overlay images are served by index via the benchmark images manifest — see
`images_manifest.json` next to result.json and the benchmark image route.
Scoring is exhaustive, but only the `OVERLAY_LIMIT` most-wrong images keep their
overlay boxes — `image_count` is the number SCORED, not the number kept.
"""

from __future__ import annotations

import heapq
import json
import time
from pathlib import Path

import numpy as np

from infra import jobs
from lib.labels.io import atomic_write_text

# sentinel class id for predictions whose class isn't defined in the dataset;
# it never matches any GT box, so it is correctly counted as a false positive.
_OTHER = -1

# result.json 에 오버레이를 남기는 이미지 수의 상한. 10만 장을 전부 실으면 파일이
# 기가 단위가 되고 화면이 그것을 통째로 받는데, 그만큼을 눈으로 넘겨볼 사람은 없다 —
# 오버레이 뷰어는 "왜 틀렸는지 몇 장 확인"하는 도구다. **채점은 전수로 한다.**
OVERLAY_LIMIT = 200


def _joined(parts: list[np.ndarray], dtype, width: int = 0) -> np.ndarray:
    """이미지마다 모아 둔 작은 배열을 한 번에 잇는다.

    numpy 는 append 가 비싸서 예측마다 배열을 늘릴 수 없다 — 이미지당 조각 하나를
    리스트에 담고 루프가 끝난 뒤 여기서 `concatenate` 한다(ultralytics 의 validator 와
    같은 방식). 조각이 하나도 없으면 모양이 맞는 빈 배열을 준다.
    """
    if parts:
        return np.concatenate(parts)
    return np.empty((0, width) if width else 0, dtype=dtype)


def _model_info(pt_path: str, model) -> dict | None:
    """가중치 파일 크기와 파라미터 수. 실패해도 채점을 막지 않는다 —
    이것 때문에 벤치마크 전체가 죽으면 손해가 이득보다 크다."""
    try:
        size = Path(pt_path).stat().st_size
    except OSError:
        size = None
    try:
        params = sum(p.numel() for p in model.model.parameters())
    except Exception:
        params = None
    if size is None and params is None:
        return None
    return {"size_bytes": size, "params": params}


def _predict_tiled(
    model, img_path, entry: dict, cls_map: dict[int, int], device: str, floor: float,
    names: dict[int, str],
) -> list[dict]:
    """타일로 잘라 추론하고 원본 좌표의 박스 목록으로 되돌린다.

    좌표 규칙과 경계 처리는 `lib/detect/tiled.py` 에 있다 — 여기서는 자르고 모델에
    넣는 일만 한다(`lib/detect/labeling.py:_detect_tiled` 와 같은 조합이다).

    `floor` 는 풀 프레임 경로가 쓰는 것과 **같은 conf 하한**이다 — 두 경로가 같은
    동작점에서 후보를 남겨야 점수를 나란히 놓을 수 있다.

    `names` 는 데이터셋의 클래스 이름(`_OTHER` → "(not in dataset)" 포함)이다.
    `collect()` 가 돌려주는 `Detection.cls` 는 이미 데이터셋 id 로 매핑되어 있어
    모델이 원래 뭐라고 불렀는지는 여기서 알 수 없다 — 그래서 표시 이름은 모델의
    원래 이름이 아니라 데이터셋 이름으로 채운다(풀 프레임 경로의 표시 이름과는
    출처가 다르지만, 오버레이가 빈 이름을 보여주는 것보다는 낫다).
    """
    from PIL import Image

    from lib.detect.tiled import TiledParams, collect, tiles_for

    params = TiledParams(
        tile_size=entry["tile_size"],
        stride=entry["stride"],
        merge_iou=entry["merge_iou"],
        border_margin_px=entry["border_margin_px"],
    )
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        img_w, img_h = im.size
        crops, offsets = [], []
        for tx, ty in tiles_for(img_w, img_h, params):
            crops.append(im.crop((tx, ty, tx + params.tile_size, ty + params.tile_size)))
            offsets.append((tx, ty))

    tile_boxes = []
    results = model.predict(
        crops, imgsz=entry["imgsz"], conf=floor, device=device, verbose=False
    )
    for (tx, ty), r in zip(offsets, results):
        if r.boxes is None:
            continue
        for b in r.boxes:
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
            tile_boxes.append((tx, ty, (x1, y1, x2, y2), int(b.cls.item()), float(b.conf.item())))

    dets = collect(tile_boxes, img_w, img_h, params, entry["entry_id"], cls_map)
    return [
        {"cls": d.cls, "name": names.get(d.cls, str(d.cls)), "score": d.score, "xyxyn": list(d.xyxy)}
        for d in dets
    ]


def _read_yaml_names(dataset_dir: Path) -> dict[int, str]:
    """Class names from the dataset's data.yaml (list or dict form). Pure — the
    API layer already verified a data.yaml exists before dispatching."""
    import yaml

    for candidate in ("data.yaml", "data.yml"):
        hits = sorted(dataset_dir.rglob(candidate))
        if hits:
            data = yaml.safe_load(hits[0].read_text()) or {}
            raw = data.get("names")
            if isinstance(raw, dict):
                return {int(k): str(v) for k, v in raw.items()}
            if isinstance(raw, list):
                return {i: str(v) for i, v in enumerate(raw)}
            return {}
    return {}


def _label_for(img: Path) -> Path | None:
    """The YOLO label path for an image: swap the last `images` segment for
    `labels` and the suffix for `.txt`. Returns None if that file is absent."""
    parts = list(img.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    else:
        return None
    label = Path(*parts).with_suffix(".txt")
    return label if label.exists() else None


def _gather_pairs(dataset_dir: Path, image_exts: set[str]) -> list[tuple[Path, Path]]:
    """(image, label) pairs for every labeled image under the dataset."""
    pairs: list[tuple[Path, Path]] = []
    for img in sorted(dataset_dir.rglob("*")):
        if img.suffix.lower() not in image_exts or not img.is_file():
            continue
        label = _label_for(img)
        if label is not None:
            pairs.append((img, label))
    return pairs


def run_compare(job_id: str, cfg: dict, jobs_dir: str) -> dict:
    from PIL import Image

    from app.core.config import resolve_device, settings
    from lib.detect.evaluate import (
        CONF_STEPS,
        IOU_THRESHOLDS,
        SIZE_BUCKETS,
        _average_precision_arrays,
        _confusion_from_arrays,
        _counts_from_arrays,
        _curves_from_arrays,
        _map_from_arrays,
        accumulate,
        aggregate,
        build_cls_map,
        match_any_class,
        match_for_ap_indexed,
        match_frame,
        size_of,
    )
    from lib.detect.labeling import DETECT_FLOOR
    from lib.detect.predict import PredictConfig, predict_image
    from lib.formats import IMAGE_EXTS
    from lib.labels.io import read_label_file
    from lib.labels.registry import normalize

    job = jobs.at(Path(jobs_dir), job_id)
    progress = job.progress_path

    dataset_dir = Path(cfg["dataset_dir"])
    conf_thr = float(cfg.get("conf", 0.4))
    iou_match = float(cfg.get("iou", 0.5))
    device = resolve_device(cfg.get("device"))

    names = _read_yaml_names(dataset_dir)
    names[_OTHER] = "(not in dataset)"
    ds_by_norm = {normalize(name): cid for cid, name in names.items() if cid != _OTHER}
    warning = None if ds_by_norm else "data.yaml has no class names — every prediction counts as a false positive."

    pairs = _gather_pairs(dataset_dir, set(IMAGE_EXTS))

    try:
        # cfg["out_dir"] 도 cfg["entries"] 처럼 잡의 것이 아니라 이 런의 입력이라
        # try 안에서 읽는다 — 밖에서 읽으면 키가 빠졌을 때 예외가 emit 도 없이
        # 새어 나가 progress.jsonl 이 영영 "running" 으로 보인다.
        out_dir = Path(cfg["out_dir"])

        if not pairs:
            raise RuntimeError(
                "No labeled images found in the dataset (expected images/ and labels/ with data.yaml)."
            )

        from ultralytics import YOLO

        entries = cfg["entries"]
        loaded: dict[str, object] = {}  # entry_id -> YOLO
        cls_maps: dict[str, dict[int, int]] = {}
        pcfgs: dict[str, PredictConfig] = {}  # entry_id -> PredictConfig (imgsz is per-entry)
        for e in entries:
            model = YOLO(e["pt"])
            try:
                model.to(device)
            except Exception:
                pass
            loaded[e["entry_id"]] = model
            # 모델이 아는 이름 → 데이터셋 id. 모르는 것은 -1 로 남겨 오검출로 센다.
            cls_maps[e["entry_id"]] = build_cls_map(
                {int(k): str(v) for k, v in (model.names or {}).items()}, ds_by_norm
            )
            pcfgs[e["entry_id"]] = PredictConfig(
                conf=conf_thr, iou_wbf=float(cfg.get("iou_wbf", 0.55)),
                imgsz=int(e["imgsz"]), device=device,
            )

        totals: dict[str, dict[int, dict[str, int]]] = {e["entry_id"]: {} for e in entries}
        det_counts: dict[str, int] = {e["entry_id"]: 0 for e in entries}
        # 채점 누적기 — 엔트리마다 **배열 넷**이고 예측 하나가 각 배열의 한 자리를 쓴다.
        #   scores  float32[N]   예측의 점수
        #   classes int32[N]     예측의 클래스(데이터셋 id, 어휘 밖은 _OTHER = -1)
        #   correct bool[N, 10]  IoU 임계 열 개 각각에서 맞았는가
        #   bucket  int8[N]      크기 구간(SIZE_BUCKETS 의 색인)
        # 임계마다 따로 쌓지 않고 `correct` 의 열로 두는 것이 핵심이다 — 파이썬 튜플로
        # 임계 열 개를 쌓으면 예측 하나가 650 B 를 먹는다.
        # 크기 구간도 **열 하나**면 된다: COCO 규칙상 예측 하나는 정확히 한 구간에만 속한다
        # (정답에 붙었으면 그 정답의 구간, 못 붙었으면 제 박스의 구간). 구간마다 따로
        # 쌓으면 같은 정보를 세 벌 드는 셈이다.
        # `extra` 는 매칭 IoU 가 mAP 스윕 밖일 때만 쓰는 열 하나다 — 스윕 안이면(0.5 · 0.7 …
        # 흔한 경우) `correct` 의 그 열을 그대로 쓴다.
        acc: dict[str, dict[str, list]] = {
            e["entry_id"]: {"scores": [], "classes": [], "correct": [], "bucket": [], "extra": []}
            for e in entries
        }
        bucket_index = {b: i for i, b in enumerate(SIZE_BUCKETS)}
        # 정답은 엔트리와 무관하게 같은 데이터셋이라 한 벌만 센다.
        size_gt: dict[str, dict[int, int]] = {b: {} for b in SIZE_BUCKETS}
        # 혼동행렬 재료 — 가장 낮은 conf 로 한 번 짝지어 두고 동작점마다 거르기만 한다
        conf_acc: dict[str, dict[str, list]] = {
            e["entry_id"]: {k: [] for k in ("m_gt", "m_pred", "m_score", "missed", "s_pred", "s_score")}
            for e in entries
        }
        # 속도 — 이미지마다 추론 호출에 걸린 벽시계 시간(ms)
        timings: dict[str, list[float]] = {e["entry_id"]: [] for e in entries}
        needs_extra = iou_match not in IOU_THRESHOLDS
        disp_thr32 = np.float32(conf_thr)  # 위 비교에 쓸 float32 임계값
        gt_by_cls: dict[int, int] = {}  # identical across entries (same dataset)
        # 오버레이 후보의 최소 힙 — (틀린 수, -원래 순번, 원래 순번, 이미지 경로, 오버레이)
        overlay_heap: list[tuple] = []
        scored = 0  # 채점한 전체 장수. 남긴 오버레이 수와 갈라진다.
        jobs.emit(progress, {"phase": "start", "total": len(pairs)})

        for i, (img_path, label_path) in enumerate(pairs):
            if job.cancelled():
                jobs.emit(progress, {"phase": "cancelled", "done": i, "total": len(pairs)})
                return {"status": "cancelled"}

            gt = [{"cls": cls, "xyxy_n": list(xyxy)} for cls, xyxy in read_label_file(label_path)]
            # 크기 구간은 **원본 프레임 픽셀**로 재야 해서 여기서 이미지 크기를 한 번 읽는다.
            with Image.open(img_path) as im:
                img_w, img_h = im.size
            for g in gt:
                gt_by_cls[g["cls"]] = gt_by_cls.get(g["cls"], 0) + 1
                g["size"] = size_of(g["xyxy_n"], img_w, img_h)
                size_gt[g["size"]][g["cls"]] = size_gt[g["size"]].get(g["cls"], 0) + 1

            wrong = 0  # 이 이미지에서 엔트리들이 틀린 수(fn + fp) — 오버레이를 고르는 잣대
            img_entry = {
                "stem": img_path.stem,
                "name": img_path.name,
                "url": (
                    f"{settings.api_prefix}/predict/benchmarks/{job_id}/images/{i}"
                    f"?project_id={cfg['project_id']}"
                ),
                "gt_boxes": [
                    {"cls": g["cls"], "name": names.get(g["cls"], str(g["cls"])), "xyxyn": g["xyxy_n"]}
                    for g in gt
                ],
                "per_entry": [],
            }
            for e in entries:
                eid = e["entry_id"]
                # 재는 것은 **추론 호출 전체**다 — 타일 엔트리는 자르기·타일마다의 추론·
                # 병합이 다 들어간다. 그것이 실제 배포 비용이기 때문이다. 이미지 읽기와
                # 채점은 밖에 있어 빠진다.
                t0 = time.perf_counter()
                if e["mode"] == "tiled":
                    boxes = _predict_tiled(loaded[eid], img_path, e, cls_maps[eid], device, DETECT_FLOOR, names)
                    preds_full = [
                        {"cls": b["cls"], "name": b["name"], "xyxyn": b["xyxyn"], "score": b["score"]}
                        for b in boxes
                    ]
                else:
                    res = predict_image([(eid, loaded[eid])], str(img_path), pcfgs[eid])
                    preds_full = []
                    for b in res["boxes"]:
                        cid = ds_by_norm.get(normalize(b["name"]))
                        if cid is None:
                            cid = _OTHER  # not a dataset class → false positive
                        preds_full.append(
                            {"cls": cid, "name": b["name"], "xyxyn": b["xyxyn"], "score": b["score"]}
                        )
                timings[eid].append((time.perf_counter() - t0) * 1000.0)
                # display-confidence subset drives P/R/F1 counts + the overlay boxes.
                # 비교는 **누적기와 같은 정밀도(float32)로** 한다 — 대표 숫자(overall)와
                # 같은 conf 의 동작점 스냅샷이 한 예측 차이로 갈리면 안 되기 때문이다.
                preds_disp = [p for p in preds_full if np.float32(p["score"]) >= disp_thr32]
                m = match_frame(gt, preds_disp, iou_match)
                accumulate(totals[eid], m["per_class"])
                det_counts[eid] += len(preds_disp)
                wrong += m["fn"] + m["fp"]
                # full prediction set drives mAP (independent of display conf)
                n_pred = len(preds_full)
                if n_pred:
                    # `match_for_ap_indexed` 는 늘 점수 내림차순으로 훑으므로 임계가 달라도
                    # **줄의 순서가 같다.** 그래서 임계 열 개를 한 배열의 열 열 개로 겹칠 수 있다.
                    # 박스는 정렬을 다시 만들어 zip 하지 않고 행이 실어 주는 예측 색인으로 집는다.
                    correct = np.empty((n_pred, len(IOU_THRESHOLDS)), dtype=bool)
                    rows50: list = []
                    for j, t in enumerate(IOU_THRESHOLDS):
                        rows = match_for_ap_indexed(gt, preds_full, t)
                        correct[:, j] = [gi >= 0 for _, _, gi, _ in rows]
                        if t == 0.5:
                            rows50 = rows
                    a = acc[eid]
                    # 점수·클래스는 **행에서 그대로** 꺼낸다. 예전에는 같은 정렬을 다시
                    # 만들어 zip 했는데, 그러면 매칭의 정렬 규칙이 조금만 바뀌어도
                    # 값이 서로 다른 예측에 조용히 붙는다.
                    a["scores"].append(np.fromiter((r[1] for r in rows50), np.float32, n_pred))
                    a["classes"].append(np.fromiter((r[0] for r in rows50), np.int32, n_pred))
                    a["correct"].append(correct)
                    # 붙은 예측은 **그 정답의 구간**, 못 붙은 헛것은 **제 박스의 구간**이다.
                    # 다른 구간 정답에 붙은 것을 오검출로 세면 그 구간 점수가 부당하게
                    # 나빠지고(COCO 규칙), 큰 헛것을 small 에도 넣으면 정작 재고 싶은
                    # "작은 객체에서의 오검출"이 흐려진다.
                    a["bucket"].append(np.fromiter(
                        (
                            bucket_index[
                                gt[gi]["size"] if gi >= 0 else size_of(p["xyxyn"], img_w, img_h)
                            ]
                            for _, _, gi, pi in rows50
                            for p in (preds_full[pi],)
                        ),
                        np.int8, n_pred,
                    ))
                    if needs_extra:
                        # 매칭 IoU 가 스윕 밖일 때만 한 번 더 짝짓는다
                        extra_rows = match_for_ap_indexed(gt, preds_full, iou_match)
                        a["extra"].append(
                            np.fromiter((gi >= 0 for _, _, gi, _ in extra_rows), bool, n_pred)
                        )
                # 혼동행렬 재료는 IoU 임계 하나(표시용 iou_match)에서 한 번만 짝지어 둔다
                matched_rows, missed_rows, spurious_rows = match_any_class(gt, preds_full, iou_match)
                ca = conf_acc[eid]
                if matched_rows:
                    n_m = len(matched_rows)
                    ca["m_gt"].append(np.fromiter((r[0] for r in matched_rows), np.int32, n_m))
                    ca["m_pred"].append(np.fromiter((r[1] for r in matched_rows), np.int32, n_m))
                    ca["m_score"].append(np.fromiter((r[2] for r in matched_rows), np.float32, n_m))
                if missed_rows:
                    ca["missed"].append(np.fromiter(missed_rows, np.int32, len(missed_rows)))
                if spurious_rows:
                    n_s = len(spurious_rows)
                    ca["s_pred"].append(np.fromiter((r[0] for r in spurious_rows), np.int32, n_s))
                    ca["s_score"].append(np.fromiter((r[1] for r in spurious_rows), np.float32, n_s))
                img_entry["per_entry"].append({
                    "entry_id": eid,
                    "pred_boxes": [
                        {"cls": p["cls"], "name": p["name"], "score": round(p["score"], 4), "xyxyn": p["xyxyn"]}
                        for p in preds_disp
                    ],
                })
            scored += 1
            # 채점은 전수로 끝났고, 여기서 **저장할 오버레이만** 고른다. 기준은 틀린 것이
            # 많은 순 — 디버깅할 때 실제로 보고 싶은 것이 그것이다. 크기 OVERLAY_LIMIT 의
            # 최소 힙을 유지하며 제일 덜 틀린 것을 그때그때 밀어낸다(전부 모아 두고 나중에
            # 고르면 메모리를 아끼는 뜻이 없다). 같은 수로 틀렸으면 앞선 이미지를 남긴다.
            heapq.heappush(overlay_heap, (wrong, -i, i, str(img_path), img_entry))
            if len(overlay_heap) > OVERLAY_LIMIT:
                heapq.heappop(overlay_heap)
            if (i + 1) % 3 == 0 or i + 1 == len(pairs):
                jobs.emit(progress, {"phase": "analyze", "done": i + 1, "total": len(pairs)})

        # 남긴 것은 **원래 순번 그대로** 늘어놓는다 — 색인이 어긋나면 오버레이 URL 이 어긋난다.
        kept = sorted(overlay_heap, key=lambda row: row[2])
        images = [row[4] for row in kept]
        manifest = {str(row[2]): row[3] for row in kept}  # image index -> absolute path

        per_entry = []
        for e in entries:
            eid = e["entry_id"]
            scores = _joined(acc[eid]["scores"], np.float32)
            classes = _joined(acc[eid]["classes"], np.int32)
            correct = _joined(acc[eid]["correct"], bool, len(IOU_THRESHOLDS))
            bucket = _joined(acc[eid]["bucket"], np.int8)
            m_gt = _joined(conf_acc[eid]["m_gt"], np.int32)
            m_pred = _joined(conf_acc[eid]["m_pred"], np.int32)
            m_score = _joined(conf_acc[eid]["m_score"], np.float32)
            missed = _joined(conf_acc[eid]["missed"], np.int32)
            s_pred = _joined(conf_acc[eid]["s_pred"], np.int32)
            s_score = _joined(conf_acc[eid]["s_score"], np.float32)

            agg = aggregate(totals[eid], names)
            ap = _map_from_arrays(scores, classes, correct, gt_by_cls)
            for row in agg["per_class"]:
                cls_ap = ap["per_class"].get(row["cls"], {})
                row["ap50"] = cls_ap.get("ap50", 0.0)
                row["ap"] = cls_ap.get("ap", 0.0)

            # 곡선·AP 는 정답이 있어야 잴 수 있다. 정답 없는 클래스를 넣으면 gt_by_cls[c] 에서 죽는다.
            scored_ids = sorted(c for c, n in gt_by_cls.items() if n > 0)
            # 혼동행렬은 **예측에 나올 수 있는 클래스를 전부** 받아야 한다(confusion_at 의 계약).
            # 데이터셋에 없는 클래스(_OTHER)로 간 예측이 실제로 있었다면 그것도 열을 가져야
            # 오검출이 숨지 않는다.
            oov = bool(
                np.any(classes == _OTHER)
                or np.any(m_pred == _OTHER)
                or np.any(s_pred == _OTHER)
            )
            matrix_ids = scored_ids + ([_OTHER] if oov else [])

            by_size = {}
            for b in SIZE_BUCKETS:
                bucket_cls = [c for c, n in size_gt[b].items() if n > 0]
                if not bucket_cls:
                    continue  # 정답이 없는 구간은 비운다 — 0 으로 적으면 거짓말이 된다
                in_bucket = bucket == bucket_index[b]
                ap50_b = sum(
                    _average_precision_arrays(
                        scores[in_bucket & (classes == c)],
                        correct[in_bucket & (classes == c), 0],
                        size_gt[b][c],
                    )
                    for c in bucket_cls
                ) / len(bucket_cls)
                by_size[b] = {"ap50": round(ap50_b, 4), "gt": sum(size_gt[b].values())}

            pr_curves, f1_curves, best = [], [], None
            for c in scored_ids:
                of_cls = classes == c
                cur = _curves_from_arrays(scores[of_cls], correct[of_cls, 0], gt_by_cls[c])
                if not cur["pr"]:
                    continue
                pr_curves.append({"cls": c, "name": names.get(c, str(c)), "points": cur["pr"]})
                f1_curves.append({"cls": c, "name": names.get(c, str(c)), "points": cur["f1_conf"]})
                # 어느 클래스의 최적점인지 함께 싣는다 — 다중 클래스에서 이름 없는 conf 하나만
                # 보여 주면 화면이 무엇을 최적화한 값인지 말할 수 없다.
                if best is None or cur["best_f1"]["value"] > best["value"]:
                    best = {**cur["best_f1"], "cls": c, "name": names.get(c, str(c))}

            # 스냅샷은 이 벤치마크가 고른 매칭 IoU 로 센다 — 대표 숫자(overall)와 같은 잣대여야
            # 슬라이더를 기본 동작점에 놓았을 때 두 숫자가 어긋나지 않는다. 곡선은 그대로
            # IoU 0.5 다(PR·F1 곡선은 관례가 0.5 이고 차트 라벨도 그렇게 적는다).
            if needs_extra:
                snap_hits = _joined(acc[eid]["extra"], bool)
            else:
                snap_hits = correct[:, IOU_THRESHOLDS.index(iou_match)]

            # 격자에 **이 런의 conf 를 반드시 넣는다.** 화면은 슬라이더를 런의 conf 에
            # 가장 가까운 단계에 놓는데, 정확히 일치하는 단계가 없으면 대표 숫자(overall)와
            # 슬라이더가 가리키는 값이 어긋난다. 지금은 생성 화면이 0.05 격자로 제한해
            # 우연히 맞지만, API 를 직접 부르면 임의의 conf 가 들어올 수 있다.
            steps = sorted({*CONF_STEPS, round(conf_thr, 4)})

            ops = []
            for step in steps:
                snap = aggregate(
                    _counts_from_arrays(scores, classes, snap_hits, gt_by_cls, step), names
                )
                ops.append({
                    "conf": step,
                    "overall": snap["overall"],
                    "per_class": snap["per_class"],
                    "confusion": _confusion_from_arrays(
                        step, m_gt, m_pred, m_score, missed, s_pred, s_score, matrix_ids, names,
                    ),
                })

            took = sorted(timings[eid][1:])  # 첫 장은 워밍업이라 버린다
            speed = None
            if took:
                med = took[len(took) // 2]
                speed = {
                    "ms_median": round(med, 2),
                    "ms_p95": round(took[min(len(took) - 1, int(len(took) * 0.95))], 2),
                    "fps": round(1000.0 / med, 2) if med > 0 else None,
                }

            per_entry.append({
                "entry_id": eid,
                "model_id": e["model_id"],
                "name": e["name"],
                "mode": e["mode"],
                # 크기까지 실어야 화면이 엔트리를 구분한다 — 같은 모델을 tile 640 과
                # tile 512 로 두 번 넣으면 mode 만으로는 두 카드가 똑같아 보인다.
                "imgsz": e["imgsz"],
                "tile_size": e["tile_size"],
                "overall": agg["overall"],
                "per_class": agg["per_class"],
                "detections": det_counts[eid],
                "map50": ap["map50"],
                "map": ap["map"],
                "ap50": ap["map50"],
                "ap75": round(
                    sum(
                        _average_precision_arrays(
                            scores[classes == c],
                            correct[classes == c, IOU_THRESHOLDS.index(0.75)],
                            gt_by_cls[c],
                        )
                        for c in scored_ids
                    ) / len(scored_ids), 4,
                ) if scored_ids else None,
                "by_size": by_size,
                "curves": {"pr": pr_curves, "f1_conf": f1_curves, "best_f1": best},
                "operating_points": ops,
                "speed": speed,
                "model": _model_info(e["pt"], loaded[eid]),
            })

        out_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(out_dir / "images_manifest.json", json.dumps(manifest))
        result = {
            "per_entry": per_entry,
            "images": images,
            # 채점한 **전체** 장수다. `images` 는 그중 남긴 표본이라 수가 갈라진다.
            "image_count": scored,
            # 무엇을 기준으로 골랐는지 화면이 사용자에게 설명할 수 있게 함께 싣는다.
            "overlay_selection": {
                "criterion": "most_errors",
                "limit": OVERLAY_LIMIT,
                "kept": len(images),
            },
            "conf": conf_thr,
            "iou": iou_match,
            "warning": warning,
        }
        atomic_write_text(out_dir / "result.json", json.dumps(result))
        jobs.emit(progress, {"phase": "done", "done": len(pairs), "total": len(pairs)})
        return {"status": "done", "images": scored}
    except Exception as e:
        jobs.emit(progress, {"phase": "error", "msg": str(e)})
        raise
