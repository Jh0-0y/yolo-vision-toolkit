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
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from infra import jobs
from lib.labels.io import atomic_write_text

# sentinel class id for predictions whose class isn't defined in the dataset;
# it never matches any GT box, so it is correctly counted as a false positive.
_OTHER = -1


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
        accumulate,
        aggregate,
        average_precision,
        build_cls_map,
        confusion_at,
        counts_at,
        curves_from_flags,
        map_from_accumulated,
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
        # acc[entry_id][iou_thr][cls] -> list of (score, is_tp) for AP
        ap_acc: dict[str, dict[float, dict[int, list]]] = {
            e["entry_id"]: {t: {} for t in IOU_THRESHOLDS} for e in entries
        }
        # 크기별 AP — **IoU 0.5 하나만** 쌓는다. "타일이 작은 객체에서 이기는가" 는 AP@0.5 로
        # 답이 나는데, 임계 열 개를 다 쌓으면 큰 분할에서 엔트리마다 수백 MB 로 붓는다.
        # 붙은 예측은 **그 정답의 구간에만** 넣는다 — 다른 구간 정답에 붙은 것을 오검출로 세면
        # 그 구간 점수가 부당하게 나빠진다(COCO 규칙). 아무 정답에도 못 붙은 헛것은
        # **제 박스 크기 구간에만** 넣는다 — 세 구간에 다 넣으면 큰 헛것이 small 점수를
        # 끌어내려, 정작 재고 싶은 "작은 객체에서의 오검출"이 흐려진다.
        size_acc: dict[str, dict[str, dict[int, list]]] = {
            e["entry_id"]: {b: {} for b in SIZE_BUCKETS} for e in entries
        }
        # 정답은 엔트리와 무관하게 같은 데이터셋이라 한 벌만 센다.
        size_gt: dict[str, dict[int, int]] = {b: {} for b in SIZE_BUCKETS}
        # 혼동행렬 재료 — 가장 낮은 conf 로 한 번 짝지어 두고 동작점마다 거르기만 한다
        conf_acc: dict[str, dict[str, list]] = {
            e["entry_id"]: {"matched": [], "missed": [], "spurious": []} for e in entries
        }
        # 속도 — 이미지마다 추론 호출에 걸린 벽시계 시간(ms)
        timings: dict[str, list[float]] = {e["entry_id"]: [] for e in entries}
        # 동작점 스냅샷은 이 벤치마크의 매칭 IoU 로 세야 한다. 그 값이 mAP 스윕에 이미
        # 있으면(0.5 · 0.7 … 흔한 경우) ap_acc 를 그대로 쓰고, 없을 때만 따로 쌓는다.
        extra_acc: dict[str, dict[int, list]] = {e["entry_id"]: {} for e in entries}
        needs_extra = iou_match not in IOU_THRESHOLDS
        gt_by_cls: dict[int, int] = {}  # identical across entries (same dataset)
        images: list[dict] = []
        manifest: dict[str, str] = {}  # image index -> absolute path (served by route)
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

            manifest[str(i)] = str(img_path)
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
                # display-confidence subset drives P/R/F1 counts + the overlay boxes
                preds_disp = [p for p in preds_full if p["score"] >= conf_thr]
                m = match_frame(gt, preds_disp, iou_match)
                accumulate(totals[eid], m["per_class"])
                det_counts[eid] += len(preds_disp)
                # full prediction set drives mAP (independent of display conf)
                rows50: list = []
                for t in IOU_THRESHOLDS:
                    bucket = ap_acc[eid][t]
                    rows = match_for_ap_indexed(gt, preds_full, t)
                    for cls, score, gi in rows:
                        bucket.setdefault(cls, []).append((score, gi >= 0))
                    if t == 0.5:
                        rows50 = rows
                if needs_extra:
                    # 매칭 IoU 가 스윕 밖일 때만 한 번 더 짝짓는다
                    for cls, score, gi in match_for_ap_indexed(gt, preds_full, iou_match):
                        extra_acc[eid].setdefault(cls, []).append((score, gi >= 0))
                # 크기 구간별(IoU 0.5). `match_for_ap_indexed` 는 예측을 점수 내림차순으로
                # 훑으며 한 줄씩 쌓으므로, 같은 정렬로 zip 하면 행과 그 박스가 정확히 짝을 이룬다.
                for p, (cls, score, gi) in zip(
                    sorted(preds_full, key=lambda x: -x["score"]), rows50
                ):
                    b = gt[gi]["size"] if gi >= 0 else size_of(p["xyxyn"], img_w, img_h)
                    size_acc[eid][b].setdefault(cls, []).append((score, gi >= 0))
                # 혼동행렬 재료는 IoU 임계 하나(표시용 iou_match)에서 한 번만 짝지어 둔다
                matched_rows, missed_rows, spurious_rows = match_any_class(gt, preds_full, iou_match)
                conf_acc[eid]["matched"].extend(matched_rows)
                conf_acc[eid]["missed"].extend(missed_rows)
                conf_acc[eid]["spurious"].extend(spurious_rows)
                img_entry["per_entry"].append({
                    "entry_id": eid,
                    "pred_boxes": [
                        {"cls": p["cls"], "name": p["name"], "score": round(p["score"], 4), "xyxyn": p["xyxyn"]}
                        for p in preds_disp
                    ],
                })
            images.append(img_entry)
            if (i + 1) % 3 == 0 or i + 1 == len(pairs):
                jobs.emit(progress, {"phase": "analyze", "done": i + 1, "total": len(pairs)})

        per_entry = []
        for e in entries:
            eid = e["entry_id"]
            agg = aggregate(totals[eid], names)
            ap = map_from_accumulated(ap_acc[eid], gt_by_cls)
            for row in agg["per_class"]:
                cls_ap = ap["per_class"].get(row["cls"], {})
                row["ap50"] = cls_ap.get("ap50", 0.0)
                row["ap"] = cls_ap.get("ap", 0.0)

            flags50 = ap_acc[eid].get(0.5, {})
            # 곡선·AP 는 정답이 있어야 잴 수 있다. 정답 없는 클래스를 넣으면 gt_by_cls[c] 에서 죽는다.
            scored_ids = sorted(c for c, n in gt_by_cls.items() if n > 0)
            # 혼동행렬은 **예측에 나올 수 있는 클래스를 전부** 받아야 한다(confusion_at 의 계약).
            # 데이터셋에 없는 클래스(_OTHER)로 간 예측이 실제로 있었다면 그것도 열을 가져야
            # 오검출이 숨지 않는다.
            oov = _OTHER in flags50 or any(
                pc == _OTHER for _, pc, _ in conf_acc[eid]["matched"]
            ) or any(pc == _OTHER for pc, _ in conf_acc[eid]["spurious"])
            matrix_ids = scored_ids + ([_OTHER] if oov else [])

            by_size = {}
            for b in SIZE_BUCKETS:
                bucket_cls = [c for c, n in size_gt[b].items() if n > 0]
                if not bucket_cls:
                    continue  # 정답이 없는 구간은 비운다 — 0 으로 적으면 거짓말이 된다
                ap50_b = sum(
                    average_precision(size_acc[eid][b].get(c, []), size_gt[b][c])
                    for c in bucket_cls
                ) / len(bucket_cls)
                by_size[b] = {"ap50": round(ap50_b, 4), "gt": sum(size_gt[b].values())}

            pr_curves, f1_curves, best = [], [], None
            for c in scored_ids:
                cur = curves_from_flags(flags50.get(c, []), gt_by_cls[c])
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
            snap_flags = ap_acc[eid].get(iou_match)
            if snap_flags is None:
                snap_flags = extra_acc[eid]

            ops = []
            for step in CONF_STEPS:
                snap = aggregate(counts_at(snap_flags, gt_by_cls, step), names)
                ops.append({
                    "conf": step,
                    "overall": snap["overall"],
                    "per_class": snap["per_class"],
                    "confusion": confusion_at(
                        step, conf_acc[eid]["matched"], conf_acc[eid]["missed"],
                        conf_acc[eid]["spurious"], matrix_ids, names,
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
                        average_precision(ap_acc[eid].get(0.75, {}).get(c, []), gt_by_cls[c])
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
            "image_count": len(images),
            "conf": conf_thr,
            "iou": iou_match,
            "warning": warning,
        }
        atomic_write_text(out_dir / "result.json", json.dumps(result))
        jobs.emit(progress, {"phase": "done", "done": len(pairs), "total": len(pairs)})
        return {"status": "done", "images": len(images)}
    except Exception as e:
        jobs.emit(progress, {"phase": "error", "msg": str(e)})
        raise
