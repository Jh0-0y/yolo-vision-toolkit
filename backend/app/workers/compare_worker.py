"""Model-comparison worker: score one or more detector entries against an
UPLOADED YOLO test set (images + labels + data.yaml), separately per entry
(a model id paired with its own inference mode — full frame or tiled), so
they can be compared. Runs in a child process (torch/ultralytics). Progress →
jobs_dir/{job_id}/progress.jsonl; result and images_manifest.json →
cfg["out_dir"] (the benchmark run's own directory, not the job directory).

Metrics per entry:
  - Precision / Recall / F1 + TP/FP/FN at the chosen display confidence
    (micro-averaged overall, and per class).
  - mAP@0.5 and mAP@0.5:0.95 + per-class AP, computed COCO-style over the FULL
    prediction set (every box down to the detector floor), independent of the
    display confidence.

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
from pathlib import Path

from infra import jobs
from lib.labels.io import atomic_write_text

# sentinel class id for predictions whose class isn't defined in the dataset;
# it never matches any GT box, so it is correctly counted as a false positive.
_OTHER = -1


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
    from app.core.config import resolve_device, settings
    from lib.detect.evaluate import (
        IOU_THRESHOLDS,
        accumulate,
        aggregate,
        build_cls_map,
        map_from_accumulated,
        match_for_ap,
        match_frame,
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
        gt_by_cls: dict[int, int] = {}  # identical across entries (same dataset)
        images: list[dict] = []
        manifest: dict[str, str] = {}  # image index -> absolute path (served by route)
        jobs.emit(progress, {"phase": "start", "total": len(pairs)})

        for i, (img_path, label_path) in enumerate(pairs):
            if job.cancelled():
                jobs.emit(progress, {"phase": "cancelled", "done": i, "total": len(pairs)})
                return {"status": "cancelled"}

            gt = [{"cls": cls, "xyxy_n": list(xyxy)} for cls, xyxy in read_label_file(label_path)]
            for g in gt:
                gt_by_cls[g["cls"]] = gt_by_cls.get(g["cls"], 0) + 1

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
                # display-confidence subset drives P/R/F1 counts + the overlay boxes
                preds_disp = [p for p in preds_full if p["score"] >= conf_thr]
                m = match_frame(gt, preds_disp, iou_match)
                accumulate(totals[eid], m["per_class"])
                det_counts[eid] += len(preds_disp)
                # full prediction set drives mAP (independent of display conf)
                for t in IOU_THRESHOLDS:
                    bucket = ap_acc[eid][t]
                    for cls, score, is_tp in match_for_ap(gt, preds_full, t):
                        bucket.setdefault(cls, []).append((score, is_tp))
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
            per_entry.append({
                "entry_id": eid,
                "model_id": e["model_id"],
                "name": e["name"],
                "mode": e["mode"],
                "overall": agg["overall"],
                "per_class": agg["per_class"],
                "detections": det_counts[eid],
                "map50": ap["map50"],
                "map": ap["map"],
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
