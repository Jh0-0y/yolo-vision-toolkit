"""YOLO 객체 탐지 + ByteTrack 추적 (+ 공 전용 타일 추론).

두 가지 검출 구성을 지원한다:

  단일 모델   — Detector: 풀 프레임 `model.track`(ByteTrack)으로 공+선수 동시 탐지.
  분리 모델   — SplitDetector:
                 · 선수: Detector 풀 프레임 track (carrier 추적에 track_id 필요)
                 · 공:   TiledBallDetector — 프레임을 타일로 슬라이스해 batch 추론,
                        박스를 원본 좌표로 복원, 겹침 중복은 NMS로 병합.
                        track_id 없이 낸다 — 클립 플래너의 스티칭이 궤적을 조립한다.

타일로 학습한 공 모델은 타일 추론이 학습 분포와 일치한다 (풀 프레임 추론은 어긋남).

⚠️ 100ms 샘플링(10fps)에서 빠른 공은 ByteTrack IoU 연계에 실패해 검출이 떨어질
수 있다(실측 track 44% vs predict 57%) — 분리 모드의 공 predict 경로는 이 문제가 없다.

클래스 매핑은 model.names에서 자동 인식한다 — COCO(person/sports ball)와
커스텀 모델(ball/player) 모두 설정 없이 동작한다.
"""

from collections.abc import Iterator

import numpy as np

from app.domain.tiling import TilingParams, tile_grid

from .errors import ErrorCode, TrackCropError
from .types import Detection

OBJECT_TYPE_BALL = "ball"
OBJECT_TYPE_PLAYER = "player"


def _map_classes(names: dict[int, str]) -> dict[int, str]:
    """model.names → {class_id: object_type} 자동 매핑."""
    mapping: dict[int, str] = {}
    for class_id, name in names.items():
        lowered = name.lower()
        if "ball" in lowered:
            mapping[class_id] = OBJECT_TYPE_BALL
        elif "person" in lowered or "player" in lowered:
            mapping[class_id] = OBJECT_TYPE_PLAYER
    return mapping


def _load_model(model_path: str):
    # ultralytics·torch는 import에 수 초가 걸려 지연 로딩한다
    try:
        from ultralytics import YOLO

        return YOLO(model_path)
    except Exception as e:
        raise TrackCropError(
            ErrorCode.MODEL_LOAD_FAILED,
            f"탐지 모델을 로딩할 수 없습니다: {model_path}",
            details={"model_path": model_path, "cause": str(e)[:300]},
        ) from e


class Detector:
    def __init__(
        self,
        model_path: str,
        device: str,
        imgsz: int,
        conf: float,
        types: tuple[str, ...] = (OBJECT_TYPE_BALL, OBJECT_TYPE_PLAYER),
    ):
        self._model = _load_model(model_path)
        self._device = device
        self._imgsz = imgsz
        self._conf = conf
        self._type_by_class = {
            cid: t for cid, t in _map_classes(self._model.names).items() if t in types
        }
        for required in types:
            if required not in self._type_by_class.values():
                raise TrackCropError(
                    ErrorCode.MODEL_LOAD_FAILED,
                    f"모델에 {required} 클래스가 없습니다.",
                    details={"names": dict(self._model.names)},
                )

    def _reset_tracker(self) -> None:
        """Detector 재사용(여러 영상 처리) 시 이전 영상의 ByteTrack 상태가
        다음 영상으로 새지 않도록 track_id·Kalman 상태를 초기화한다."""
        predictor = getattr(self._model, "predictor", None)
        trackers = getattr(predictor, "trackers", None) if predictor is not None else None
        for tracker in trackers or []:
            reset = getattr(tracker, "reset", None)
            if callable(reset):
                reset()

    def track(
        self, frames: Iterator[tuple[int, np.ndarray]]
    ) -> Iterator[tuple[int, list[Detection]]]:
        """(offset_ms, Detection 목록)을 Sample 단위로 생성한다."""
        self._reset_tracker()  # 새 영상마다 tracker 상태 초기화
        for offset_ms, frame in frames:
            yield offset_ms, self.track_frame(frame, offset_ms)

    def track_frame(self, frame: np.ndarray, offset_ms: int) -> list[Detection]:
        """프레임 1장 track — ByteTrack persist=True로 프레임 간 상태 유지."""
        try:
            result = self._model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=list(self._type_by_class),
                imgsz=self._imgsz,
                conf=self._conf,
                device=self._device,
                verbose=False,
            )[0]
        except Exception as e:
            raise TrackCropError(
                ErrorCode.OBJECT_DETECTION_FAILED,
                "객체 탐지·추적 추론 중 오류가 발생했습니다.",
                details={"video_offset_ms": offset_ms, "cause": str(e)[:300]},
            ) from e
        return self._to_detections(result, offset_ms)

    def _to_detections(self, result, offset_ms: int) -> list[Detection]:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        xywh = boxes.xywh.tolist()  # (center_x, center_y, w, h)
        classes = boxes.cls.tolist()
        confs = boxes.conf.tolist()
        # ByteTrack track_id — 아직 track에 붙지 못한 Detection은 None
        ids = boxes.id.int().tolist() if boxes.id is not None else [None] * len(xywh)

        detections = []
        for (cx, cy, w, h), cls, conf, tid in zip(xywh, classes, confs, ids, strict=True):
            object_type = self._type_by_class.get(int(cls))
            if object_type is None:
                continue
            detections.append(
                Detection(
                    object_type=object_type,
                    track_id=int(tid) if tid is not None else None,  # ByteTrack 부여
                    bbox_x=cx - w / 2,
                    bbox_y=cy - h / 2,
                    bbox_width=w,
                    bbox_height=h,
                    confidence=conf,
                    video_offset_ms=offset_ms,
                )
            )
        return detections


# ---------------------------------------------------------------------------
# 공 전용 타일 추론


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """xyxy IoU."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def merge_tile_boxes(
    boxes: list[tuple[float, float, float, float, float]], iou_threshold: float
) -> list[tuple[float, float, float, float, float]]:
    """겹침 구간 중복 병합 — conf 내림차순 greedy NMS.

    boxes: [(x1, y1, x2, y2, conf)] 원본 좌표. 타일 경계의 반쪽 박스는 conf가
    낮게 나와, 겹침 덕에 온전히 담긴 쪽이 이긴다 (타일링 겹침 강제의 이유와 대칭).
    """
    kept: list[tuple[float, float, float, float, float]] = []
    for box in sorted(boxes, key=lambda b: b[4], reverse=True):
        if all(_iou(box[:4], k[:4]) < iou_threshold for k in kept):
            kept.append(box)
    return kept


class TiledBallDetector:
    """공 전용 타일 추론 — 슬라이스 → batch 추론 → 좌표 복원 → NMS 병합.

    track_id는 부여하지 않는다(None). 궤적 조립은 클립 플래너의 스티칭 담당.
    """

    def __init__(
        self,
        model_path: str,
        device: str,
        conf: float,
        tile_size: int = 640,
        stride: int = 480,
        merge_iou: float = 0.5,
    ):
        self._model = _load_model(model_path)
        self._device = device
        self._conf = conf
        self._grid = TilingParams(tile_size=tile_size, stride=stride)
        self._merge_iou = merge_iou
        self._ball_classes = [
            cid
            for cid, t in _map_classes(self._model.names).items()
            if t == OBJECT_TYPE_BALL
        ]
        if not self._ball_classes:
            raise TrackCropError(
                ErrorCode.MODEL_LOAD_FAILED,
                "공 모델에 ball 클래스가 없습니다.",
                details={"names": dict(self._model.names)},
            )

    def detect(self, frame: np.ndarray, offset_ms: int) -> list[Detection]:
        h, w = frame.shape[:2]
        tile = self._grid.tile_size
        grid = tile_grid(w, h, self._grid)
        tiles = [frame[ty : ty + tile, tx : tx + tile] for _, _, tx, ty in grid]

        try:
            results = self._model.predict(
                tiles,  # batch 추론 — GPU 한 번에
                classes=self._ball_classes,
                imgsz=tile,
                conf=self._conf,
                device=self._device,
                verbose=False,
            )
        except Exception as e:
            raise TrackCropError(
                ErrorCode.OBJECT_DETECTION_FAILED,
                "공 타일 추론 중 오류가 발생했습니다.",
                details={"video_offset_ms": offset_ms, "cause": str(e)[:300]},
            ) from e

        # 타일 좌표 → 원본 좌표 복원 (박스만 조립)
        restored: list[tuple[float, float, float, float, float]] = []
        for (_, _, tx, ty), result in zip(grid, results, strict=True):
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            for (x1, y1, x2, y2), conf in zip(
                boxes.xyxy.tolist(), boxes.conf.tolist(), strict=True
            ):
                restored.append((x1 + tx, y1 + ty, x2 + tx, y2 + ty, conf))

        return [
            Detection(
                object_type=OBJECT_TYPE_BALL,
                track_id=None,  # 스티칭이 궤적을 조립 — id 불필요
                bbox_x=x1,
                bbox_y=y1,
                bbox_width=x2 - x1,
                bbox_height=y2 - y1,
                confidence=conf,
                video_offset_ms=offset_ms,
            )
            for x1, y1, x2, y2, conf in merge_tile_boxes(restored, self._merge_iou)
        ]


class FullFrameBallDetector:
    """공 전용 풀 프레임 predict — 슬라이스 없이 원본 그대로 추론. track_id 없음."""

    def __init__(self, model_path: str, device: str, conf: float, imgsz: int = 1920):
        self._model = _load_model(model_path)
        self._device = device
        self._conf = conf
        self._imgsz = imgsz
        self._ball_classes = [
            cid
            for cid, t in _map_classes(self._model.names).items()
            if t == OBJECT_TYPE_BALL
        ]
        if not self._ball_classes:
            raise TrackCropError(
                ErrorCode.MODEL_LOAD_FAILED,
                "공 모델에 ball 클래스가 없습니다.",
                details={"names": dict(self._model.names)},
            )

    def detect(self, frame: np.ndarray, offset_ms: int) -> list[Detection]:
        try:
            result = self._model.predict(
                frame,
                classes=self._ball_classes,
                imgsz=self._imgsz,
                conf=self._conf,
                device=self._device,
                verbose=False,
            )[0]
        except Exception as e:
            raise TrackCropError(
                ErrorCode.OBJECT_DETECTION_FAILED,
                "공 풀 프레임 추론 중 오류가 발생했습니다.",
                details={"video_offset_ms": offset_ms, "cause": str(e)[:300]},
            ) from e
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        return [
            Detection(
                object_type=OBJECT_TYPE_BALL,
                track_id=None,
                bbox_x=x1,
                bbox_y=y1,
                bbox_width=x2 - x1,
                bbox_height=y2 - y1,
                confidence=conf,
                video_offset_ms=offset_ms,
            )
            for (x1, y1, x2, y2), conf in zip(
                boxes.xyxy.tolist(), boxes.conf.tolist(), strict=True
            )
        ]


# 서로 다른 공 검출기(모델)가 같은 공을 중복으로 잡았을 때 병합하는 IoU
_CROSS_MODEL_MERGE_IOU = 0.5


def _merge_ball_detections(balls: list[Detection]) -> list[Detection]:
    """여러 공 검출기의 출력을 합칠 때 모델 간 중복을 NMS로 정리한다."""
    if len(balls) <= 1:
        return balls
    boxes = [
        (d.bbox_x, d.bbox_y, d.bbox_x + d.bbox_width, d.bbox_y + d.bbox_height, d.confidence)
        for d in balls
    ]
    kept = set(merge_tile_boxes(boxes, _CROSS_MODEL_MERGE_IOU))
    return [d for d, b in zip(balls, boxes, strict=True) if b in kept]


class SplitDetector:
    """분리 검출 — 선수는 풀 프레임 track, 공은 별도 검출기 목록(타일/풀스캔).

    Detector와 같은 `.track(frames)` 인터페이스를 낸다.
    """

    def __init__(
        self, player: Detector, balls: list[TiledBallDetector | FullFrameBallDetector]
    ):
        self._player = player
        self._balls = balls

    def track(
        self, frames: Iterator[tuple[int, np.ndarray]]
    ) -> Iterator[tuple[int, list[Detection]]]:
        self._player._reset_tracker()
        for offset_ms, frame in frames:
            players = self._player.track_frame(frame, offset_ms)
            balls = [d for b in self._balls for d in b.detect(frame, offset_ms)]
            yield offset_ms, players + _merge_ball_detections(balls)


def build_detector(
    model_path: str,
    device: str,
    imgsz: int,
    conf: float,
    *,
    extras: list[dict] | None = None,
) -> Detector | SplitDetector:
    """검출기 팩토리.

    extras 없음 → 단일 모델(공+선수 풀 프레임 track).
    extras 있음 → 베이스 모델은 선수 전담(track), extras가 공을 전담한다.
      extras 항목: {"pt", "mode": "full"|"tiled", "conf"?, "imgsz"?,
                    "tile_size"?, "stride"?, "merge_iou"?}
    """
    if not extras:
        return Detector(model_path=model_path, device=device, imgsz=imgsz, conf=conf)

    balls: list[TiledBallDetector | FullFrameBallDetector] = []
    for e in extras:
        e_conf = e.get("conf")
        if e.get("mode") == "tiled":
            balls.append(
                TiledBallDetector(
                    model_path=e["pt"],
                    device=device,
                    conf=e_conf if e_conf is not None else conf,
                    tile_size=int(e.get("tile_size", 640)),
                    stride=int(e.get("stride", 480)),
                    merge_iou=float(e.get("merge_iou", 0.5)),
                )
            )
        else:
            balls.append(
                FullFrameBallDetector(
                    model_path=e["pt"],
                    device=device,
                    conf=e_conf if e_conf is not None else conf,
                    imgsz=int(e.get("imgsz", imgsz)),
                )
            )
    return SplitDetector(
        player=Detector(
            model_path=model_path,
            device=device,
            imgsz=imgsz,
            conf=conf,
            types=(OBJECT_TYPE_PLAYER,),
        ),
        balls=balls,
    )
