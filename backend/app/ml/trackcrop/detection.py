"""YOLO 객체 탐지 + 자체 track_id 부여.

탐지는 predict로 수행하고 track_id는 프레임 간 최근접 매칭으로 자체 부여한다.
ByteTrack을 쓰지 않는 이유: 100ms 샘플링(10fps)에서는 빠른 공이 프레임 사이에
자기 크기보다 멀리 이동해 IoU 연계가 실패하고, 공 검출이 결과에서 통째로
탈락한다 (실측: track 44% vs predict 57%). 시간적 연속성은 BallTracker
(tracking.py, 등속 예측)가 담당한다.

클래스 매핑은 model.names에서 자동 인식한다 — COCO(person/sports ball)와
커스텀 모델(ball/player) 모두 설정 없이 동작한다.
"""

from collections.abc import Iterator

import numpy as np

from .errors import ErrorCode, TrackCropError
from .types import Detection

OBJECT_TYPE_BALL = "ball"
OBJECT_TYPE_PLAYER = "player"

# 프레임 간 동일 객체 판정 최대 이동 거리 (px / 100ms 샘플 간격) — 구현 세부
_MATCH_MAX_DIST = {OBJECT_TYPE_BALL: 400.0, OBJECT_TYPE_PLAYER: 200.0}


class _IdAssigner:
    """프레임 간 최근접 중심 매칭으로 track_id를 부여한다."""

    def __init__(self) -> None:
        self._next_id = 1
        self._previous: list[tuple[int, str, float, float]] = []  # (id, type, cx, cy)

    def assign(self, detections: list[Detection]) -> None:
        current: list[tuple[int, str, float, float]] = []
        available = list(self._previous)
        for det in detections:
            cx, cy = det.center_x, det.bbox_y + det.bbox_height / 2
            best_i, best_dist = None, _MATCH_MAX_DIST[det.object_type]
            for i, (_, prev_type, px, py) in enumerate(available):
                if prev_type != det.object_type:
                    continue
                dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                if dist < best_dist:
                    best_i, best_dist = i, dist
            if best_i is not None:
                det.track_id = available.pop(best_i)[0]
            else:
                det.track_id = self._next_id
                self._next_id += 1
            current.append((det.track_id, det.object_type, cx, cy))
        self._previous = current


class Detector:
    def __init__(self, model_path: str, device: str, imgsz: int, conf: float):
        # ultralytics·torch는 import에 수 초가 걸려 지연 로딩한다
        try:
            from ultralytics import YOLO

            self._model = YOLO(model_path)
        except Exception as e:
            raise TrackCropError(
                ErrorCode.MODEL_LOAD_FAILED,
                f"탐지 모델을 로딩할 수 없습니다: {model_path}",
                details={"model_path": model_path, "cause": str(e)[:300]},
            ) from e

        self._device = device
        self._imgsz = imgsz
        self._conf = conf
        self._type_by_class = self._map_classes(self._model.names)
        if OBJECT_TYPE_BALL not in self._type_by_class.values():
            raise TrackCropError(
                ErrorCode.MODEL_LOAD_FAILED,
                "모델에 ball 클래스가 없습니다.",
                details={"names": dict(self._model.names)},
            )

    @staticmethod
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

    def track(
        self, frames: Iterator[tuple[int, np.ndarray]]
    ) -> Iterator[tuple[int, list[Detection]]]:
        """(offset_ms, Detection 목록)을 Sample 단위로 생성한다."""
        assigner = _IdAssigner()
        target_classes = list(self._type_by_class)
        for offset_ms, frame in frames:
            try:
                result = self._model.predict(
                    frame,
                    classes=target_classes,
                    imgsz=self._imgsz,
                    conf=self._conf,
                    device=self._device,
                    verbose=False,
                )[0]
            except Exception as e:
                raise TrackCropError(
                    ErrorCode.OBJECT_DETECTION_FAILED,
                    "객체 탐지 추론 중 오류가 발생했습니다.",
                    details={"video_offset_ms": offset_ms, "cause": str(e)[:300]},
                ) from e

            detections = self._to_detections(result, offset_ms)
            assigner.assign(detections)
            yield offset_ms, detections

    def _to_detections(self, result, offset_ms: int) -> list[Detection]:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        xywh = boxes.xywh.tolist()  # (center_x, center_y, w, h)
        classes = boxes.cls.tolist()
        confs = boxes.conf.tolist()

        detections = []
        for (cx, cy, w, h), cls, conf in zip(xywh, classes, confs, strict=True):
            object_type = self._type_by_class.get(int(cls))
            if object_type is None:
                continue
            detections.append(
                Detection(
                    object_type=object_type,
                    track_id=None,  # _IdAssigner가 부여
                    bbox_x=cx - w / 2,
                    bbox_y=cy - h / 2,
                    bbox_width=w,
                    bbox_height=h,
                    confidence=conf,
                    video_offset_ms=offset_ms,
                )
            )
        return detections
