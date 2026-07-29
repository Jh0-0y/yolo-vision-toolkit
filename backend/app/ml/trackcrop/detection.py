"""YOLO 객체 탐지 + ByteTrack 추적.

탐지·추적은 ultralytics 내장 ByteTrack(`model.track`)으로 수행하고 track_id는
`boxes.id`에서 받는다 (스펙 DET-01·DET-05). 공의 시간적 연속성·가림 예측은
BallTracker(tracking.py, 등속 예측)가 보조한다.

⚠️ 100ms 샘플링(10fps)에서는 빠른 공이 프레임 사이 IoU 연계에 실패해 검출이
떨어질 수 있다(실측 track 44% vs predict 57%). 그럼에도 원본 스펙이 ByteTrack을
규정하므로 이를 따른다. 트레이드오프 논의는 프로젝트 문서(trackcrop-considerations) 참고.

클래스 매핑은 model.names에서 자동 인식한다 — COCO(person/sports ball)와
커스텀 모델(ball/player) 모두 설정 없이 동작한다.
"""

from collections.abc import Iterator

import numpy as np

from .errors import ErrorCode, TrackCropError
from .types import Detection

OBJECT_TYPE_BALL = "ball"
OBJECT_TYPE_PLAYER = "player"


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
        """(offset_ms, Detection 목록)을 Sample 단위로 생성한다.

        ByteTrack을 persist=True로 프레임 간 상태를 유지하며, track_id는 boxes.id에서 받는다.
        """
        self._reset_tracker()  # 새 영상마다 tracker 상태 초기화
        target_classes = list(self._type_by_class)
        for offset_ms, frame in frames:
            try:
                result = self._model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    classes=target_classes,
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

            yield offset_ms, self._to_detections(result, offset_ms)

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
