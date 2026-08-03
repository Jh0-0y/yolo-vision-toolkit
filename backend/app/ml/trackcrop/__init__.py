"""trackcrop — 가로형 스포츠 영상에서 객체를 추적해 세로형 Crop X 좌표를 계산한다.

LivePick CropWorker의 탐지·추적·좌표 계산 파이프라인을 서버/API/S3 의존 없이
독립 실행할 수 있게 추출한 모듈이다. 실제 영상 크롭은 하지 않고 좌표만 낸다.

좌표 계산은 클립 플래너(클립 전체를 놓고 공 트랙을 확정하고 크롭 경로를 전역
최적화하는 2-패스)가 단일 경로다 — docs/trackcrop-clip-planner.md 참고.

기본 사용:

    from trackcrop import analyze_video

    result = analyze_video("clip.mp4", model_path="yolo26m.pt", device="cpu")
    print(result.to_json())
    for kf in result.keyframes:
        print(kf.video_offset_ms, kf.x)

검출 캐시 재사용 (라이브 프리뷰 — 추론 1번, 좌표 재계산 반복):

    from trackcrop import Detector
    from trackcrop.pipeline import detect_video, plan_from_detections

    detected = detect_video("clip.mp4", model_path="yolo26m.pt")
    result = plan_from_detections(detected, overrides={"dead_zone_width": 260})
"""

from .balltrack import ClipPlanConfig, player_group_center, resolve_clip_config
from .clip_planner import plan_clip, resolve_targets_clip
from .detection import Detector
from .errors import ErrorCode, TrackCropError
from .keyframe import reduce_keyframes, to_crop_x
from .pipeline import analyze_video, detect_video, plan_from_detections
from .result import build_crop_result, validate_crop_result
from .sampling import sample_frames
from .types import CropResult, Detection, Keyframe, TargetSample, TargetType

__all__ = [
    "analyze_video",
    "detect_video",
    "plan_from_detections",
    "plan_clip",
    "resolve_targets_clip",
    "ClipPlanConfig",
    "resolve_clip_config",
    "player_group_center",
    "Detector",
    "CropResult",
    "Detection",
    "Keyframe",
    "TargetSample",
    "TargetType",
    "reduce_keyframes",
    "to_crop_x",
    "sample_frames",
    "build_crop_result",
    "validate_crop_result",
    "TrackCropError",
    "ErrorCode",
]
