"""trackcrop — 가로형 스포츠 영상에서 객체를 추적해 세로형 Crop X 좌표를 계산한다.

LivePick CropWorker의 탐지·추적·좌표 계산 파이프라인을 서버/API/S3 의존 없이
독립 실행할 수 있게 추출한 모듈이다. 실제 영상 크롭은 하지 않고 좌표만 낸다.

기본 사용:

    from trackcrop import analyze_video

    result = analyze_video("clip.mp4", model_path="yolo26m.pt", device="cpu")
    print(result.to_json())
    for kf in result.keyframes:
        print(kf.video_offset_ms, kf.x)

모델을 한 번만 로딩해 여러 영상 처리:

    from trackcrop import Detector, analyze_video

    det = Detector(model_path="yolo26m.pt", device="cpu", imgsz=1280, conf=0.10)
    for path in paths:
        result = analyze_video(path, detector=det)
"""

from .detection import Detector
from .errors import ErrorCode, TrackCropError
from .keyframe import reduce_keyframes, to_crop_x
from .pipeline import analyze_video
from .result import build_crop_result, validate_crop_result
from .sampling import sample_frames
from .stabilization import stabilize
from .target import resolve_targets
from .tracking import BallTracker, player_group_center, select_ball
from .types import CropResult, Detection, Keyframe, TargetSample, TargetType

__all__ = [
    "analyze_video",
    "Detector",
    "CropResult",
    "Detection",
    "Keyframe",
    "TargetSample",
    "TargetType",
    "BallTracker",
    "select_ball",
    "player_group_center",
    "resolve_targets",
    "stabilize",
    "reduce_keyframes",
    "to_crop_x",
    "sample_frames",
    "build_crop_result",
    "validate_crop_result",
    "TrackCropError",
    "ErrorCode",
]
