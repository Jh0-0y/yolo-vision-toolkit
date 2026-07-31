"""추적 → Crop 좌표 파이프라인 오케스트레이터.

  프레임 샘플링 → 탐지·추적ID → Target 선정 → 안정화 → Keyframe 축약 → 결과 조립

순수 계산만 하며 파일 입출력·네트워크는 하지 않는다 (영상 경로 읽기만).
"""

from pathlib import Path

from .detection import Detector
from .keyframe import reduce_keyframes
from .result import build_crop_result, validate_crop_result
from .sampling import sample_frames
from .stabilization import stabilize
from .target import resolve_targets
from .types import CropResult

# YOLO 기본값 (원본 CropWorker Settings 기준)
DEFAULT_MODEL_PATH = "yolo26m.pt"
DEFAULT_DEVICE = "mps"  # "cpu" | "cuda" | "mps"
DEFAULT_IMGSZ = 1920
DEFAULT_CONF = 0.10  # 낮은 값 + 추적기 필터로 공 검출률 확보


def analyze_video(
    video_path: str | Path,
    *,
    detector: Detector | None = None,
    model_path: str = DEFAULT_MODEL_PATH,
    device: str = DEFAULT_DEVICE,
    imgsz: int = DEFAULT_IMGSZ,
    conf: float = DEFAULT_CONF,
    validate: bool = True,
) -> CropResult:
    """영상을 분석해 Crop 좌표(CropResult)를 계산한다.

    detector를 주면 재사용한다 (여러 영상 처리 시 모델 1회 로딩).
    주지 않으면 model_path/device/imgsz/conf로 새로 만든다.

    validate=True면 결과 자체 검증에 실패할 때 ValueError를 던진다.
    """
    if detector is None:
        detector = Detector(model_path=model_path, device=device, imgsz=imgsz, conf=conf)

    video = Path(video_path)
    detected = list(detector.track(sample_frames(video)))
    targets = resolve_targets(detected)
    stabilized = stabilize(targets)
    keyframes = reduce_keyframes(stabilized)
    result = build_crop_result(stabilized, keyframes)

    if validate:
        violations = validate_crop_result(result)
        if violations:
            raise ValueError(f"Crop 결과 검증 실패: {violations}")

    return result
