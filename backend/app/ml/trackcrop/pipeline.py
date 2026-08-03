"""추적 → Crop 좌표 파이프라인 오케스트레이터.

  프레임 샘플링 → 탐지·추적ID → 클립 플래너(트랙 확정·타깃 결정·경로 최적화)
  → Keyframe 축약 → 결과 조립

순수 계산만 하며 파일 입출력·네트워크는 하지 않는다 (영상 경로 읽기만).

파이프라인은 두 단계로 나뉜다:
  - `detect_video` — 프레임 샘플링 + YOLO·ByteTrack 추론 (비쌈, 모델 필요).
  - `plan_from_detections` — 클립 플래너 + Keyframe 축약 (쌈, 순수 계산).
`analyze_video`는 둘을 이어 붙인 편의 함수다. 검출 결과를 캐시해두고 튜닝
파라미터만 바꿔 좌표를 즉시 재계산하려면(라이브 프리뷰) 두 단계를 따로 부른다.
"""

from collections.abc import Callable
from pathlib import Path

from .balltrack import ClipPlanConfig, resolve_clip_config
from .clip_planner import plan_clip
from .detection import Detector
from .keyframe import reduce_keyframes
from .result import build_crop_result, validate_crop_result
from .sampling import sample_frames
from .types import CropResult, Detection

# YOLO 기본값 (원본 CropWorker Settings 기준)
DEFAULT_MODEL_PATH = "yolo26m.pt"
DEFAULT_DEVICE = "mps"  # "cpu" | "cuda" | "mps"
DEFAULT_IMGSZ = 1920
DEFAULT_CONF = 0.10  # 낮은 값 + 추적기 필터로 공 검출률 확보

# (video_offset_ms, 그 시점의 Detection 목록) 시퀀스 — 검출 단계의 산출물.
DetectedSamples = list[tuple[int, list[Detection]]]


def detect_video(
    video_path: str | Path,
    *,
    detector: Detector | None = None,
    model_path: str = DEFAULT_MODEL_PATH,
    device: str = DEFAULT_DEVICE,
    imgsz: int = DEFAULT_IMGSZ,
    conf: float = DEFAULT_CONF,
    sampling_interval_ms: int | None = None,
    on_progress: Callable[[int], None] | None = None,
) -> DetectedSamples:
    """영상을 샘플링해 시점별 Detection 목록을 낸다 (비싼 추론 단계).

    이 결과는 튜닝 파라미터와 무관하다(모델·conf·imgsz·샘플링 간격에만 의존).
    한 번 구해 캐시해두면 좌표 재계산은 `plan_from_detections`로 반복할 수 있다.

    detector를 주면 재사용한다. sampling_interval_ms 미지정 시 기본값(constants).
    on_progress(done)가 있으면 샘플 처리마다 누적 개수를 통지한다(진행률·취소용).
    """
    if detector is None:
        detector = Detector(model_path=model_path, device=device, imgsz=imgsz, conf=conf)
    interval = sampling_interval_ms or ClipPlanConfig().sampling_interval_ms

    detected: DetectedSamples = []
    for i, (offset_ms, dets) in enumerate(
        detector.track(sample_frames(Path(video_path), interval))
    ):
        detected.append((offset_ms, dets))
        if on_progress is not None:
            on_progress(i + 1)
    return detected


def plan_from_detections(
    detected: DetectedSamples,
    *,
    overrides: dict | None = None,
    collect_debug: bool = False,
    validate: bool = True,
) -> CropResult:
    """캐시된 Detection 시퀀스에서 Crop 좌표(CropResult)를 계산한다 (싼 계산 단계).

    클립 플래너(plan_clip)가 트랙 스티칭·타깃 결정·전역 경로 최적화를 한 번에
    수행한다 — 스무딩·속도 캡이 pathopt에서 끝나므로 별도 안정화 단계가 없다.

    torch/모델 없이 순수 계산만 하므로 튜닝만 바꿔 밀리초 단위로 재호출할 수 있다.
    overrides의 sampling_interval_ms는 여기선 무시된다(간격 변경은 detect_video 재실행).
    """
    cfg = resolve_clip_config(overrides)
    debug: list | None = [] if collect_debug else None

    planned, _info = plan_clip(detected, cfg, debug)
    keyframes = reduce_keyframes(planned)
    result = build_crop_result(planned, keyframes, debug=debug)

    if validate:
        violations = validate_crop_result(result, cfg.max_move_px_per_second)
        if violations:
            raise ValueError(f"Crop 결과 검증 실패: {violations}")

    return result


def analyze_video(
    video_path: str | Path,
    *,
    detector: Detector | None = None,
    model_path: str = DEFAULT_MODEL_PATH,
    device: str = DEFAULT_DEVICE,
    imgsz: int = DEFAULT_IMGSZ,
    conf: float = DEFAULT_CONF,
    validate: bool = True,
    overrides: dict | None = None,
    collect_debug: bool = False,
) -> CropResult:
    """영상을 분석해 Crop 좌표(CropResult)를 계산한다 (검출 + 좌표계산 일괄).

    detector를 주면 재사용한다 (여러 영상 처리 시 모델 1회 로딩).
    주지 않으면 model_path/device/imgsz/conf로 새로 만든다.

    overrides로 튜닝 값(데드존·샘플링 간격 등)을 런타임 오버라이드한다.
    collect_debug=True면 CropResult.debug에 시점별 선택 공/소유선수 bbox를 채운다.
    validate=True면 결과 자체 검증에 실패할 때 ValueError를 던진다.
    """
    cfg = resolve_clip_config(overrides)

    if detector is None:
        detector = Detector(model_path=model_path, device=device, imgsz=imgsz, conf=conf)

    detected = detect_video(
        video_path, detector=detector, sampling_interval_ms=cfg.sampling_interval_ms
    )
    return plan_from_detections(
        detected, overrides=overrides, collect_debug=collect_debug, validate=validate
    )
