"""검출 결과(DetectedSamples)의 JSON 직렬화.

`detect_video`가 낸 `list[(video_offset_ms, list[Detection])]`를 파일로 저장하고
다시 읽어 `plan_from_detections`에 넘기기 위한 헬퍼. Detection은 원시 필드만 가진
dataclass라 손실 없이 왕복한다(bbox 좌표는 소수 1자리로 반올림해 용량만 줄인다).
"""

from __future__ import annotations

from .types import Detection


def _round(v: float) -> float:
    """오버레이 정밀도엔 충분하고 JSON 용량은 줄이는 소수 1자리 반올림."""
    return round(float(v), 1)


def dump_detections(detected: list[tuple[int, list[Detection]]]) -> list[dict]:
    """DetectedSamples → JSON 직렬화 가능한 dict 목록."""
    return [
        {
            "video_offset_ms": int(offset_ms),
            "detections": [
                {
                    "object_type": d.object_type,
                    "track_id": d.track_id,
                    "bbox_x": _round(d.bbox_x),
                    "bbox_y": _round(d.bbox_y),
                    "bbox_width": _round(d.bbox_width),
                    "bbox_height": _round(d.bbox_height),
                    "confidence": round(float(d.confidence), 4),
                    "video_offset_ms": int(offset_ms),
                }
                for d in dets
            ],
        }
        for offset_ms, dets in detected
    ]


def load_detections(data: list[dict]) -> list[tuple[int, list[Detection]]]:
    """dump_detections의 역변환 — plan_from_detections에 바로 넘길 수 있는 형태."""
    out: list[tuple[int, list[Detection]]] = []
    for sample in data:
        offset_ms = int(sample["video_offset_ms"])
        dets = [
            Detection(
                object_type=d["object_type"],
                track_id=d["track_id"],
                bbox_x=float(d["bbox_x"]),
                bbox_y=float(d["bbox_y"]),
                bbox_width=float(d["bbox_width"]),
                bbox_height=float(d["bbox_height"]),
                confidence=float(d["confidence"]),
                video_offset_ms=int(d.get("video_offset_ms", offset_ms)),
            )
            for d in sample.get("detections", [])
        ]
        out.append((offset_ms, dets))
    return out
