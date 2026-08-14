"""adaptive-crop 어댑터(lib.crop.plan) 검증 — 번역이 맞는지만 본다.

크롭 알고리즘 자체는 라이브러리 쪽 테스트가 담당한다. 여기서 확인하는 것은
"UI가 보내는 형태"를 라이브러리가 받는 형태로 옮기는 규칙이다.
"""

import pytest
from adaptive_crop import ClipPlanConfig

from lib.crop.geometry import crop_window_for
from lib.crop.plan import (
    crop_spec_for,
    detector_entries,
    resolve_clip_config,
    video_info_from_meta,
)

# ---------------------------------------------------------------------------
# 튜닝 오버라이드


def test_overrides_apply_and_defaults_remain():
    cfg = resolve_clip_config({"dead_zone_width": 300, "w_acc": 20.0})
    assert cfg.dead_zone_width == 300
    assert cfg.w_acc == 20.0
    assert cfg.ball_weight == ClipPlanConfig().ball_weight  # 안 준 값은 기본값


def test_unknown_and_empty_keys_are_dropped():
    # 프론트가 빈 노브를 None으로 보내거나 구버전 키를 남겨도 500이 되면 안 된다
    cfg = resolve_clip_config({"dead_zone_width": None, "nope": 1, "w_vel": 0.3})
    assert cfg.dead_zone_width == ClipPlanConfig().dead_zone_width
    assert cfg.w_vel == 0.3


def test_no_overrides_is_library_default():
    assert resolve_clip_config(None) == ClipPlanConfig()
    assert resolve_clip_config({}) == ClipPlanConfig()


# ---------------------------------------------------------------------------
# 크롭 창 규격


@pytest.mark.parametrize(
    ("source_w", "source_h"), [(1920, 1080), (1280, 720), (3840, 2160), (720, 1280)]
)
def test_crop_spec_matches_the_renderer(source_w, source_h):
    """좌표를 내는 쪽과 화면에 그리는 쪽의 창 폭은 같아야 한다."""
    spec = crop_spec_for(source_w, source_h)
    assert (spec.width, spec.height, spec.y) == crop_window_for(source_w, source_h)
    spec.resolve(source_w, source_h)  # 소스 밖으로 나가지 않는다


def test_crop_spec_1080p_is_the_608_window():
    """기본 px 는 1080p 소스에서 옛 9:16 과 같은 창이다 — 세로는 전체 높이."""
    spec = crop_spec_for(1920, 1080)
    assert (spec.width, spec.height, spec.y) == (608, 1080, 0)


def test_crop_spec_takes_absolute_pixels():
    """비율이 아니다 — 준 값이 곧 창 크기다."""
    assert crop_spec_for(1920, 1080, 1080, 1080).width == 1080  # 정사각
    assert crop_spec_for(3840, 2160, 608, 1080).width == 608  # 4K 라고 커지지 않는다


def test_crop_spec_clamps_to_a_smaller_source():
    """소스보다 큰 창은 만들 수 없다 — 들어가는 만큼 잘라 낸다."""
    spec = crop_spec_for(500, 400, 608, 1080)
    assert (spec.width, spec.height, spec.y) == (500, 400, 0)


def test_a_shorter_window_is_centred_vertically():
    """세로를 낮게 잡으면 위아래가 실제로 잘린다 — 가운데를 남긴다."""
    spec = crop_spec_for(1920, 1080, 608, 720)
    assert (spec.height, spec.y) == (720, 180)


# ---------------------------------------------------------------------------
# 검출기 엔트리


def test_conf_falls_back_to_default_and_iou_is_renamed():
    entries = detector_entries(
        [{"pt": "ball.pt", "mode": "tiled", "conf": None, "merge_iou": 0.4}],
        default_conf=0.1,
    )
    assert entries[0]["conf"] == 0.1  # 라이브러리는 conf가 없으면 ValueError
    assert entries[0]["tile_merge_iou"] == 0.4
    assert "merge_iou" not in entries[0]
    assert entries[0]["tile_size"] == 640 and entries[0]["stride"] == 480


def test_only_the_first_full_entry_gets_track_ids():
    """라이브러리는 track_id를 부여하는 검출기를 하나만 허용한다."""
    entries = detector_entries(
        [
            {"pt": "player.pt", "mode": "full", "conf": 0.6},
            {"pt": "ball_a.pt", "mode": "tiled", "conf": 0.6},
            {"pt": "player_b.pt", "mode": "full", "conf": 0.5},
        ],
        default_conf=0.1,
    )
    assert [e["track_ids"] for e in entries] == [True, False, False]
    assert entries[0]["imgsz"] == 1920  # full 엔트리에만 imgsz


def test_tiled_only_setup_has_no_tracker():
    entries = detector_entries([{"pt": "ball.pt", "mode": "tiled", "conf": 0.6}], 0.1)
    assert entries[0]["track_ids"] is False


# ---------------------------------------------------------------------------
# 영상 정보


def test_video_info_comes_from_the_detection_time_meta():
    meta = {
        "source_width": 1280,
        "source_height": 720,
        "fps": 29.97,
        "duration_ms": 12990,
        "sample_count": 130,
    }
    video = video_info_from_meta(meta)
    assert (video.width, video.height, video.duration_ms) == (1280, 720, 12990)
    assert video.fps == pytest.approx(29.97)
