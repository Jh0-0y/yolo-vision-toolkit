"""파일 이름 규칙 — 무엇을 살리고 무엇을 막는가.

전에는 영숫자 외를 전부 밀어서 한글 이름 영상 두 개가 똑같이 `video` 가 되고
서로 덮어썼다. 그래서 **깨지는 문자만** 막고 나머지는 살린다. 여기서 그 경계를
못 박는다 — 느슨해지면 URL 이 깨지고, 빡빡해지면 이름이 사라진다.
"""

import unicodedata

import pytest

from lib.fsutil import safe_stem, unique_stem


# ---------- 살려야 하는 것 ----------


@pytest.mark.parametrize(
    "name, expected",
    [
        ("경기영상.mp4", "경기영상"),  # 한글이 통째로 사라지던 자리
        ("2025-07-04 vs A (1).mp4", "2025-07-04 vs A (1)"),  # 공백·괄호는 파일명에 문제없다
        ("game_01.mp4", "game_01"),
        ("한글 mixed 이름.mov", "한글 mixed 이름"),
    ],
)
def test_keeps_names_that_are_not_dangerous(name, expected):
    assert safe_stem(name) == expected


def test_two_different_korean_names_stay_different():
    """옛 규칙은 둘 다 `video` 로 만들어 프레임이 서로 덮어썼다."""
    assert safe_stem("경기영상.mp4") != safe_stem("연습영상.mp4")


# ---------- 막아야 하는 것 ----------


@pytest.mark.parametrize(
    "name",
    [
        "a#b.mp4",  # 프래그먼트로 잘려 이미지 URL 이 404
        "a?b.mp4",  # 쿼리로 새어 404
        "a%b.mp4",  # 잘못된 퍼센트 이스케이프
        "a:b.mp4",  # 윈도우에서 불법 — 내보낸 zip 을 못 푼다
        'a"b.mp4',
        "a<b.mp4",
        "a|b.mp4",
    ],
)
def test_replaces_characters_that_break_urls_or_windows(name):
    assert safe_stem(name) == "a_b"


def test_strips_path_separators():
    assert safe_stem("dir/sub/clip.mp4") == "clip"
    assert safe_stem("a\\b.mp4") == "a_b"


@pytest.mark.parametrize("name", [".hidden.mp4", "..", ".", "...mp4"])
def test_leading_dots_are_removed(name):
    """점으로 시작하면 목록에서 빠지고 서빙도 막혀 **조용히 사라진다.**"""
    assert not safe_stem(name).startswith(".")


def test_falls_back_when_nothing_survives():
    assert safe_stem("###.mp4", fallback="video") == "_"  # 치환된 것은 남는다
    assert safe_stem("...", fallback="video") == "video"
    assert safe_stem("", fallback="video") == "video"


def test_control_characters_go():
    assert safe_stem("a\x00b\x1fc.mp4") == "a_b_c"


# ---------- 정규화와 길이 ----------


def test_normalizes_to_nfc():
    """macOS 가 NFD 로 돌려주는 자리가 있다. 키가 어긋나면 검수·분할이 조용히 깨진다."""
    nfd = unicodedata.normalize("NFD", "경기.mp4")
    assert safe_stem(nfd) == unicodedata.normalize("NFC", "경기")


def test_caps_the_length_in_bytes():
    """한글은 3바이트다 — 글자 수가 아니라 바이트로 재야 255 한계에 안 걸린다."""
    stem = safe_stem("가" * 200 + ".mp4")
    assert len(stem.encode()) <= 150
    assert stem  # 통째로 날아가지는 않는다


# ---------- 이름이 겹칠 때 ----------
#
# 덮어쓰지 않는다. 이름이 같아도 다른 데이터일 수 있고, 덮어쓰면 사라진 쪽을
# 되돌릴 수 없다. `(2)` 로 남겨 두면 사람이 보고 정한다.


def test_free_stem_is_returned_as_is():
    assert unique_stem("경기 영상", lambda s: False) == "경기 영상"


def test_taken_stem_gets_a_number():
    taken = {"경기 영상"}
    assert unique_stem("경기 영상", lambda s: s in taken) == "경기 영상 (2)"


def test_numbers_keep_climbing():
    taken = {"a", "a (2)", "a (3)"}
    assert unique_stem("a", lambda s: s in taken) == "a (4)"


def test_gives_up_loudly_rather_than_looping_forever():
    with pytest.raises(RuntimeError, match="free name"):
        unique_stem("a", lambda s: True, limit=3)
