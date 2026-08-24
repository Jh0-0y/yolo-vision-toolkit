"""클래스 레지스트리(classes.json) — 순수 파일 조작, DB 도 웹도 없다."""

from lib.labels.classes import count_boxes_by_class, count_boxes_with_class


def _write_labels(tmp_path, files: dict[str, str]):
    """`{파일명: 내용}` 으로 라벨 디렉터리를 만든다."""
    labels = tmp_path / "labels"
    labels.mkdir(parents=True, exist_ok=True)
    for name, body in files.items():
        (labels / name).write_text(body)
    return tmp_path


def test_counts_every_class_in_one_pass(tmp_path):
    pdir = _write_labels(
        tmp_path,
        {
            "a.txt": "0 0.1 0.1 0.2 0.2\n1 0.3 0.3 0.2 0.2\n0 0.5 0.5 0.1 0.1\n",
            "b.txt": "2 0.4 0.4 0.2 0.2\n0 0.6 0.6 0.1 0.1\n",
        },
    )

    assert count_boxes_by_class(pdir) == {0: 3, 1: 1, 2: 1}


def test_missing_labels_dir_counts_nothing(tmp_path):
    assert count_boxes_by_class(tmp_path) == {}


def test_empty_and_malformed_lines_are_skipped(tmp_path):
    """라벨 파일에 빈 줄이나 깨진 줄이 있어도 세는 일이 멈추면 안 된다."""
    pdir = _write_labels(
        tmp_path,
        {"a.txt": "\n0 0.1 0.1 0.2 0.2\nball 0.3 0.3 0.2 0.2\n\n1 0.4 0.4 0.1 0.1\n"},
    )

    assert count_boxes_by_class(pdir) == {0: 1, 1: 1}


def test_a_class_with_no_boxes_is_simply_absent(tmp_path):
    """쓰이지 않은 클래스는 키가 없다 — 화면에서 0 으로 채우는 것은 호출자 몫이다."""
    pdir = _write_labels(tmp_path, {"a.txt": "0 0.1 0.1 0.2 0.2\n"})

    counts = count_boxes_by_class(pdir)

    assert counts == {0: 1}
    assert counts.get(5, 0) == 0


def test_agrees_with_the_single_class_counter(tmp_path):
    """삭제 경로가 쓰는 기존 함수와 답이 갈리면 안 된다."""
    pdir = _write_labels(
        tmp_path,
        {
            "a.txt": "0 0.1 0.1 0.2 0.2\n1 0.3 0.3 0.2 0.2\n",
            "b.txt": "1 0.4 0.4 0.2 0.2\n1 0.6 0.6 0.1 0.1\n",
        },
    )

    counts = count_boxes_by_class(pdir)

    for cid in (0, 1, 2):
        assert counts.get(cid, 0) == count_boxes_with_class(pdir, cid)
