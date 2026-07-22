"""Routing rules: which boxes are trusted, which images need human review."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.ensemble import FusedBox, iou


@dataclass
class DecisionConfig:
    conf_min: float = 0.10  # inference floor; boxes below never surface
    conf_confirm: float = 0.60  # default auto-accept threshold
    per_class_confirm: dict[int, float] = field(default_factory=dict)
    iou_conflict: float = 0.50  # different-class overlap flagged above this
    # what to do with images that have zero boxes: "review" | "negative"
    empty_policy: str = "review"

    def confirm_thr(self, cls: int) -> float:
        return self.per_class_confirm.get(cls, self.conf_confirm)


@dataclass
class BoxVerdict:
    box: FusedBox
    status: str  # "auto_ok" | "needs_review"
    reason: str | None = None  # low_conf | disagreement | class_conflict


def judge_boxes(
    boxes: list[FusedBox],
    class_sources: dict[int, list[str]],
    cfg: DecisionConfig,
) -> list[BoxVerdict]:
    verdicts: list[BoxVerdict] = []
    for b in boxes:
        multi_source = len(class_sources.get(b.cls, [])) >= 2
        if b.score < cfg.confirm_thr(b.cls):
            verdicts.append(BoxVerdict(b, "needs_review", "low_conf"))
        elif multi_source and b.agree_count < 2:
            verdicts.append(BoxVerdict(b, "needs_review", "disagreement"))
        else:
            verdicts.append(BoxVerdict(b, "auto_ok"))

    # Class conflict: heavy overlap between different-class boxes — flag the
    # lower-scoring one (suspected duplicate detection of the same object).
    for i in range(len(verdicts)):
        for j in range(i + 1, len(verdicts)):
            a, b = verdicts[i].box, verdicts[j].box
            if a.cls == b.cls:
                continue
            if iou(a.xyxy, b.xyxy) > cfg.iou_conflict:
                loser = verdicts[j] if b.score <= a.score else verdicts[i]
                if loser.status == "auto_ok":
                    loser.status = "needs_review"
                    loser.reason = "class_conflict"
    return verdicts


def route_image(verdicts: list[BoxVerdict], cfg: DecisionConfig) -> str:
    """Returns the bucket for an image: "confirmed" | "review" | "negative"."""
    if not verdicts:
        return "negative" if cfg.empty_policy == "negative" else "review"
    if all(v.status == "auto_ok" for v in verdicts):
        return "confirmed"
    return "review"


def uncertainty_score(verdicts: list[BoxVerdict]) -> float:
    flagged = [v.box.score for v in verdicts if v.status == "needs_review"]
    if not flagged:
        return 0.0
    return 1.0 - sum(flagged) / len(flagged)
