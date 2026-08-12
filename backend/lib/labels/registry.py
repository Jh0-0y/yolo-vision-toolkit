"""Global class registry: unions class sets of N models by normalized name.

Global ids are append-only — adding a model later never renumbers existing
classes, because confirmed label files reference these ids on disk.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def normalize(name: str) -> str:
    return name.strip().lower()


@dataclass
class GlobalClass:
    id: int
    name: str
    sources: list[str] = field(default_factory=list)


class ClassRegistry:
    def __init__(self, classes: list[GlobalClass] | None = None):
        self.classes: list[GlobalClass] = classes or []
        self._by_norm: dict[str, GlobalClass] = {normalize(c.name): c for c in self.classes}

    def add_model(self, model_id: str, names: dict[int, str]) -> dict[int, int]:
        """Register a model's class names. Returns local id -> global id mapping."""
        mapping: dict[int, int] = {}
        for local_id, name in sorted(names.items()):
            key = normalize(name)
            gc = self._by_norm.get(key)
            if gc is None:
                gc = GlobalClass(id=len(self.classes), name=name.strip())
                self.classes.append(gc)
                self._by_norm[key] = gc
            if model_id not in gc.sources:
                gc.sources.append(model_id)
            mapping[local_id] = gc.id
        return mapping

    @property
    def names(self) -> dict[int, str]:
        return {c.id: c.name for c in self.classes}

    def class_sources(self) -> dict[int, list[str]]:
        return {c.id: list(c.sources) for c in self.classes}

    def to_dict(self) -> dict:
        return {
            "classes": [
                {"id": c.id, "name": c.name, "sources": c.sources} for c in self.classes
            ]
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ClassRegistry":
        classes = [
            GlobalClass(id=c["id"], name=c["name"], sources=list(c.get("sources", [])))
            for c in sorted(data.get("classes", []), key=lambda c: c["id"])
        ]
        for i, c in enumerate(classes):
            if c.id != i:
                raise ValueError(f"class registry ids must be contiguous, got {c.id} at index {i}")
        return cls(classes)
