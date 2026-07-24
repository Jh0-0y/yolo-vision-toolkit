"""One-off migration: bucket layout (confirmed/review/rejected) → labels/ + reviewed.json.

- confirmed/labels/*.txt  → labels/{stem}.txt, stem marked reviewed (was user-approved)
- review/{stem}.json      → labels/{stem}.txt from "edits" (if present) else non-rejected
                            "boxes"; stem stays un-reviewed
- confirmed/, review/, rejected/ directories are removed (confirmed/images are
  copies — the originals live in raw/)

Usage (from backend/):
    uv run python scripts/migrate_buckets.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import settings  # noqa: E402
from app.domain.labels import read_reviewed, write_reviewed  # noqa: E402
from app.domain.yolo_io import write_label_file  # noqa: E402


def migrate_project(pdir: Path, dry_run: bool) -> dict:
    stats = {"confirmed": 0, "review": 0, "skipped": 0}
    labels_dir = pdir / "labels"
    reviewed = read_reviewed(pdir)

    confirmed_labels = pdir / "confirmed" / "labels"
    if confirmed_labels.exists():
        for txt in sorted(confirmed_labels.glob("*.txt")):
            dst = labels_dir / txt.name
            if dst.exists():
                stats["skipped"] += 1
                continue
            if not dry_run:
                labels_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(txt, dst)
            reviewed.add(txt.stem)
            stats["confirmed"] += 1

    review_dir = pdir / "review"
    if review_dir.exists():
        for jf in sorted(review_dir.glob("*.json")):
            dst = labels_dir / f"{jf.stem}.txt"
            if dst.exists():
                stats["skipped"] += 1
                continue
            try:
                payload = json.loads(jf.read_text())
            except (json.JSONDecodeError, OSError):
                stats["skipped"] += 1
                continue
            boxes = payload.get("edits") or payload.get("boxes") or []
            rows = [
                (int(b["cls"]), tuple(float(v) for v in b["xyxy_n"]))
                for b in boxes
                if b.get("status") != "rejected"
            ]
            if not dry_run:
                write_label_file(dst, rows)
            stats["review"] += 1

    if not dry_run:
        write_reviewed(pdir, reviewed)
        for sub in ("confirmed", "review", "rejected"):
            shutil.rmtree(pdir / sub, ignore_errors=True)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    projects_dir = settings.projects_dir
    if not projects_dir.exists():
        print(f"no projects dir: {projects_dir}")
        return

    for pdir in sorted(p for p in projects_dir.iterdir() if p.is_dir()):
        stats = migrate_project(pdir, args.dry_run)
        tag = " (dry-run)" if args.dry_run else ""
        print(
            f"{pdir.name}{tag}: confirmed→labels {stats['confirmed']}, "
            f"review→labels {stats['review']}, skipped {stats['skipped']}"
        )


if __name__ == "__main__":
    main()
