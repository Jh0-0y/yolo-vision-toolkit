from pathlib import Path

from PIL import Image

THUMB_SIZE = 256


def get_thumbnail(src: Path, thumbs_dir: Path) -> Path:
    """Return a cached 256px JPEG thumbnail for src, generating it if needed."""
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    thumb = thumbs_dir / f"{src.stem}.jpg"
    if thumb.exists() and thumb.stat().st_mtime >= src.stat().st_mtime:
        return thumb
    with Image.open(src) as im:
        im = im.convert("RGB")
        im.thumbnail((THUMB_SIZE, THUMB_SIZE))
        im.save(thumb, "JPEG", quality=80)
    return thumb
