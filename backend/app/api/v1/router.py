"""Versioned API aggregation.

Mounts every v1 endpoint router under `settings.api_prefix` (/api/v1). This is
the single place routers are wired — `main.py` includes only this one router.
"""

from fastapi import APIRouter

from app.api.v1.endpoints import (
    classes,
    crops,
    exports,
    files,
    jobs,
    labels,
    models,
    predict,
    projects,
    system,
    training,
    videos,
)
from app.core.config import settings

api_router = APIRouter(prefix=settings.api_prefix)


@api_router.get("/health", tags=["system"])
def health():
    return {"status": "ok", "data_dir": str(settings.data_dir)}


for _router in (
    models.router,
    projects.router,
    predict.router,
    jobs.router,
    exports.router,
    crops.router,
    training.router,
    files.router,
    system.router,
    videos.router,
    labels.router,
    classes.router,
):
    api_router.include_router(_router)
