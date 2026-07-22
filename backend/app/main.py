from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.db import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.ensure_dirs()
    init_db()
    from app.jobs.train_manager import train_manager

    train_manager.reconcile_on_boot()
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="yolo-vision-toolkit", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health():
        return {"status": "ok", "data_dir": str(settings.data_dir)}

    from app.api import datasets, files, jobs, models, projects, review, system, training, videos

    app.include_router(models.router)
    app.include_router(projects.router)
    app.include_router(jobs.router)
    app.include_router(review.router)
    app.include_router(datasets.router)
    app.include_router(training.router)
    app.include_router(files.router)
    app.include_router(system.router)
    app.include_router(videos.router)

    return app


app = create_app()
