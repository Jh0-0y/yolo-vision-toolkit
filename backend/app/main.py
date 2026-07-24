from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.db import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.ensure_dirs()
    init_db()
    from app.services.train_manager import train_manager

    train_manager.reconcile_on_boot()
    from app.services.test_jobs import sweep_old_annotations

    sweep_old_annotations()  # clear stale annotated videos from a previous run
    yield
    # release the warm inference worker (and its framework context) on shutdown
    from app.services.infer_manager import infer_manager

    infer_manager.shutdown()


def create_app() -> FastAPI:
    app = FastAPI(title="yolo-vision-toolkit", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # unversioned infra health check (docker/uptime probes hit this)
    @app.get("/api/health")
    def health():
        return {"status": "ok", "data_dir": str(settings.data_dir)}

    from app.api.v1.router import api_router

    app.include_router(api_router)

    return app


app = create_app()
