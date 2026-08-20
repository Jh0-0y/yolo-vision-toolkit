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
    from app.services import lab_crop_runs

    # 크롭 워커 풀은 이 프로세스가 소유한다 — 재시작하면 돌던 런은 죽었다.
    lab_crop_runs.reconcile_on_boot()
    from app.services import benchmarks

    # 벤치마크 워커 풀도 이 프로세스가 소유한다 — 재시작하면 돌던 런은 죽었다.
    benchmarks.reconcile_on_boot()
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
