"""System / runtime info: resolved compute device, accelerator, VRAM/RAM, and a
resource guard for the Test playground."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool

from app.core.config import get_device_info, get_resource_info

router = APIRouter(prefix="/system", tags=["system"])


@router.get("/device")
async def device_info():
    # torch calls (mem_get_info, sysctl) are sync/blocking → offload
    return await run_in_threadpool(get_device_info)


def _warning(info: dict) -> str | None:
    """Human-facing memory/contention warning for running inference now."""
    accel = info.get("accelerator")
    training = info.get("training_active")
    if training and accel == "mps":
        return "학습이 진행 중입니다. 통합 메모리를 공유하므로 추론은 CPU 사용을 권장합니다."
    if training and accel == "cuda":
        free = info.get("vram_free_mb")
        if free is not None and free < 1500:
            return f"학습 중이며 VRAM 여유가 적습니다 (≈{free}MB). OOM 위험이 있습니다."
        return "학습이 GPU에서 진행 중입니다. 함께 실행하면 둘 다 느려질 수 있습니다."
    if accel == "cuda":
        free = info.get("vram_free_mb")
        if free is not None and free < 800:
            return f"VRAM 여유가 적습니다 (≈{free}MB)."
    else:
        avail, total = info.get("ram_available_mb"), info.get("ram_total_mb")
        if avail and total and avail < total * 0.1:
            return f"시스템 메모리 여유가 적습니다 (≈{avail}MB)."
    return None


@router.get("/resources")
async def resources():
    """Device + live memory pressure + training/residency state → warning.
    Used by the Test page's resource banner (safe to poll — never touches the
    inference worker)."""

    def _build() -> dict:
        from app.services.infer_manager import infer_manager
        from app.services.train_manager import train_manager

        info = get_resource_info()
        info["training_active"] = train_manager.has_active()
        info["resident_models"] = infer_manager.resident_count()
        info["warning"] = _warning(info)
        return info

    return await run_in_threadpool(_build)
