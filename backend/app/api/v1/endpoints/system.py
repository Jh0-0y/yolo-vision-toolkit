"""System / runtime info: resolved compute device, accelerator, VRAM."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool

from app.core.config import get_device_info

router = APIRouter(prefix="/system", tags=["system"])


@router.get("/device")
async def device_info():
    # torch calls (mem_get_info, sysctl) are sync/blocking → offload
    return await run_in_threadpool(get_device_info)
