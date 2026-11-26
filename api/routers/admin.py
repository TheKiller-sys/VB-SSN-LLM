from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

from fastapi import APIRouter, Request

from ..schemas import ShutdownReq

router = APIRouter()


@router.get("/health")
async def health(request: Request):
    q = request.app.state.jobs
    return {
        "ok": True,
        "model_loaded": request.app.state.model.info().get("loaded", False),
        "jobs": q.stats(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }


@router.get("/state")
async def state(request: Request):
    mgr = request.app.state.model
    return {
        "model": mgr.info(),
        "knowledge": mgr.knowledge(),
        "jobs": request.app.state.jobs.stats(),
        "graph_cache_size": len(request.app.state.graph_cache),
        "cwd": os.getcwd(),
        "env": {
            "DATA_ROOT": os.environ.get("SNNLM_DATA_ROOT", "data"),
            "CKPT_ROOT": os.environ.get("SNNLM_CKPT_ROOT", "checkpoints"),
            "LOG_ROOT": os.environ.get("SNNLM_LOG_ROOT", "logs"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }


@router.get("/metrics")
async def metrics(request: Request):
    analyzer = getattr(request.app.state, "analyzer", None)
    return {
        "analyzer": analyzer.get_state() if analyzer else None,
    }


@router.get("/logs")
async def list_logs():
    root = Path("logs")
    if not root.exists():
        return {"logs": []}
    return {
        "logs": [
            {"name": p.name, "size": p.stat().st_size}
            for p in sorted(root.iterdir())
            if p.is_file()
        ]
    }


@router.post("/shutdown")
async def shutdown(req: ShutdownReq):
    if not req.confirm:
        return {"ok": False, "reason": "confirmation required"}
    import asyncio
    import signal

    async def _kill():
        await asyncio.sleep(0.5)
        os.kill(os.getpid(), signal.SIGTERM)

    asyncio.create_task(_kill())
    return {"ok": True}
