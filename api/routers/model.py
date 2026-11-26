from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from ..schemas import LoadModelReq, SignalsReq

router = APIRouter()


def _ckpt_root() -> Path:
    return Path("data/checkpoints")


@router.get("/info")
async def info(request: Request):
    return request.app.state.model.info()


@router.get("/knowledge")
async def knowledge(request: Request):
    return request.app.state.model.knowledge()


@router.get("/ckpts")
async def list_ckpts():
    root = _ckpt_root()
    if not root.exists():
        return {"ckpts": []}
    out = []
    for p in sorted(root.rglob("*.pt")):
        try:
            out.append({
                "path": str(p),
                "name": p.name,
                "size": p.stat().st_size,
                "mtime": p.stat().st_mtime,
            })
        except OSError:
            pass
    return {"ckpts": out}


@router.post("/load")
async def load(req: LoadModelReq, request: Request):
    p = Path(req.ckpt)
    if not p.exists():
        raise HTTPException(404, f"ckpt not found: {req.ckpt}")
    try:
        return request.app.state.model.load(str(p), req.device)
    except Exception as e:
        raise HTTPException(400, str(e))


@router.post("/promote")
async def promote(req: LoadModelReq, request: Request):
    p = Path(req.ckpt)
    if not p.exists():
        raise HTTPException(404, f"ckpt not found: {req.ckpt}")
    try:
        return request.app.state.model.promote(str(p))
    except Exception as e:
        raise HTTPException(400, str(e))


@router.post("/unload")
async def unload(request: Request):
    request.app.state.model.unload()
    return {"ok": True}


@router.post("/signals")
async def signals(req: SignalsReq, request: Request):
    from snn_lm_v3 import SignalProbe
    mgr = request.app.state.model
    if not mgr.info().get("loaded"):
        raise HTTPException(400, "No model loaded")
    try:
        probe = SignalProbe(mgr._model, mgr._device)
        return probe.probe(req.prompt).to_dict()
    except Exception as e:
        raise HTTPException(400, str(e))


@router.post("/consolidate")
async def consolidate(request: Request):
    try:
        return request.app.state.model.consolidate()
    except RuntimeError as e:
        raise HTTPException(400, str(e))


@router.post("/reset_memory")
async def reset_memory(request: Request):
    request.app.state.model.reset_memory()
    return {"ok": True}
