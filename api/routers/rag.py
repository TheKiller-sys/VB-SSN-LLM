from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from ..schemas import RagAddReq, RagLoadReq, RagSearchReq

router = APIRouter()


@router.get("/kbs")
async def list_kbs():
    root = Path("data/checkpoints")
    if not root.exists():
        return {"kbs": []}
    return {
        "kbs": [
            {"path": str(p), "name": p.name, "size": p.stat().st_size}
            for p in sorted(root.rglob("*.pt"))
            if "kb" in p.name.lower()
        ]
    }


@router.post("/load")
async def load(req: RagLoadReq, request: Request):
    p = Path(req.path)
    if not p.exists():
        raise HTTPException(404, req.path)
    try:
        n = request.app.state.model.load_rag_from_path(str(p))
    except Exception as e:
        raise HTTPException(400, str(e))
    return {"ok": True, "docs": n}


@router.post("/unload")
async def unload(request: Request):
    request.app.state.model.clear_rag()
    return {"ok": True}


@router.post("/search")
async def search(req: RagSearchReq, request: Request):
    mgr = request.app.state.model
    if mgr._rag_kb is None:
        raise HTTPException(400, "No RAG KB loaded")
    try:
        return {"results": mgr.rag_search(req.query, k=req.k)}
    except Exception as e:
        raise HTTPException(400, str(e))


@router.post("/build")
async def build(req: RagAddReq, request: Request):
    q = request.app.state.jobs
    cfg = {
        "corpus": req.corpus,
        "chunk_size": req.chunk_size,
        "chunk_overlap": req.chunk_overlap,
        "encoder": req.encoder if not req.ckpt else "hidden",
    }
    if req.kb_path_used(req):
        pass
    if req.ckpt:
        cfg["ckpt"] = req.ckpt
    # Si kb y out: build new; si solo kb: extend
    if req.encoder == "sentence_transformer":
        cfg["encoder"] = "sentence_transformer:all-MiniLM-L6-v2"
    job_id = q.submit("rag_index", {
        **cfg,
        "kb": getattr(req, "kb", None),
        "out": getattr(req, "out", None),
    })
    return {"id": job_id, "job": q.get(job_id)}
