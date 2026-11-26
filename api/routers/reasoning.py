from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from ..schemas import GraphApplyReq, GraphLoadReq, GraphStatsReq

router = APIRouter()


def _load_graph_cached(request: Request, path: str):
    cache = request.app.state.graph_cache
    if path not in cache:
        from snn_lm_v3 import load_graph
        cache[path] = load_graph(path)
    return cache[path]


@router.get("/graphs")
async def list_graphs():
    root = Path("data/checkpoints")
    if not root.exists():
        return {"graphs": []}
    return {
        "graphs": [
            {"path": str(p), "name": p.name, "size": p.stat().st_size}
            for p in sorted(root.rglob("*.pt"))
            if "graph" in p.name.lower()
        ]
    }


@router.post("/load")
async def load_graph_endpoint(req: GraphLoadReq, request: Request):
    p = Path(req.path)
    if not p.exists():
        raise HTTPException(404, req.path)
    try:
        g = _load_graph_cached(request, str(p))
    except Exception as e:
        raise HTTPException(400, str(e))
    return g.get_state()


@router.get("/state")
async def graph_state(request: Request, path: str | None = None):
    cache = request.app.state.graph_cache
    if not cache:
        return {"loaded": False}
    if path:
        g = cache.get(path)
    else:
        g = next(iter(cache.values()))
    if g is None:
        raise HTTPException(404, "graph not loaded")
    return {"loaded": True, "state": g.get_state()}


@router.post("/mine")
async def mine(request: Request):
    q = request.app.state.jobs
    # req: {corpus, out, chunk_size}
    from fastapi import Request as _R  # noqa
    body = await request.json()
    cfg = {
        "corpus": body["corpus"],
        "out": body["out"],
        "chunk_size": body.get("chunk_size", 500),
    }
    job_id = q.submit("reasoning", cfg)
    return {"id": job_id, "job": q.get(job_id)}


@router.post("/apply")
async def apply(req: GraphApplyReq, request: Request):
    if req.path:
        g = _load_graph_cached(request, req.path)
    elif request.app.state.model._graph is not None:
        g = request.app.state.model._graph
    else:
        raise HTTPException(400, "No graph loaded")
    return {"consequences": g.apply(req.facts)}


@router.post("/stats")
async def stats(req: GraphStatsReq, request: Request):
    if req.path:
        g = _load_graph_cached(request, req.path)
    elif request.app.state.model._graph is not None:
        g = request.app.state.model._graph
    else:
        raise HTTPException(400, "No graph loaded")
    state = g.get_state()
    return {
        "state": {k: v for k, v in state.items() if k not in ("top_rules",)},
        "top_rules": state.get("top_rules", [])[: req.top],
    }


@router.post("/bias/load")
async def load_bias(req: GraphLoadReq, request: Request):
    # espera graph_path + mapper_path en body
    body = await request.json()
    try:
        state = request.app.state.model.load_reasoning_from_paths(
            graph_path=body["graph_path"],
            mapper_path=body["mapper_path"],
            strength=body.get("strength", 0.1),
            top_k=body.get("top_k", 16),
            remine_every=body.get("remine_every", 8),
            facts_window=body.get("facts_window", 8),
            min_confidence=body.get("min_confidence", 0.5),
        )
    except Exception as e:
        raise HTTPException(400, str(e))
    return state


@router.post("/bias/unload")
async def unload_bias(request: Request):
    request.app.state.model.clear_reasoning()
    return {"ok": True}
