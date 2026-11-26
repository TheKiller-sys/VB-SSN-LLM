from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from ..schemas import ChatReq

router = APIRouter()


@router.post("/")
async def chat(req: ChatReq, request: Request):
    mgr = request.app.state.model
    if not mgr.info().get("loaded"):
        raise HTTPException(400, "No model loaded")

    use_rag = req.use_rag
    use_bias = req.use_reasoning_bias
    if use_rag and mgr._rag_kb is None:
        raise HTTPException(400, "RAG requested but no KB loaded")
    if use_bias and mgr._reasoning_bias is None:
        raise HTTPException(400, "Reasoning bias requested but not loaded")

    async def gen():
        loop = asyncio.get_event_loop()
        try:
            text = await loop.run_in_executor(
                None,
                lambda: mgr.generate(
                    req.prompt,
                    max_new_tokens=req.max_tokens,
                    temperature=req.temperature,
                    top_k=req.top_k,
                    top_p=req.top_p,
                    rag_top_k=req.rag_k,
                ),
            )
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
            return
        for i in range(0, len(text), 4):
            chunk = text[i:i + 4]
            yield f"event: token\ndata: {json.dumps({'text': chunk})}\n\n"
            await asyncio.sleep(0.02)
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@router.post("/verify")
async def verify(req: ChatReq, request: Request):
    mgr = request.app.state.model
    if not mgr.info().get("loaded"):
        raise HTTPException(400, "No model loaded")
    try:
        return mgr.verify(
            req.prompt,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
        )
    except Exception as e:
        raise HTTPException(400, str(e))
