import asyncio
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

router = APIRouter()


class ChatReq(BaseModel):
    prompt: str
    max_tokens: int = 200
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.95


@router.post("/")
async def chat(req: ChatReq, request: Request):
    mgr = request.app.state.model
    if not mgr.info()["loaded"]:
        return {"error": "No model loaded"}

    async def gen():
        # generate() no es streaming; lo corremos en executor
        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(
            None,
            lambda: mgr.generate(
                req.prompt,
                max_new_tokens=req.max_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
            ),
        )
        # Simular streaming para que el frontend lo consuma uniformemente
        for chunk in _chunk(text, 4):
            yield f"data: {chunk}\n\n"
            await asyncio.sleep(0.02)
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


def _chunk(text: str, n: int):
    for i in range(0, len(text), n):
        yield text[i:i + n].replace("\n", "\\n")
