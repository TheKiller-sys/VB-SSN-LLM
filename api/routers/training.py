from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse, PlainTextResponse

from ..job_schemas import JOB_SCHEMAS
from ..schemas import JobSubmit

router = APIRouter()


@router.get("/schema")
async def schemas():
    return JOB_SCHEMAS


@router.get("/schema/{job_type}")
async def schema(job_type: str):
    s = JOB_SCHEMAS.get(job_type)
    if not s:
        raise HTTPException(404, f"Unknown job_type: {job_type}")
    return s


@router.post("/jobs")
async def submit(req: JobSubmit, request: Request):
    q = request.app.state.jobs
    try:
        job_id = q.submit(req.type, req.config)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"id": job_id, "job": q.get(job_id)}


@router.get("/jobs")
async def list_jobs(
    request: Request,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    type: str | None = None,
    status: str | None = None,
):
    return {"jobs": request.app.state.jobs.list(limit, offset, type, status)}


@router.get("/jobs/{job_id}")
async def get_job(job_id: str, request: Request):
    job = request.app.state.jobs.get(job_id)
    if not job:
        raise HTTPException(404, job_id)
    return job


@router.post("/jobs/{job_id}/cancel")
async def cancel(job_id: str, request: Request):
    ok = request.app.state.jobs.cancel(job_id)
    return {"ok": ok}


@router.post("/jobs/{job_id}/retry")
async def retry(job_id: str, request: Request):
    try:
        new_id = request.app.state.jobs.retry(job_id)
    except KeyError:
        raise HTTPException(404, job_id)
    return {"id": new_id, "job": request.app.state.jobs.get(new_id)}


@router.delete("/jobs/{job_id}")
async def delete(job_id: str, request: Request):
    try:
        request.app.state.jobs.delete(job_id)
    except KeyError:
        raise HTTPException(404, job_id)
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    return {"ok": True}


@router.get("/jobs/{job_id}/log")
async def log(
    job_id: str, request: Request,
    tail: int = Query(0, ge=0),
):
    try:
        text = request.app.state.jobs.read_log(job_id, tail=tail)
    except KeyError:
        raise HTTPException(404, job_id)
    return PlainTextResponse(text)


@router.get("/jobs/{job_id}/log/stream")
async def stream_log(job_id: str, request: Request):
    q = request.app.state.jobs
    if not q.get(job_id):
        raise HTTPException(404, job_id)

    async def gen():
        try:
            async for chunk in q.stream_log(job_id):
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@router.get("/stats")
async def stats(request: Request):
    return request.app.state.jobs.stats()
