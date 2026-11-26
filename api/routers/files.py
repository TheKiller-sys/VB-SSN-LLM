from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse

from ..schemas import FileMkdirReq

router = APIRouter()

BASE = Path("data").resolve()


def _safe(rel: str) -> Path:
    target = (BASE / rel).resolve()
    if not str(target).startswith(str(BASE)):
        raise HTTPException(400, "path escapes base")
    return target


@router.get("/list")
async def list_dir(path: str = ""):
    p = _safe(path)
    if not p.exists():
        raise HTTPException(404, path)
    if p.is_file():
        return {"type": "file", "path": str(p.relative_to(BASE)),
                "size": p.stat().st_size}
    entries = []
    for c in sorted(p.iterdir()):
        try:
            entries.append({
                "name": c.name,
                "type": "dir" if c.is_dir() else "file",
                "size": c.stat().st_size if c.is_file() else 0,
            })
        except OSError:
            pass
    return {
        "type": "dir",
        "path": str(p.relative_to(BASE)),
        "entries": entries,
    }


@router.get("/read")
async def read_file(path: str, max_bytes: int = Query(200_000, ge=1)):
    p = _safe(path)
    if not p.exists() or not p.is_file():
        raise HTTPException(404, path)
    data = p.read_bytes()[:max_bytes]
    return PlainTextResponse(data.decode("utf-8", errors="replace"))


@router.get("/download")
async def download(path: str):
    p = _safe(path)
    if not p.exists() or not p.is_file():
        raise HTTPException(404, path)
    return FileResponse(p, filename=p.name)


@router.post("/upload")
async def upload(file: UploadFile = File(...), path: str = Form("")):
    target_dir = _safe(path)
    if target_dir.is_file():
        raise HTTPException(400, "path is a file")
    target_dir.mkdir(parents=True, exist_ok=True)
    dest = target_dir / file.filename
    with open(dest, "wb") as f:
        while True:
            chunk = await file.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    return {"ok": True, "path": str(dest.relative_to(BASE)),
            "size": dest.stat().st_size}


@router.post("/mkdir")
async def mkdir(req: FileMkdirReq):
    p = _safe(req.path)
    p.mkdir(parents=True, exist_ok=True)
    return {"ok": True}


@router.delete("/delete")
async def delete(path: str):
    p = _safe(path)
    if not p.exists():
        raise HTTPException(404, path)
    if p.is_dir():
        shutil.rmtree(p)
    else:
        p.unlink()
    return {"ok": True}


@router.get("/disk")
async def disk():
    import os
    total, used, free = shutil.disk_usage(str(BASE))
    return {"total": total, "used": used, "free": free}
