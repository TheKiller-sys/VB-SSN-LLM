from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .model_manager import ModelManager
from .jobs import JobQueue
from .routers import (
    chat, model, training, rag, reasoning, files, admin,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = ModelManager()
    app.state.jobs = JobQueue(
        os.environ.get("SNNLM_STATE_DB", "state.db"),
        os.environ.get("SNNLM_LOG_ROOT", "logs"),
    )
    app.state.graph_cache = {}
    app.state.analyzer = None
    await app.state.jobs.start()

    auto = Path("data/checkpoints/best.pt")
    if auto.exists():
        try:
            app.state.model.load(str(auto))
            print(f"[startup] auto-loaded {auto}")
        except Exception as e:
            print(f"[startup] auto-load failed: {e}")

    yield

    await app.state.jobs.stop()


app = FastAPI(
    title="SNN-LM v4.3 Dashboard",
    version="4.3.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "type": type(exc).__name__},
    )


app.include_router(chat.router,      prefix="/api/chat",     tags=["chat"])
app.include_router(model.router,     prefix="/api/model",    tags=["model"])
app.include_router(training.router,  prefix="/api/train",    tags=["train"])
app.include_router(rag.router,       prefix="/api/rag",      tags=["rag"])
app.include_router(reasoning.router, prefix="/api/reason",   tags=["reason"])
app.include_router(files.router,     prefix="/api/files",    tags=["files"])
app.include_router(admin.router,     prefix="/api/admin",    tags=["admin"])

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
