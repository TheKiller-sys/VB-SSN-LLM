import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .model_manager import ModelManager
from .jobs import JobQueue
from .routers import chat, model, training, rag, reasoning, files, admin


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = ModelManager()
    app.state.jobs = JobQueue("state.db")
    await app.state.jobs.start()

    # Auto-cargar el "best.pt" si existe
    auto = Path("data/checkpoints/best.pt")
    if auto.exists():
        try:
            app.state.model.load(str(auto))
        except Exception as e:
            print(f"[startup] auto-load failed: {e}")

    yield

    await app.state.jobs.stop()


app = FastAPI(title="SNN-LM Dashboard", lifespan=lifespan)

app.include_router(chat.router,      prefix="/api/chat",     tags=["chat"])
app.include_router(model.router,     prefix="/api/model",    tags=["model"])
app.include_router(training.router,  prefix="/api/train",    tags=["train"])
app.include_router(rag.router,       prefix="/api/rag",      tags=["rag"])
app.include_router(reasoning.router, prefix="/api/reason",   tags=["reason"])
app.include_router(files.router,     prefix="/api/files",    tags=["files"])
app.include_router(admin.router,     prefix="/api/admin",    tags=["admin"])

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
