import asyncio
import json
import sqlite3
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path


class JobQueue:
    def __init__(self, db_path: str = "state.db", log_dir: str = "logs"):
        self.db_path = db_path
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self._procs: dict[str, asyncio.subprocess.Process] = {}
        self._worker: asyncio.Task | None = None
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    config TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    log_path TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    return_code INTEGER,
                    error TEXT
                )
            """)

    def submit(self, job_type: str, config: dict) -> str:
        job_id = uuid.uuid4().hex[:10]
        log_path = str(self.log_dir / f"{job_id}.log")
        Path(log_path).touch()
        with sqlite3.connect(self.db_path) as c:
            c.execute(
                "INSERT INTO jobs (id, type, config, log_path, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (job_id, job_type, json.dumps(config), log_path,
                 datetime.now(timezone.utc).isoformat()),
            )
        return job_id

    def list(self, limit: int = 50) -> list[dict]:
        with sqlite3.connect(self.db_path) as c:
            c.row_factory = sqlite3.Row
            rows = c.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get(self, job_id: str) -> dict | None:
        with sqlite3.connect(self.db_path) as c:
            c.row_factory = sqlite3.Row
            r = c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return dict(r) if r else None

    def cancel(self, job_id: str) -> bool:
        p = self._procs.get(job_id)
        if p and p.returncode is None:
            p.kill()
            return True
        return False

    async def start(self):
        self._worker = asyncio.create_task(self._loop())

    async def stop(self):
        if self._worker:
            self._worker.cancel()
        for p in self._procs.values():
            if p.returncode is None:
                p.kill()

    async def _loop(self):
        while True:
            try:
                await self._tick()
            except Exception as e:
                print(f"[jobs] worker error: {e}")
            await asyncio.sleep(2)

    async def _tick(self):
        with sqlite3.connect(self.db_path) as c:
            c.row_factory = sqlite3.Row
            r = c.execute(
                "SELECT * FROM jobs WHERE status='pending' "
                "ORDER BY created_at LIMIT 1"
            ).fetchone()
        if not r:
            return
        job = dict(r)
        self._set(job["id"], status="running",
                  started_at=datetime.now(timezone.utc).isoformat())
        try:
            await self._run(job)
            self._set(job["id"], status="completed",
                      finished_at=datetime.now(timezone.utc).isoformat(),
                      return_code=0)
        except Exception as e:
            self._set(job["id"], status="failed",
                      finished_at=datetime.now(timezone.utc).isoformat(),
                      error=str(e)[:500])

    async def _run(self, job: dict):
        cfg = json.loads(job["config"])
        cmd = self._build_cmd(job["type"], cfg)
        with open(job["log_path"], "wb") as logf:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=logf, stderr=subprocess.STDOUT,
            )
            self._procs[job["id"]] = proc
            rc = await proc.wait()
            self._procs.pop(job["id"], None)
        if rc != 0:
            raise RuntimeError(f"exit code {rc}")

    def _build_cmd(self, job_type: str, cfg: dict) -> list[str]:
        if job_type == "train":
            cmd = ["python", "scripts/train.py", *cfg["data"]]
            mapping = {
                "--size": cfg.get("size"),
                "--core": cfg.get("core"),
                "--ffn-mode": cfg.get("ffn_mode"),
                "--snn-mode": cfg.get("snn_mode"),
                "--max-steps": cfg.get("max_steps"),
                "--batch-size": cfg.get("batch_size"),
                "--seq-len": cfg.get("seq_len"),
                "--lr": cfg.get("lr"),
                "--out": cfg.get("out"),
                "--device": cfg.get("device", "auto"),
            }
            for flag, val in mapping.items():
                if val is not None:
                    cmd.extend([flag, str(val)])
            if cfg.get("resume"):
                cmd.extend(["--resume", cfg["resume"]])
            return cmd

        if job_type == "lora":
            return [
                "python", "scripts/train_lora.py", *cfg["data"],
                "--base", cfg["base"],
                "--out", cfg["out"],
                "--max-steps", str(cfg.get("max_steps", 1000)),
            ]

        if job_type == "rag_index":
            return [
                "python", "scripts/add_knowledge.py",
                "--corpus", *cfg["corpus"],
                "--out", cfg["out"],
                "--encoder", "hidden",
                "--ckpt", cfg["ckpt"],
            ]

        if job_type == "reasoning":
            return [
                "python", "scripts/reason.py",
                "--corpus", cfg["corpus"],
                "--mine",
                "--save", cfg["out"],
            ]

        raise ValueError(f"Unknown job type: {job_type}")

    def _set(self, job_id: str, **kw):
        if not kw:
            return
        cols = ", ".join(f"{k}=?" for k in kw)
        with sqlite3.connect(self.db_path) as c:
            c.execute(f"UPDATE jobs SET {cols} WHERE id=?",
                      (*kw.values(), job_id))
