from __future__ import annotations

import asyncio
import json
import sqlite3
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, Optional

from .job_schemas import build_cli_cmd, validate_config


DEFAULT_TIMEOUTS = {
    "train": 24 * 3600,
    "lora": 6 * 3600,
    "rag_index": 1 * 3600,
    "reasoning": 30 * 60,
    "bpe_train": 1 * 3600,
}

FINAL_STATUSES = {"completed", "failed", "cancelled", "timed_out"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobQueue:
    def __init__(
        self,
        db_path: str = "state.db",
        log_dir: str = "logs",
        timeout_defaults: Optional[dict[str, int]] = None,
        poll_interval: float = 2.0,
    ):
        self.db_path = db_path
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._procs: dict[str, asyncio.subprocess.Process] = {}
        self._worker: Optional[asyncio.Task] = None
        self._poll_interval = poll_interval
        self._timeouts = {**DEFAULT_TIMEOUTS, **(timeout_defaults or {})}
        self._init_db()

    # ---------------------------------------------------------- db
    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    config TEXT NOT NULL,
                    cmd TEXT NOT NULL DEFAULT '[]',
                    status TEXT NOT NULL DEFAULT 'pending',
                    log_path TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    return_code INTEGER,
                    error TEXT
                )
            """)
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_status "
                "ON jobs(status, created_at)"
            )

    # ---------------------------------------------------------- submit
    def submit(self, job_type: str, config: dict) -> str:
        cfg = validate_config(job_type, config)
        cmd = build_cli_cmd(job_type, cfg)
        job_id = uuid.uuid4().hex[:10]
        log_path = str(self.log_dir / f"{job_id}.log")
        Path(log_path).touch()

        with sqlite3.connect(self.db_path) as c:
            c.execute(
                "INSERT INTO jobs (id, type, config, cmd, log_path, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    job_id, job_type, json.dumps(cfg),
                    json.dumps(cmd), log_path, _now(),
                ),
            )
        return job_id

    # ---------------------------------------------------------- read
    def list(
        self,
        limit: int = 100,
        offset: int = 0,
        job_type: Optional[str] = None,
        status: Optional[str] = None,
    ) -> list[dict]:
        q = "SELECT * FROM jobs"
        params: list = []
        where = []
        if job_type:
            where.append("type = ?")
            params.append(job_type)
        if status:
            where.append("status = ?")
            params.append(status)
        if where:
            q += " WHERE " + " AND ".join(where)
        q += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with sqlite3.connect(self.db_path) as c:
            c.row_factory = sqlite3.Row
            rows = c.execute(q, params).fetchall()
        return [dict(r) for r in rows]

    def count(self, status: Optional[str] = None) -> int:
        q = "SELECT COUNT(*) FROM jobs"
        params: list = []
        if status:
            q += " WHERE status = ?"
            params.append(status)
        with sqlite3.connect(self.db_path) as c:
            return int(c.execute(q, params).fetchone()[0])

    def get(self, job_id: str) -> Optional[dict]:
        with sqlite3.connect(self.db_path) as c:
            c.row_factory = sqlite3.Row
            r = c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return dict(r) if r else None

    def read_log(self, job_id: str, tail: int = 0) -> str:
        job = self.get(job_id)
        if not job:
            raise KeyError(job_id)
        p = Path(job["log_path"])
        if not p.exists():
            return ""
        text = p.read_text(errors="replace")
        if tail > 0:
            return "\n".join(text.splitlines()[-tail:])
        return text

    # ---------------------------------------------------------- control
    def cancel(self, job_id: str) -> bool:
        job = self.get(job_id)
        if not job:
            return False
        if job["status"] in FINAL_STATUSES:
            return False
        p = self._procs.get(job_id)
        if p and p.returncode is None:
            try:
                p.terminate()
            except ProcessLookupError:
                pass
            return True
        # pending: marcar cancelled
        self._set(job_id, status="cancelled", finished_at=_now(),
                  error="cancelled before start")
        return True

    def retry(self, job_id: str) -> str:
        old = self.get(job_id)
        if not old:
            raise KeyError(job_id)
        cfg = json.loads(old["config"])
        return self.submit(old["type"], cfg)

    def delete(self, job_id: str) -> bool:
        job = self.get(job_id)
        if not job:
            return False
        if job["status"] not in FINAL_STATUSES:
            raise RuntimeError("Cannot delete a running job; cancel first.")
        with sqlite3.connect(self.db_path) as c:
            c.execute("DELETE FROM jobs WHERE id=?", (job_id,))
        p = Path(job["log_path"])
        if p.exists():
            try:
                p.unlink()
            except OSError:
                pass
        return True

    # ---------------------------------------------------------- lifecycle
    async def start(self) -> None:
        # Reanudar jobs "running" del arranque anterior como failed
        with sqlite3.connect(self.db_path) as c:
            c.execute(
                "UPDATE jobs SET status='failed', "
                "finished_at=?, error='orphaned on restart' "
                "WHERE status='running'",
                (_now(),),
            )
        self._worker = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._worker:
            self._worker.cancel()
            try:
                await self._worker
            except (asyncio.CancelledError, Exception):
                pass
        for p in list(self._procs.values()):
            if p.returncode is None:
                try:
                    p.terminate()
                except ProcessLookupError:
                    pass
        await asyncio.sleep(0.5)
        for p in list(self._procs.values()):
            if p.returncode is None:
                try:
                    p.kill()
                except ProcessLookupError:
                    pass
        self._procs.clear()

    async def _loop(self) -> None:
        while True:
            try:
                await self._tick()
            except asyncio.CancelledError:
                return
            except Exception as e:
                print(f"[jobs] worker error: {e}")
            await asyncio.sleep(self._poll_interval)

    async def _tick(self) -> None:
        with sqlite3.connect(self.db_path) as c:
            c.row_factory = sqlite3.Row
            r = c.execute(
                "SELECT * FROM jobs WHERE status='pending' "
                "ORDER BY created_at LIMIT 1"
            ).fetchone()
        if not r:
            return
        job = dict(r)
        self._set(
            job["id"], status="running", started_at=_now(),
            error=None, return_code=None,
        )
        job["status"] = "running"
        try:
            await self._run(job)
            self._set(
                job["id"], status="completed", finished_at=_now(),
                return_code=0, error=None,
            )
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            self._set(
                job["id"], status="timed_out", finished_at=_now(),
                error=f"timeout after {self._timeouts.get(job['type'])}s",
            )
        except Exception as e:
            self._set(
                job["id"], status="failed", finished_at=_now(),
                error=str(e)[:1000],
            )

    async def _run(self, job: dict) -> None:
        cmd = json.loads(job["cmd"])
        timeout = self._timeouts.get(job["type"], 3600)

        with open(job["log_path"], "ab") as logf:
            logf.write(f"\n=== {_now()} :: {' '.join(cmd)} ===\n".encode())
            logf.flush()
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=logf, stderr=subprocess.STDOUT,
            )
            self._procs[job["id"]] = proc
            try:
                rc = await asyncio.wait_for(proc.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                raise
            finally:
                self._procs.pop(job["id"], None)

        if rc != 0:
            raise RuntimeError(f"exit code {rc}")

    # ---------------------------------------------------------- stream
    async def stream_log(self, job_id: str) -> AsyncIterator[str]:
        job = self.get(job_id)
        if not job:
            raise KeyError(job_id)
        path = Path(job["log_path"])
        pos = 0
        while True:
            if path.exists():
                size = path.stat().st_size
                if size > pos:
                    with open(path, "rb") as f:
                        f.seek(pos)
                        chunk = f.read(size - pos)
                    if chunk:
                        pos = size
                        yield chunk.decode("utf-8", errors="replace")
            job = self.get(job_id)
            if job is None:
                yield "\n[job deleted]\n"
                return
            if job["status"] in FINAL_STATUSES:
                # drenar remanente
                if path.exists():
                    size = path.stat().st_size
                    if size > pos:
                        with open(path, "rb") as f:
                            f.seek(pos)
                            yield f.read(size - pos).decode("utf-8", errors="replace")
                return
            await asyncio.sleep(0.5)

    # ---------------------------------------------------------- util
    def _set(self, job_id: str, **kw) -> None:
        if not kw:
            return
        cols = ", ".join(f"{k}=?" for k in kw)
        with sqlite3.connect(self.db_path) as c:
            c.execute(
                f"UPDATE jobs SET {cols} WHERE id=?",
                (*kw.values(), job_id),
            )

    def stats(self) -> dict:
        with sqlite3.connect(self.db_path) as c:
            rows = c.execute(
                "SELECT status, COUNT(*) FROM jobs GROUP BY status"
            ).fetchall()
        return {s: n for s, n in rows}
