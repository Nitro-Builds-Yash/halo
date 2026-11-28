"""Background ingestion jobs.

Indexing large or scanned PDFs can take minutes. `submit` runs ingestion on a single worker
thread (so index writes never race) and keeps per-job progress that the API exposes at
/api/v1/jobs/{id}. Jobs live in memory for this process; the newest MAX_JOBS are kept.
"""
import shutil
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional

from halo.core.log import get_logger

logger = get_logger(__name__)

MAX_JOBS = 50


@dataclass
class Job:
    id: str
    files: List[str]
    status: str = "queued"           # queued | running | done | failed
    total: int = 0
    done: int = 0
    current: Optional[str] = None
    pages: int = 0
    chunks: int = 0
    failed_files: List[str] = field(default_factory=list)
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["progress"] = round(self.done / self.total, 3) if self.total else 0.0
        return d


class JobRegistry:
    def __init__(self, ingest: Callable, on_finish: Optional[Callable[[], None]] = None):
        self._ingest = ingest
        self._on_finish = on_finish
        self._jobs: "OrderedDict[str, Job]" = OrderedDict()
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="halo-ingest")

    def submit(self, paths: List[str], names: List[str], cleanup_dir: Optional[str] = None) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], files=list(names), total=len(paths))
        with self._lock:
            self._jobs[job.id] = job
            while len(self._jobs) > MAX_JOBS:
                self._jobs.popitem(last=False)
        self._pool.submit(self._run, job, paths, names, cleanup_dir)
        return job

    def _progress(self, job: Job, event: str, name: str, info: Optional[str] = None) -> None:
        with self._lock:
            if event == "start":
                job.current = name
            elif event in ("done", "failed"):
                job.done += 1
                job.current = None
                if event == "failed":
                    job.failed_files.append(f"{name}: {info or 'could not be read'}")

    def _run(self, job: Job, paths: List[str], names: List[str], cleanup_dir: Optional[str]) -> None:
        job.status = "running"
        try:
            pages, chunks = self._ingest(paths, names, progress=lambda e, n, i=None: self._progress(job, e, n, i))
            job.pages, job.chunks = pages, chunks
            job.status = "failed" if job.failed_files and not chunks else "done"
            if job.status == "failed":
                job.error = "No file could be indexed."
        except Exception as e:  # keep the worker alive
            logger.exception("Ingestion job %s failed", job.id)
            # Exception text can carry provider details (keys, URLs): keep it in the logs only.
            job.status, job.error = "failed", f"Indexing failed ({type(e).__name__}). Check the server logs."
        finally:
            job.finished_at = time.time()
            job.current = None
            if cleanup_dir:
                shutil.rmtree(cleanup_dir, ignore_errors=True)
            if self._on_finish and job.chunks:
                try:
                    self._on_finish()
                except Exception:
                    logger.exception("on_finish hook failed")

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def recent(self, limit: int = 20) -> List[Job]:
        with self._lock:
            return list(self._jobs.values())[-limit:][::-1]

    def wait(self, job_id: str, timeout: float = 30.0) -> Optional[Job]:
        """Block until a job finishes (used by tests and the CLI)."""
        end = time.time() + timeout
        while time.time() < end:
            job = self.get(job_id)
            if job is None or job.status in ("done", "failed"):
                return job
            time.sleep(0.02)
        return self.get(job_id)
