"""Async VLM worker: drains clothing_jobs, runs InternVL, writes appearances.

    F["Sync pipeline"] -. crop + person_id .-> H["clothing_jobs table"]
    H --> I["Async VLM worker\nbounded GPU concurrency"]
    I --> G[("PostgreSQL\npersons / segments / camera_events")]

finalize.py enqueues a 'pending' clothing_jobs row (fast INSERT, no VLM call) once
GlobalMemory.register() resolves a person_id for a segment -- see
forensics/global_memory/store.py::insert_clothing_job. This module is the other half:
a standalone, long-running worker that polls clothing_jobs, runs the clothing
description model (reusing nodes/describe_clothing.py's model-loading /
crop-reading / InternVL-call code, not duplicating it), and writes results into
`appearances` keyed by person_id/segment_id.

Design constraints from the realtime architecture:
- Bounded GPU concurrency: a single-threaded poll loop *is* the hard cap of 1
  in-flight VLM inference -- there is no thread pool here, jobs are processed one at
  a time, deliberately. A dedicated CUDA stream (when CUDA is available) keeps this
  worker's inference from contending with the sync detection pipeline's default
  stream for GPU cycles.
- Not latency-sensitive: a backlog draining later is fine, so the poll interval and
  retry backoff are generous by default.
- Same reliability shape as the segment state machine: pending -> processing ->
  done, or failed_retryable (retried with backoff, up to CLOTHING_JOB_MAX_ATTEMPTS)
  -> failed_final. See global_memory/store.py::claim_next_clothing_job /
  mark_clothing_job_done / mark_clothing_job_failed.
"""

from __future__ import annotations

import threading
from datetime import date as _date

from forensics.global_memory import config as gm_config
from forensics.global_memory.store import GlobalMemory
from forensics.person_creation.models.clothing_describer import get_clothing_describer
from forensics.person_creation.nodes.describe_clothing import _FALLBACK, _describe_paths


class VLMWorker:
    def __init__(
        self,
        dsn: str | None = None,
        poll_interval_seconds: float = 2.0,
        max_attempts: int | None = None,
        backoff_seconds: tuple[float, ...] | None = None,
        model_id: str | None = None,
        device: str = "auto",
    ) -> None:
        self.db_path = dsn  # kept as .db_path for backward-compat attribute access; holds a Postgres DSN override, not a filesystem path
        self.poll_interval_seconds = max(0.1, float(poll_interval_seconds))
        self.max_attempts = gm_config.CLOTHING_JOB_MAX_ATTEMPTS if max_attempts is None else max_attempts
        self.backoff_seconds = backoff_seconds or gm_config.CLOTHING_JOB_RETRY_BACKOFF_SECONDS
        self.model_id = model_id
        self.device = device

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._cuda_stream = None

        self.jobs_processed = 0
        self.jobs_succeeded = 0
        self.jobs_failed = 0

    # -- lifecycle ------------------------------------------------------------

    def start(self) -> None:
        self._ensure_model_loaded()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="vlm-worker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def run_once(self) -> bool:
        """Process at most one pending job synchronously. Used by tests and by
        `_run` internally; returns False if there was nothing to do.
        """
        self._ensure_model_loaded()
        gm = GlobalMemory(self.db_path)
        try:
            job = gm.claim_next_clothing_job()
            if job is None:
                return False
            self._process_job(gm, job)
            return True
        finally:
            gm.close()

    # -- internals --------------------------------------------------------------

    def _ensure_model_loaded(self) -> None:
        describer = get_clothing_describer()
        if not describer.is_loaded():
            kwargs = {"device": self.device}
            if self.model_id:
                kwargs["model_id"] = self.model_id
            describer.load(**kwargs)

        if self._cuda_stream is None:
            try:
                import torch

                if torch.cuda.is_available():
                    self._cuda_stream = torch.cuda.Stream()
            except Exception:
                self._cuda_stream = None

    def _run(self) -> None:
        gm = GlobalMemory(self.db_path)
        try:
            while not self._stop_event.is_set():
                job = gm.claim_next_clothing_job()
                if job is None:
                    self._stop_event.wait(self.poll_interval_seconds)
                    continue
                self._process_job(gm, job)
        finally:
            gm.close()

    def _process_job(self, gm: GlobalMemory, job: dict) -> None:
        job_id = job["job_id"]
        person_id = job["person_id"]
        segment_id = job["segment_id"]
        crop_path = job["crop_path"]
        self.jobs_processed += 1

        try:
            describer = get_clothing_describer()
            if self._cuda_stream is not None:
                import torch

                with torch.cuda.stream(self._cuda_stream):
                    raw, structured = _describe_paths(describer, [crop_path])
                torch.cuda.current_stream().wait_stream(self._cuda_stream)
            else:
                raw, structured = _describe_paths(describer, [crop_path])

            if not raw and structured == _FALLBACK:
                raise RuntimeError("clothing description produced no result (unreadable crop or VLM failure)")

            gm.upsert_appearance_for_segment(
                person_id=person_id,
                segment_id=segment_id,
                date=_date.today().isoformat(),
                top=structured.get("top"),
                bottom=structured.get("bottom"),
                shoes=structured.get("shoes"),
                full_description=structured.get("full") or raw,
                best_body_crops=[crop_path],
            )
            gm.mark_clothing_job_done(job_id)
            self.jobs_succeeded += 1
            print(f"[vlm_worker] job={job_id} person={person_id} segment={segment_id} -> done")
        except Exception as exc:  # noqa: BLE001 - worker must never crash the loop
            status = gm.mark_clothing_job_failed(
                job_id, str(exc), max_attempts=self.max_attempts, backoff_seconds=self.backoff_seconds,
            )
            self.jobs_failed += 1
            print(f"[vlm_worker] job={job_id} person={person_id} segment={segment_id} -> {status} ({exc})")


def main() -> None:
    """Standalone entrypoint: python -m forensics.person_creation.vlm_worker"""
    import signal

    worker = VLMWorker()
    worker.start()
    print("[vlm_worker] started, polling clothing_jobs. Ctrl+C to stop.")

    stop_event = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop_event.set())
    signal.signal(signal.SIGTERM, lambda *_: stop_event.set())
    stop_event.wait()

    print("[vlm_worker] stopping...")
    worker.stop()
    print(
        f"[vlm_worker] stopped. processed={worker.jobs_processed} "
        f"succeeded={worker.jobs_succeeded} failed={worker.jobs_failed}"
    )


if __name__ == "__main__":
    main()
