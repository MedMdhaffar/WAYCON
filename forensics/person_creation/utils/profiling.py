from __future__ import annotations

import json
import os
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any

try:
    import psutil
except ImportError:
    psutil = None

try:
    import torch
except ImportError:
    torch = None


class PipelineProfiler:
    """
    Thread-safe profiler for pipeline stages.

    Records:
    - wall-clock elapsed time
    - CPU process usage
    - RAM usage
    - GPU allocated/reserved VRAM
    """

    def __init__(self) -> None:
        self._records: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    @staticmethod
    def _system_metrics() -> dict[str, Any]:
        metrics: dict[str, Any] = {}

        if psutil is not None:
            process = psutil.Process(os.getpid())

            metrics["process_rss_mb"] = (
                process.memory_info().rss / 1024**2
            )
            metrics["process_cpu_percent"] = process.cpu_percent(
                interval=None
            )

        if (
            torch is not None
            and torch.cuda.is_available()
        ):
            device = torch.cuda.current_device()

            metrics["gpu_allocated_mb"] = (
                torch.cuda.memory_allocated(device) / 1024**2
            )
            metrics["gpu_reserved_mb"] = (
                torch.cuda.memory_reserved(device) / 1024**2
            )
            metrics["gpu_peak_allocated_mb"] = (
                torch.cuda.max_memory_allocated(device) / 1024**2
            )

        return metrics

    @contextmanager
    def measure(
        self,
        name: str,
        metadata: dict[str, Any] | None = None,
        synchronize_cuda: bool = True,
    ):
        if (
            synchronize_cuda
            and torch is not None
            and torch.cuda.is_available()
        ):
            torch.cuda.synchronize()

        before_metrics = self._system_metrics()
        start = time.perf_counter()

        error: str | None = None

        try:
            yield
        except Exception as exc:
            error = repr(exc)
            raise
        finally:
            if (
                synchronize_cuda
                and torch is not None
                and torch.cuda.is_available()
            ):
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start
            after_metrics = self._system_metrics()

            record = {
                "name": name,
                "elapsed_seconds": elapsed,
                "before": before_metrics,
                "after": after_metrics,
                "metadata": metadata or {},
                "error": error,
            }

            with self._lock:
                self._records.append(record)

            print(
                f"[PROFILE] {name}: "
                f"{elapsed:.4f} seconds"
            )

    def add_record(
        self,
        name: str,
        elapsed_seconds: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        record = {
            "name": name,
            "elapsed_seconds": elapsed_seconds,
            "metadata": metadata or {},
        }

        with self._lock:
            self._records.append(record)

    def summary(self) -> dict[str, Any]:
        grouped: dict[str, list[float]] = defaultdict(list)

        with self._lock:
            records = list(self._records)

        for record in records:
            grouped[record["name"]].append(
                record["elapsed_seconds"]
            )

        stage_summary: dict[str, Any] = {}

        for name, values in grouped.items():
            stage_summary[name] = {
                "calls": len(values),
                "total_seconds": sum(values),
                "average_seconds": sum(values) / len(values),
                "minimum_seconds": min(values),
                "maximum_seconds": max(values),
            }

        return {
            "records": records,
            "summary": stage_summary,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as file:
            json.dump(
                self.summary(),
                file,
                indent=2,
                default=str,
            )


profiler = PipelineProfiler()