from __future__ import annotations

import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import threading
import time
import traceback
import uuid
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

try:
    import psutil
except ImportError:  # pragma: no cover - environment dependent
    psutil = None

try:
    import torch
except ImportError:  # pragma: no cover - environment dependent
    torch = None

try:
    import pynvml
except ImportError:  # pragma: no cover - optional dependency
    pynvml = None


_MB = 1024**2
_active_profiler: ContextVar["PipelineProfiler | None"] = ContextVar(
    "person_creation_profiler", default=None
)
_history_lock = threading.Lock()
_sequence_lock = threading.Lock()
_job_sequence = 0
_completed_jobs = 0


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _next_job_sequence() -> int:
    global _job_sequence
    with _sequence_lock:
        _job_sequence += 1
        return _job_sequence


def _mark_job_completed() -> int:
    global _completed_jobs
    with _sequence_lock:
        _completed_jobs += 1
        return _completed_jobs


@dataclass(frozen=True)
class ProfilingConfig:
    enabled: bool = False
    cuda_sync: bool = False
    resource_interval_seconds: float = 1.0
    verbose: bool = False
    frame_level: bool = False
    sql: bool = False

    @classmethod
    def from_env(cls) -> ProfilingConfig:
        try:
            interval = max(
                0.1, float(os.getenv("PERSON_CREATION_PROFILE_RESOURCE_INTERVAL", "1.0"))
            )
        except ValueError:
            interval = 1.0
        return cls(
            enabled=_env_flag("PERSON_CREATION_PROFILE"),
            cuda_sync=_env_flag("PERSON_CREATION_PROFILE_CUDA_SYNC"),
            resource_interval_seconds=interval,
            verbose=_env_flag("PERSON_CREATION_PROFILE_VERBOSE"),
            frame_level=_env_flag("PERSON_CREATION_PROFILE_FRAME_LEVEL"),
            sql=_env_flag("PERSON_CREATION_PROFILE_SQL"),
        )


def _cuda_available() -> bool:
    try:
        return torch is not None and bool(torch.cuda.is_available())
    except Exception:
        return False


def _resource_snapshot() -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    if psutil is not None:
        try:
            process = psutil.Process(os.getpid())
            memory = process.memory_info()
            cpu = process.cpu_times()
            metrics.update(
                process_rss_mb=memory.rss / _MB,
                process_vms_mb=memory.vms / _MB,
                cpu_user_seconds_total=float(cpu.user),
                cpu_system_seconds_total=float(cpu.system),
                process_thread_count=process.num_threads(),
            )
        except Exception:
            pass
    if _cuda_available():
        try:
            device = torch.cuda.current_device()
            metrics.update(
                gpu_allocated_mb=torch.cuda.memory_allocated(device) / _MB,
                gpu_reserved_mb=torch.cuda.memory_reserved(device) / _MB,
                gpu_peak_allocated_mb=torch.cuda.max_memory_allocated(device) / _MB,
            )
        except Exception:
            pass
    return metrics


def _delta(after: dict[str, Any], before: dict[str, Any], key: str) -> float | None:
    if key not in before or key not in after:
        return None
    return float(after[key]) - float(before[key])


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


class PipelineProfiler:
    """Thread-safe, optional profiler shared by one pipeline execution."""

    def __init__(
        self,
        enabled: bool | None = None,
        config: ProfilingConfig | None = None,
    ) -> None:
        base = config or ProfilingConfig.from_env()
        if enabled is not None:
            base = ProfilingConfig(**{**asdict(base), "enabled": bool(enabled)})
        self.config = base
        self._records: list[dict[str, Any]] = []
        self._resource_samples: list[dict[str, Any]] = []
        self._run_metadata: dict[str, Any] = {}
        self._query_plans: list[dict[str, Any]] = []
        self._sql_trace: list[str] = []
        self._lock = threading.RLock()

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    @contextmanager
    def measure(
        self,
        name: str,
        metadata: dict[str, Any] | None = None,
        synchronize_cuda: bool = False,
    ) -> Iterator[None]:
        if not self.enabled:
            yield
            return

        should_sync = synchronize_cuda and self.config.cuda_sync and _cuda_available()
        if should_sync:
            torch.cuda.synchronize()

        before = _resource_snapshot()
        start_ns = time.perf_counter_ns()
        started_at = _utc_now()
        error: str | None = None
        try:
            yield
        except BaseException as exc:
            error = repr(exc)
            raise
        finally:
            if should_sync:
                torch.cuda.synchronize()
            end_ns = time.perf_counter_ns()
            ended_at = _utc_now()
            after = _resource_snapshot()
            elapsed = max(0.0, (end_ns - start_ns) / 1_000_000_000)
            cpu_user = _delta(after, before, "cpu_user_seconds_total")
            cpu_system = _delta(after, before, "cpu_system_seconds_total")
            cpu_total = (
                cpu_user + cpu_system
                if cpu_user is not None and cpu_system is not None
                else None
            )
            record = {
                "name": name,
                "start_timestamp": started_at,
                "end_timestamp": ended_at,
                "start_perf_counter_ns": start_ns,
                "end_perf_counter_ns": end_ns,
                "elapsed_seconds": elapsed,
                "elapsed_ms": elapsed * 1000.0,
                "thread_id": threading.get_ident(),
                "process_id": os.getpid(),
                "metadata": metadata or {},
                "error": error,
                "cpu_user_seconds": cpu_user,
                "cpu_system_seconds": cpu_system,
                "cpu_total_seconds": cpu_total,
                "average_cpu_cores": cpu_total / elapsed if cpu_total is not None and elapsed else 0.0,
                "rss_before_mb": before.get("process_rss_mb"),
                "rss_after_mb": after.get("process_rss_mb"),
                "rss_delta_mb": _delta(after, before, "process_rss_mb"),
                "cuda_allocated_before_mb": before.get("gpu_allocated_mb"),
                "cuda_allocated_after_mb": after.get("gpu_allocated_mb"),
                "cuda_allocated_delta_mb": _delta(after, before, "gpu_allocated_mb"),
                "cuda_reserved_before_mb": before.get("gpu_reserved_mb"),
                "cuda_reserved_after_mb": after.get("gpu_reserved_mb"),
                "cuda_reserved_delta_mb": _delta(after, before, "gpu_reserved_mb"),
                "cuda_peak_allocated_mb": after.get("gpu_peak_allocated_mb"),
            }
            with self._lock:
                self._records.append(record)
            if self.config.verbose:
                print(f"[PROFILE] {name}: {elapsed:.6f} seconds")

    def add_record(
        self,
        name: str,
        elapsed_seconds: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return
        now = _utc_now()
        with self._lock:
            self._records.append(
                {
                    "name": name,
                    "start_timestamp": now,
                    "end_timestamp": now,
                    "elapsed_seconds": float(elapsed_seconds),
                    "elapsed_ms": float(elapsed_seconds) * 1000.0,
                    "thread_id": threading.get_ident(),
                    "process_id": os.getpid(),
                    "metadata": metadata or {},
                    "error": None,
                }
            )

    def update_run_metadata(self, **values: Any) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._run_metadata.update(values)

    def add_query_plan(self, name: str, rows: list[Any]) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._query_plans.append({"name": name, "rows": rows})

    def add_sql_statement(self, statement: str) -> None:
        if not self.enabled or not self.config.sql:
            return
        compact = " ".join(statement.split())
        redacted = re.sub(r"(?:x)?'[^']*'", "?", compact, flags=re.IGNORECASE)
        with self._lock:
            self._sql_trace.append(redacted[:2000])

    def attach_resource_samples(self, samples: list[dict[str, Any]]) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._resource_samples = list(samples)

    def records(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._records)

    def summary(self) -> dict[str, Any]:
        with self._lock:
            records = list(self._records)
            run_metadata = dict(self._run_metadata)
            resource_samples = list(self._resource_samples)
            query_plans = list(self._query_plans)
            sql_trace = list(self._sql_trace)
        grouped: dict[str, list[float]] = defaultdict(list)
        for record in records:
            grouped[record["name"]].append(float(record["elapsed_seconds"]))

        operation_summary: dict[str, Any] = {}
        for name, values in grouped.items():
            operation_summary[name] = {
                "calls": len(values),
                "total_seconds": sum(values),
                "average_seconds": statistics.fmean(values),
                "minimum_seconds": min(values),
                "maximum_seconds": max(values),
                "median_seconds": statistics.median(values),
                "p50_seconds": _percentile(values, 0.50),
                "p90_seconds": _percentile(values, 0.90),
                "p95_seconds": _percentile(values, 0.95),
                "p99_seconds": _percentile(values, 0.99),
                "standard_deviation_seconds": statistics.pstdev(values),
            }
        operation_summary = dict(
            sorted(operation_summary.items(), key=lambda item: item[1]["total_seconds"], reverse=True)
        )
        pipeline_total = operation_summary.get("person_creation_pipeline", {}).get("total_seconds", 0.0)
        top_level_nodes = {
            name: {
                **stats,
                "pipeline_percent": (stats["total_seconds"] / pipeline_total * 100.0)
                if pipeline_total
                else None,
            }
            for name, stats in operation_summary.items()
            if name.startswith("node.")
        }
        nested_operations = {
            name: stats
            for name, stats in operation_summary.items()
            if not name.startswith("node.") and name != "person_creation_pipeline"
        }
        database_breakdown = {
            name: stats for name, stats in operation_summary.items() if name.startswith("db.")
        }
        video_seconds = operation_summary.get("video.complete", {}).get("total_seconds", 0.0)
        video_duration = float(run_metadata.get("video_duration_seconds") or 0.0)
        actual_selected = int(run_metadata.get("actual_selected_frames") or 0)
        source_frames = int(run_metadata.get("total_source_frames") or 0)
        performance_metrics = {
            "selected_frames_per_second": actual_selected / video_seconds if video_seconds else None,
            "source_frames_per_second": source_frames / video_seconds if video_seconds else None,
            "video_real_time_factor": video_seconds / video_duration if video_duration else None,
            "pipeline_real_time_factor": pipeline_total / video_duration if video_duration else None,
        }
        return {
            "run_metadata": run_metadata,
            "records": records,
            "summary": operation_summary,
            "top_level_node_breakdown": top_level_nodes,
            "nested_operation_breakdown": nested_operations,
            "database_breakdown": database_breakdown,
            "performance_metrics": performance_metrics,
            "resource_samples": resource_samples,
            "query_plans": query_plans,
            "sql_trace_statement_count": len(sql_trace),
        }

    def save(self, path: str | Path) -> None:
        if self.enabled:
            _atomic_json(Path(path), self.summary())

    def save_bundle(self, output_dir: str | Path) -> dict[str, str]:
        if not self.enabled:
            return {}
        report_dir = Path(output_dir) / "profiling"
        report = self.summary()
        paths = {
            "pipeline_profile": report_dir / "pipeline_profile.json",
            "resource_samples": report_dir / "resource_samples.json",
            "database_profile": report_dir / "database_profile.json",
            "profiling_summary": report_dir / "profiling_summary.txt",
        }
        _atomic_json(paths["pipeline_profile"], report)
        _atomic_json(paths["resource_samples"], {"samples": report["resource_samples"]})
        db_records = [r for r in report["records"] if r["name"].startswith("db.")]
        _atomic_json(
            paths["database_profile"],
            {
                "records": db_records,
                "summary": report["database_breakdown"],
                "query_plans": report["query_plans"],
            },
        )
        _atomic_text(paths["profiling_summary"], self.text_summary(report))
        if self.config.sql:
            sql_trace_path = report_dir / "sql_trace.log"
            with self._lock:
                sql_trace = list(self._sql_trace)
            _atomic_text(sql_trace_path, "\n".join(sql_trace) + ("\n" if sql_trace else ""))
            paths["sql_trace"] = sql_trace_path
        return {name: str(path.resolve()) for name, path in paths.items()}

    @staticmethod
    def text_summary(report: dict[str, Any]) -> str:
        metadata = report.get("run_metadata", {})
        summary = report.get("summary", {})
        pipeline_total = summary.get("person_creation_pipeline", {}).get("total_seconds", 0.0)
        duration = float(metadata.get("video_duration_seconds") or 0.0)
        lines = ["PROFILING SUMMARY", "", f"Pipeline total: {pipeline_total:.3f} s"]
        if duration:
            lines.extend(
                [f"Video duration: {duration:.3f} s", f"Real-time factor: {pipeline_total / duration:.3f}"]
            )
        lines.extend(["", "TOP-LEVEL NODES"])
        for name, stats in report.get("top_level_node_breakdown", {}).items():
            pct = stats.get("pipeline_percent")
            lines.append(
                f"{name:<38} calls={stats['calls']} total={stats['total_seconds']:.3f} s"
                + (f" pct={pct:.1f}%" if pct is not None else "")
            )
        for title, prefix in (("VIDEO OPERATIONS", ("video.", "frame.")), ("DATABASE", ("db.",))):
            lines.extend(["", title])
            for name, stats in summary.items():
                if name.startswith(prefix):
                    lines.append(
                        f"{name:<38} calls={stats['calls']} total={stats['total_seconds']:.3f} s "
                        f"avg={stats['average_seconds']:.6f} p95={stats['p95_seconds']:.6f}"
                    )
        samples = report.get("resource_samples", [])
        lines.extend(["", "RESOURCES"])
        if samples:
            def maximum(key: str) -> float | None:
                values = [float(s[key]) for s in samples if s.get(key) is not None]
                return max(values) if values else None
            lines.append(f"Peak RSS MB: {maximum('process_rss_mb')}")
            lines.append(f"Peak CUDA allocated MB: {maximum('gpu_allocated_mb')}")
            lines.append(f"Peak CUDA reserved MB: {maximum('gpu_reserved_mb')}")
            lines.append(f"Maximum process CPU percent: {maximum('process_cpu_percent')}")
            lines.append(f"Maximum GPU utilization percent: {maximum('gpu_utilization_percent')}")
        return "\n".join(lines) + "\n"


class ResourceSampler:
    """Non-blocking daemon sampler for process and optional NVIDIA resources."""

    def __init__(self, interval_seconds: float = 1.0, enabled: bool = True) -> None:
        self.interval_seconds = max(0.1, float(interval_seconds))
        self.enabled = enabled
        self._samples: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_ns = 0
        self._process = None
        self._nvml_handle = None

    @property
    def samples(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._samples)

    def start(self) -> None:
        if not self.enabled or (self._thread and self._thread.is_alive()):
            return
        self._stop_event.clear()
        self._start_ns = time.perf_counter_ns()
        if psutil is not None:
            try:
                self._process = psutil.Process(os.getpid())
                self._process.cpu_percent(interval=None)
            except Exception:
                self._process = None
        if pynvml is not None:
            try:
                pynvml.nvmlInit()
                index = torch.cuda.current_device() if _cuda_available() else 0
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            except Exception:
                self._nvml_handle = None
        self._thread = threading.Thread(target=self._run, name="pipeline-resource-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self._thread:
            return
        self._stop_event.set()
        self._thread.join(timeout=max(2.0, self.interval_seconds * 2))
        self._thread = None
        if self._nvml_handle is not None and pynvml is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml_handle = None

    def _run(self) -> None:
        while not self._stop_event.is_set():
            sample = self._sample()
            with self._lock:
                self._samples.append(sample)
            self._stop_event.wait(self.interval_seconds)

    def _sample(self) -> dict[str, Any]:
        sample: dict[str, Any] = {
            "timestamp": _utc_now(),
            "elapsed_seconds": (time.perf_counter_ns() - self._start_ns) / 1_000_000_000,
        }
        if self._process is not None:
            try:
                memory = self._process.memory_info()
                sample.update(
                    process_rss_mb=memory.rss / _MB,
                    process_vms_mb=memory.vms / _MB,
                    process_cpu_percent=self._process.cpu_percent(interval=None),
                    system_cpu_percent=psutil.cpu_percent(interval=None),
                    process_thread_count=self._process.num_threads(),
                )
            except Exception:
                pass
        sample.update({k: v for k, v in _resource_snapshot().items() if k.startswith("gpu_")})
        if self._nvml_handle is not None:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                memory = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                sample.update(
                    gpu_utilization_percent=float(util.gpu),
                    gpu_memory_utilization_percent=float(util.memory),
                    gpu_total_vram_used_mb=memory.used / _MB,
                    gpu_temperature_c=pynvml.nvmlDeviceGetTemperature(
                        self._nvml_handle, pynvml.NVML_TEMPERATURE_GPU
                    ),
                )
                try:
                    sample["gpu_power_watts"] = pynvml.nvmlDeviceGetPowerUsage(self._nvml_handle) / 1000.0
                except Exception:
                    pass
            except Exception:
                pass
        return sample


class ProfilingRun:
    """Own one profiler and sampler for one Flask job or CLI invocation."""

    def __init__(self, output_dir: str | Path, metadata: dict[str, Any] | None = None) -> None:
        self.output_dir = Path(output_dir)
        self.profiler = PipelineProfiler()
        self.sampler = ResourceSampler(
            self.profiler.config.resource_interval_seconds, enabled=self.profiler.enabled
        )
        self.metadata = dict(metadata or {})
        self.paths: dict[str, str] = {}
        self.report_error: str | None = None
        self._token: Token | None = None
        self._before: dict[str, Any] = {}
        self._sequence = 0
        self._completed_count = 0

    def __enter__(self) -> ProfilingRun:
        if not self.profiler.enabled:
            return self
        self._sequence = _next_job_sequence()
        self._before = _resource_snapshot()
        if _cuda_available():
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
        self.profiler.update_run_metadata(
            **self.metadata,
            process_id=os.getpid(),
            process_job_sequence=self._sequence,
            start_time=_utc_now(),
            python_version=sys.version,
            operating_system=platform.platform(),
            profiling_configuration=asdict(self.profiler.config),
            **_runtime_metadata(),
        )
        self._token = _active_profiler.set(self.profiler)
        self.sampler.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if not self.profiler.enabled:
            return False
        self.sampler.stop()
        self.profiler.attach_resource_samples(self.sampler.samples)
        after = _resource_snapshot()
        self._completed_count = _mark_job_completed()
        self.profiler.update_run_metadata(
            end_time=_utc_now(),
            pipeline_success=exc_type is None,
            pipeline_error=repr(exc) if exc is not None else None,
            rss_before_job_mb=self._before.get("process_rss_mb"),
            rss_after_job_mb=after.get("process_rss_mb"),
            cuda_allocated_before_job_mb=self._before.get("gpu_allocated_mb"),
            cuda_allocated_after_job_mb=after.get("gpu_allocated_mb"),
            cuda_reserved_before_job_mb=self._before.get("gpu_reserved_mb"),
            cuda_reserved_after_job_mb=after.get("gpu_reserved_mb"),
            completed_jobs_in_process=self._completed_count,
        )
        if self._token is not None:
            _active_profiler.reset(self._token)
        try:
            self.paths = self.profiler.save_bundle(self.output_dir)
            self._append_process_history(after, exc)
        except Exception:
            self.report_error = traceback.format_exc()
            print(f"[profiling] failed to save report: {self.report_error}")
        return False

    def _append_process_history(self, after: dict[str, Any], exc: BaseException | None) -> None:
        path = self.output_dir / "profiling" / "process_history.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": _utc_now(),
            "process_id": os.getpid(),
            "job_sequence": self._sequence,
            "completed_jobs_in_process": self._completed_count,
            "success": exc is None,
            "rss_before_mb": self._before.get("process_rss_mb"),
            "rss_after_mb": after.get("process_rss_mb"),
            "cuda_allocated_before_mb": self._before.get("gpu_allocated_mb"),
            "cuda_allocated_after_mb": after.get("gpu_allocated_mb"),
            "cuda_reserved_before_mb": self._before.get("gpu_reserved_mb"),
            "cuda_reserved_after_mb": after.get("gpu_reserved_mb"),
        }
        with _history_lock, path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(entry, default=str) + "\n")
            file.flush()
            os.fsync(file.fileno())
        self.paths["process_history"] = str(path.resolve())


def get_active_profiler() -> PipelineProfiler | None:
    profiler = _active_profiler.get()
    return profiler if profiler is not None and profiler.enabled else None


@contextmanager
def use_profiler(profiler: PipelineProfiler) -> Iterator[PipelineProfiler]:
    token = _active_profiler.set(profiler)
    try:
        yield profiler
    finally:
        _active_profiler.reset(token)


@contextmanager
def profile_measure(
    name: str,
    metadata: dict[str, Any] | None = None,
    synchronize_cuda: bool = False,
) -> Iterator[None]:
    profiler = get_active_profiler()
    if profiler is None:
        yield
    else:
        with profiler.measure(name, metadata=metadata, synchronize_cuda=synchronize_cuda):
            yield


@contextmanager
def cuda_event_measure(name: str, metadata: dict[str, Any] | None = None) -> Iterator[None]:
    """Measure CUDA work once with events when accurate CUDA mode is enabled."""
    profiler = get_active_profiler()
    if profiler is None or not profiler.config.cuda_sync or not _cuda_available():
        yield
        return
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    try:
        yield
    finally:
        end_event.record()
        torch.cuda.synchronize()
        event_metadata = dict(metadata or {})
        event_metadata["timing_method"] = "cuda_event"
        profiler.add_record(name, start_event.elapsed_time(end_event) / 1000.0, event_metadata)


def _runtime_metadata() -> dict[str, Any]:
    data: dict[str, Any] = {"git_commit": None}
    try:
        data["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=2, check=False
        ).stdout.strip() or None
    except Exception:
        pass
    if torch is not None:
        data.update(torch_version=getattr(torch, "__version__", None), cuda_runtime_version=torch.version.cuda)
    if _cuda_available():
        try:
            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)
            data.update(
                gpu_name=torch.cuda.get_device_name(device),
                gpu_total_memory_mb=props.total_memory / _MB,
                device_actually_used=f"cuda:{device}",
            )
        except Exception:
            pass
    return data


def _atomic_json(path: Path, data: Any) -> None:
    _atomic_text(path, json.dumps(data, indent=2, default=str))


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


# Backward-compatible module object; new pipeline runs use ProfilingRun instances.
profiler = PipelineProfiler()
