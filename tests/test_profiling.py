import json
import time

import numpy as np
import pytest

from forensics.person_creation.utils import profiling
from forensics.person_creation.utils.profiling import (
    PipelineProfiler,
    ProfilingConfig,
    ProfilingRun,
    ResourceSampler,
    use_profiler,
)


def _config(**overrides):
    values = {
        "enabled": True,
        "cuda_sync": False,
        "resource_interval_seconds": 0.1,
        "verbose": False,
        "frame_level": False,
        "sql": False,
    }
    values.update(overrides)
    return ProfilingConfig(**values)


def test_measure_success_records_timing():
    profiler = PipelineProfiler(config=_config())
    with profiler.measure("operation"):
        pass

    record = profiler.records()[0]
    assert record["name"] == "operation"
    assert record["elapsed_seconds"] >= 0
    assert record["elapsed_ms"] >= 0
    assert record["error"] is None
    assert record["thread_id"]


def test_measure_exception_records_and_reraises():
    profiler = PipelineProfiler(config=_config())
    with pytest.raises(RuntimeError, match="controlled"):
        with profiler.measure("failing"):
            raise RuntimeError("controlled")

    assert "controlled" in profiler.records()[0]["error"]


def test_cpu_delta_uses_process_cpu_times(monkeypatch):
    snapshots = iter([
        {"cpu_user_seconds_total": 1.0, "cpu_system_seconds_total": 2.0},
        {"cpu_user_seconds_total": 1.3, "cpu_system_seconds_total": 2.2},
    ])
    monkeypatch.setattr(profiling, "_resource_snapshot", lambda: next(snapshots))
    profiler = PipelineProfiler(config=_config())
    with profiler.measure("cpu"):
        pass

    record = profiler.records()[0]
    assert record["cpu_user_seconds"] == pytest.approx(0.3)
    assert record["cpu_system_seconds"] == pytest.approx(0.2)
    assert record["cpu_total_seconds"] == pytest.approx(0.5)
    assert record["average_cpu_cores"] >= 0


def test_percentile_and_statistical_summary():
    profiler = PipelineProfiler(config=_config())
    for value in (1.0, 2.0, 3.0, 4.0):
        profiler.add_record("repeated", value)

    stats = profiler.summary()["summary"]["repeated"]
    assert stats["median_seconds"] == pytest.approx(2.5)
    assert stats["p50_seconds"] == pytest.approx(2.5)
    assert stats["p95_seconds"] == pytest.approx(3.85)
    assert stats["standard_deviation_seconds"] > 0


def test_resource_sampler_starts_and_stops():
    sampler = ResourceSampler(interval_seconds=0.1)
    sampler.start()
    time.sleep(0.15)
    sampler.stop()
    assert sampler.samples
    assert "timestamp" in sampler.samples[0]


def test_disabled_profiler_is_noop(tmp_path):
    profiler = PipelineProfiler(config=_config(enabled=False))
    with profiler.measure("disabled"):
        pass
    path = tmp_path / "disabled.json"
    profiler.save(path)
    assert profiler.records() == []
    assert not path.exists()


def test_no_cuda_behavior(monkeypatch):
    monkeypatch.setattr(profiling, "torch", None)
    profiler = PipelineProfiler(config=_config(cuda_sync=True))
    with profiler.measure("cpu_only", synchronize_cuda=True):
        pass
    record = profiler.records()[0]
    assert record["cuda_allocated_before_mb"] is None
    assert record["cuda_reserved_after_mb"] is None


def test_database_profiling_wrapper(tmp_path):
    from forensics.global_memory.store import GlobalMemory

    profiler = PipelineProfiler(config=_config(sql=True))
    profile = {
        "face_embedding": np.asarray([1.0, 0.0], dtype=np.float32),
        "face_crops": [],
        "appearance": {},
    }
    with use_profiler(profiler):
        memory = GlobalMemory(str(tmp_path / "memory.db"))
        try:
            person_id = memory.register(profile)
            matches = memory.query_by_face([1.0, 0.0], threshold=0.0)
        finally:
            memory.close()

    names = {record["name"] for record in profiler.records()}
    assert person_id == "person_001"
    assert matches[0]["person_id"] == person_id
    assert "db.connect" in names
    assert "db.person.insert" in names
    assert "db.commit" in names
    assert "db.search_by_face.fetch" in names
    assert "db.close" in names
    assert profiler.summary()["query_plans"]


def test_report_serialization_after_controlled_error(tmp_path):
    profiler = PipelineProfiler(config=_config())
    with pytest.raises(ValueError):
        with profiler.measure("controlled_failure"):
            raise ValueError("expected")
    path = tmp_path / "report.json"
    profiler.save(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["records"][0]["error"] == "ValueError('expected')"


def test_profiling_run_saves_bundle_when_pipeline_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("PERSON_CREATION_PROFILE", "1")
    run = ProfilingRun(tmp_path, metadata={"job_id": "controlled"})
    with pytest.raises(RuntimeError):
        with run:
            with run.profiler.measure("person_creation_pipeline"):
                raise RuntimeError("controlled failure")

    assert run.report_error is None
    report_path = tmp_path / "profiling" / "pipeline_profile.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["run_metadata"]["pipeline_success"] is False
