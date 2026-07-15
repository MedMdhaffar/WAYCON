from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

from forensics.person_creation.models import clothing_describer
from forensics.person_creation.nodes import describe_clothing as clothing
from forensics.person_creation.nodes import finalize as finalize_node


def _state(path: Path) -> dict:
    return {
        "per_cluster_best_body_crops": {0: [str(path)]},
        "best_body_crops": [str(path)],
    }


def _wait_for_workers_to_close(timeout: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not any(
            thread.name.startswith("person-creation-vlm")
            for thread in threading.enumerate()
        ):
            return True
        time.sleep(0.01)
    return False


def _install(
    monkeypatch,
    describer,
    *,
    timeout: float = 0.05,
    read_crops=None,
) -> None:
    monkeypatch.setattr(clothing_describer, "get_clothing_describer", lambda: describer)
    monkeypatch.setattr(clothing, "async_vlm_config", lambda: (True, timeout))
    if read_crops is None:
        read_crops = lambda _paths: [np.zeros((8, 8, 3), dtype=np.uint8)]
    monkeypatch.setattr(clothing, "_read_crops", read_crops)


def test_successful_async_completion_preserves_output(monkeypatch, tmp_path):
    structured = {
        "top": "blue shirt",
        "bottom": "black trousers",
        "shoes": "white shoes",
        "full": "blue shirt, black trousers, white shoes",
    }

    class Describer:
        def describe(self, crops):
            assert len(crops) == 1
            return "raw-vlm-output", structured

    _install(monkeypatch, Describer())

    result = clothing.describe_clothing(_state(tmp_path / "body.jpg"))

    assert result == {
        "per_cluster_clothing": {
            0: {"raw": "raw-vlm-output", "structured": structured}
        },
        "clothing_raw": "raw-vlm-output",
        "clothing_structured": structured,
    }
    assert _wait_for_workers_to_close()


def test_timeout_returns_fallback_without_post_return_path_reads(
    monkeypatch, tmp_path, capsys
):
    crop_path = tmp_path / "body.jpg"
    crop_path.write_bytes(b"crop")
    controller_thread = threading.get_ident()
    read_threads = []
    worker_entered = threading.Event()
    release_worker = threading.Event()
    worker_exited = threading.Event()

    def read_crops(paths):
        read_threads.append(threading.get_ident())
        assert paths == [str(crop_path)]
        assert crop_path.read_bytes() == b"crop"
        return [np.zeros((8, 8, 3), dtype=np.uint8)]

    class BlockingDescriber:
        def describe(self, _crops):
            worker_entered.set()
            try:
                release_worker.wait(timeout=1.0)
                return "late", {"top": "late"}
            finally:
                worker_exited.set()

    _install(
        monkeypatch,
        BlockingDescriber(),
        timeout=0.02,
        read_crops=read_crops,
    )

    started = time.monotonic()
    try:
        result = clothing.describe_clothing(_state(crop_path))
        elapsed = time.monotonic() - started

        assert worker_entered.is_set()
        assert elapsed < 0.5
        assert result == clothing._fallback_result({0: [str(crop_path)]})
        assert read_threads == [controller_thread]

        # Finalization can remove the source while inference still owns arrays.
        finalize_node._cleanup_staging({}, tmp_path)
        time.sleep(0.03)
        assert read_threads == [controller_thread]
    finally:
        release_worker.set()

    assert worker_exited.wait(timeout=1.0)
    assert _wait_for_workers_to_close()
    output = capsys.readouterr().out
    assert "VLM timeout reached" in output
    assert "future cancel requested" in output
    assert "future cancel result=false" in output
    assert "[finalize] cleanup completed" in output


def test_worker_exception_uses_existing_fallback(monkeypatch, tmp_path):
    class FailingDescriber:
        def describe(self, _crops):
            raise RuntimeError("mock VLM failure")

    _install(monkeypatch, FailingDescriber())

    result = clothing.describe_clothing(_state(tmp_path / "body.jpg"))

    assert result == clothing._fallback_result({0: [str(tmp_path / "body.jpg")]})
    assert _wait_for_workers_to_close()


def test_path_reading_finishes_before_worker_submission(monkeypatch, tmp_path, capsys):
    crop_path = tmp_path / "body.jpg"
    crop_path.write_bytes(b"crop")
    events = []

    def read_crops(_paths):
        events.append("reader_finished")
        return [np.zeros((8, 8, 3), dtype=np.uint8)]

    class Describer:
        def describe(self, _crops):
            events.append("worker_entered")
            return "raw", {
                "top": "unknown",
                "bottom": "unknown",
                "shoes": "unknown",
                "full": "Clothing description unavailable.",
            }

    _install(monkeypatch, Describer(), read_crops=read_crops)

    clothing.describe_clothing(_state(crop_path))
    output = capsys.readouterr().out
    crop_path.unlink()

    assert events == ["reader_finished", "worker_entered"]
    assert output.index("worker entered") < output.index("worker exited")
    assert output.index("executor shutdown entered") < output.index(
        "executor shutdown completed"
    )
    assert not crop_path.exists()
    assert _wait_for_workers_to_close()


def test_sync_mode_output_is_unchanged(monkeypatch, tmp_path):
    structured = {
        "top": "green coat",
        "bottom": "unknown",
        "shoes": "unknown",
        "full": "green coat",
    }

    class Describer:
        def describe(self, _crops):
            return "sync-raw", structured

    monkeypatch.setattr(clothing_describer, "get_clothing_describer", lambda: Describer())
    monkeypatch.setattr(clothing, "async_vlm_config", lambda: (False, 60.0))
    monkeypatch.setattr(
        clothing,
        "_read_crops",
        lambda _paths: [np.zeros((8, 8, 3), dtype=np.uint8)],
    )

    result = clothing.describe_clothing(_state(tmp_path / "body.jpg"))

    assert result["clothing_raw"] == "sync-raw"
    assert result["clothing_structured"] == structured


def test_default_timeout_remains_sixty_seconds(monkeypatch):
    monkeypatch.delenv("PERSON_CREATION_ASYNC_VLM", raising=False)
    monkeypatch.delenv("PERSON_CREATION_VLM_TIMEOUT_SECONDS", raising=False)

    assert clothing.async_vlm_config() == (True, 60.0)
