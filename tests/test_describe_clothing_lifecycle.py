from __future__ import annotations

from concurrent.futures import Future, TimeoutError as FutureTimeoutError
import time

import numpy as np
import pytest

from forensics.person_creation.models import clothing_describer
from forensics.person_creation.nodes import describe_clothing as clothing


def _state(*cluster_ids: int) -> dict:
    ids = cluster_ids or (0,)
    return {
        "per_cluster_best_body_crops": {
            cluster_id: [f"body-{cluster_id}-a.jpg", f"body-{cluster_id}-b.jpg"]
            for cluster_id in ids
        },
        "best_body_crops": [],
    }


def _install(monkeypatch, describer, *, retries: int = 1, async_enabled: bool = False):
    monkeypatch.setattr(clothing_describer, "get_clothing_describer", lambda: describer)
    monkeypatch.setattr(clothing, "async_vlm_config", lambda: (async_enabled, 0.01))
    monkeypatch.setattr(clothing, "vlm_max_retries", lambda: retries)
    monkeypatch.setattr(
        clothing,
        "_read_crops",
        lambda paths: (
            [np.zeros((8, 6, 3), dtype=np.uint8) for _ in paths],
            [
                {"basename": path, "width": 6, "height": 8}
                for path in paths
            ],
        ),
    )


def _valid(top: str) -> tuple[str, dict]:
    return "raw", {
        "top": top,
        "bottom": "blue jeans",
        "shoes": "white sneakers",
        "full": f"{top}, blue jeans, white sneakers",
    }


def test_one_cluster_success_returns_structured_status(monkeypatch):
    class Describer:
        def describe(self, _crops):
            return _valid("black jacket")

    _install(monkeypatch, Describer())
    result = clothing.describe_clothing(_state(4))

    record = result["per_cluster_clothing"][4]
    assert record == {
        "status": "ok",
        "attempts": 1,
        "top": "black jacket",
        "bottom": "blue jeans",
        "shoes": "white sneakers",
        "full": "black jacket, blue jeans, white sneakers",
        "failure_reason": None,
    }
    assert result["clothing_raw"] == ""
    assert result["clothing_diagnostics"][0]["selected_body_crop"] == "body-4-a.jpg"


def test_multi_cluster_success_and_assignment_are_independent(monkeypatch):
    calls = 0

    class Describer:
        def describe(self, _crops):
            nonlocal calls
            calls += 1
            return _valid(f"top-{calls}")

    _install(monkeypatch, Describer())
    result = clothing.describe_clothing(_state(7, 2))

    assert result["per_cluster_clothing"][2]["top"] == "top-1"
    assert result["per_cluster_clothing"][7]["top"] == "top-2"


@pytest.mark.parametrize(
    ("response", "reason"),
    [
        (("", {}), "empty_output"),
        (("malformed", {"top": "unknown", "bottom": "unknown"}), "invalid_output"),
    ],
)
def test_empty_and_malformed_output_retry_then_fail(monkeypatch, response, reason):
    class Describer:
        def describe(self, _crops):
            return response

    _install(monkeypatch, Describer(), retries=1)
    record = clothing.describe_clothing(_state())["per_cluster_clothing"][0]

    assert record["status"] == "failed"
    assert record["attempts"] == 2
    assert record["failure_reason"] == reason
    assert record["top"] is None


def test_exception_isolated_and_other_cluster_succeeds(monkeypatch):
    calls = 0

    class Describer:
        def describe(self, _crops):
            nonlocal calls
            calls += 1
            if calls <= 2:
                raise RuntimeError("private model error")
            return _valid("green coat")

    _install(monkeypatch, Describer(), retries=1)
    result = clothing.describe_clothing(_state(0, 1))

    assert result["per_cluster_clothing"][0]["failure_reason"] == "inference_error"
    assert result["per_cluster_clothing"][1]["status"] == "ok"


def test_first_attempt_fails_and_next_best_crop_succeeds(monkeypatch):
    calls = []

    class Describer:
        def describe(self, crops):
            calls.append(len(crops))
            return ("", {}) if len(calls) == 1 else _valid("red sweater")

    _install(monkeypatch, Describer(), retries=1)
    record = clothing.describe_clothing(_state())["per_cluster_clothing"][0]

    assert calls == [2, 1]
    assert record["status"] == "ok"
    assert record["attempts"] == 2


def test_no_valid_crop_has_explicit_failure(monkeypatch):
    class Describer:
        def describe(self, _crops):
            raise AssertionError("must not run")

    _install(monkeypatch, Describer())
    result = clothing.describe_clothing({"per_cluster_best_body_crops": {3: []}})

    assert result["per_cluster_clothing"][3]["failure_reason"] == "no_valid_body_crop"
    assert result["per_cluster_clothing"][3]["attempts"] == 0


def test_image_decode_failure_is_explicit(monkeypatch):
    class Describer:
        def describe(self, _crops):
            raise AssertionError("must not run")

    _install(monkeypatch, Describer())
    monkeypatch.setattr(clothing, "_read_crops", lambda _paths: ([], []))
    record = clothing.describe_clothing(_state())["per_cluster_clothing"][0]

    assert record["failure_reason"] == "image_decode_failed"


def test_running_timeout_is_not_claimed_cancelled_or_retried(monkeypatch):
    class RunningFuture(Future):
        def __init__(self):
            super().__init__()
            self.set_running_or_notify_cancel()

        def result(self, timeout=None):
            raise FutureTimeoutError()

    class Worker:
        def submit(self, *_args):
            return RunningFuture()

    class Describer:
        def describe(self, _crops):
            return _valid("late")

    _install(monkeypatch, Describer(), retries=1, async_enabled=True)
    monkeypatch.setattr(clothing, "_VLM_WORKER", Worker())
    record = clothing.describe_clothing(_state())["per_cluster_clothing"][0]

    assert record["failure_reason"] == "timeout"
    assert record["attempts"] == 1


def test_one_slow_timed_out_cluster_does_not_erase_later_success(monkeypatch):
    calls = 0

    class Describer:
        def describe(self, _crops):
            nonlocal calls
            calls += 1
            if calls == 1:
                time.sleep(0.06)
                return _valid("late top")
            return _valid("prompt top")

    _install(monkeypatch, Describer(), retries=0, async_enabled=True)
    monkeypatch.setattr(clothing, "async_vlm_config", lambda: (True, 0.05))
    monkeypatch.setattr(clothing, "_VLM_WORKER", clothing._ControlledVLMWorker())

    result = clothing.describe_clothing(_state(0, 1))

    assert result["per_cluster_clothing"][0]["failure_reason"] == "timeout"
    assert result["per_cluster_clothing"][1]["status"] == "ok"
    assert result["per_cluster_clothing"][1]["top"] == "prompt top"


def test_controlled_worker_bounds_pending_tasks(monkeypatch):
    worker = clothing._ControlledVLMWorker()
    release = __import__("threading").Event()

    def block():
        release.wait(timeout=1)

    try:
        assert worker.submit(block) is not None
        assert worker.submit(block) is not None
        assert worker.submit(block) is None
        assert worker.pending == 2
    finally:
        release.set()


def test_default_timeout_and_retry_configuration(monkeypatch):
    monkeypatch.delenv("PERSON_CREATION_ASYNC_VLM", raising=False)
    monkeypatch.delenv("PERSON_CREATION_VLM_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("PERSON_CREATION_VLM_MAX_RETRIES", raising=False)

    assert clothing.async_vlm_config() == (True, 60.0)
    assert clothing.vlm_max_retries() == 1
