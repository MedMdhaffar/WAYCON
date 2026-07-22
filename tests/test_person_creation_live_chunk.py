from __future__ import annotations

import json
import threading
from copy import deepcopy
from types import SimpleNamespace

import pytest

from forensics.person_creation.live_stream import (
    BufferedFrame,
    LiveFrameBufferLifecycleError,
    mask_camera_uri,
)
from forensics.person_creation.nodes import process_live_stream as live_node


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class FakeBuffer:
    def __init__(self, clock: FakeClock, items=(), *, on_get=None) -> None:
        self.clock = clock
        self.items = list(items)
        self.on_get = on_get
        self.start_calls = 0
        self.stop_calls = 0
        self.get_calls = 0
        self.get_timeouts: list[float] = []
        self.frames_read = 0
        self.frames_dropped = 0
        self.error = None
        self.ended = False
        self.stream_opened = False
        self.stream_state = "connected"
        self.stream_reconnect_count = 0
        self.stream_warning = None

    @property
    def empty(self) -> bool:
        return not self.items

    def start(self) -> None:
        self.start_calls += 1
        self.stream_opened = True

    def stop(self) -> None:
        self.stop_calls += 1

    def get(self, timeout: float):
        self.get_calls += 1
        self.get_timeouts.append(timeout)
        if self.on_get is not None:
            self.on_get(self)
        if self.items:
            self.clock.advance(0.1)
            self.frames_read += 1
            return self.items.pop(0)
        self.clock.advance(timeout)
        return None

    def stats(self, frames_processed: int, warnings=None) -> dict:
        return {
            "stream_opened": self.stream_opened,
            "frames_read": self.frames_read,
            "frames_processed": frames_processed,
            "frames_dropped": self.frames_dropped,
            "buffer_max_size": 30,
            "first_frame_time": None,
            "last_frame_time": None,
            "stream_state": self.stream_state,
            "stream_reconnect_count": self.stream_reconnect_count,
            "stream_warning": self.stream_warning,
            "last_frame_age_seconds": None,
            "warnings": list(warnings or []),
        }


def _frame(index: int) -> BufferedFrame:
    return BufferedFrame(frame_idx=index, timestamp=f"t{index}", frame=object())


def _capture(buffer, clock, **overrides):
    options = {
        "buffer": buffer,
        "duration_seconds": 2.0,
        "every_n": 1,
        "source_stem": "live",
        "source_metadata": {"source_type": "live_camera"},
        "body_dir": "body",
        "face_dir": "face",
        "person_detector": object(),
        "face_detector": object(),
        "frame_timeout_seconds": 10.0,
        "monotonic": clock,
    }
    options.update(overrides)
    return live_node.capture_live_chunk(**options)


def test_chunk_does_not_start_or_stop_buffer():
    clock = FakeClock()
    buffer = FakeBuffer(clock)

    _capture(buffer, clock)

    assert buffer.start_calls == 0
    assert buffer.stop_calls == 0


def test_normal_chunk_exits_after_duration_without_busy_spin():
    clock = FakeClock()
    buffer = FakeBuffer(clock)

    result = _capture(buffer, clock, duration_seconds=2.0)

    assert result.elapsed_seconds == pytest.approx(2.0)
    assert buffer.get_calls == 2
    assert all(0 < timeout <= 1.0 for timeout in buffer.get_timeouts)


def test_reconnecting_empty_window_is_nonterminal_and_publishes_status():
    clock = FakeClock()
    notifications = []
    buffer = FakeBuffer(clock)
    buffer.stream_state = "reconnecting"
    buffer.stream_reconnect_count = 2
    buffer.stream_warning = "Temporary camera interruption; reconnecting."

    result = _capture(
        buffer,
        clock,
        duration_seconds=2.0,
        frame_timeout_seconds=0.5,
        notify=lambda status, update=None: notifications.append((status, update)),
    )

    assert result.stop_requested is False
    assert result.frames_read == 0
    assert buffer.ended is False
    assert "reader remains active" in " ".join(result.warnings)
    assert notifications[-1][1]["stream_stats"]["stream_state"] == "reconnecting"
    assert notifications[-1][1]["stream_stats"]["stream_reconnect_count"] == 2


def test_pre_set_stop_returns_immediate_empty_result():
    clock = FakeClock()
    buffer = FakeBuffer(clock)
    stop_event = threading.Event()
    stop_event.set()

    result = _capture(buffer, clock, stop_event=stop_event)

    assert result.stop_requested is True
    assert result.elapsed_seconds == pytest.approx(0.0)
    assert result.frames_processed == 0
    assert result.body_crops == []
    assert result.face_crops == []
    assert buffer.get_calls == 0


def test_stop_during_wait_exits_within_bounded_get_timeout():
    clock = FakeClock()
    stop_event = threading.Event()
    buffer = FakeBuffer(clock, on_get=lambda _buffer: stop_event.set())

    result = _capture(buffer, clock, stop_event=stop_event)

    assert result.stop_requested is True
    assert result.elapsed_seconds <= 1.0
    assert buffer.get_calls == 1
    assert buffer.get_timeouts == [1.0]


def test_crops_and_counters_are_returned(monkeypatch):
    clock = FakeClock()
    buffer = FakeBuffer(clock, [_frame(0), _frame(1), _frame(2)])

    def fake_detect(_frame_value, **kwargs):
        index = kwargs["frame_idx"]
        return ([{"path": f"body-{index}"}], [{"path": f"face-{index}"}])

    monkeypatch.setattr(live_node, "detect_and_save_frame", fake_detect)
    result = _capture(buffer, clock, every_n=2)

    assert result.frames_read == 3
    assert result.frames_processed == 2
    assert result.frames_skipped == 1
    assert result.body_detection_count == 2
    assert result.face_detection_count == 2
    assert [crop["path"] for crop in result.body_crops] == ["body-0", "body-2"]


def test_nonzero_chunk_uses_collision_safe_source_stem(monkeypatch):
    stems = []

    def fake_detect(_frame_value, **kwargs):
        stems.append(kwargs["source_stem"])
        return [], []

    monkeypatch.setattr(live_node, "detect_and_save_frame", fake_detect)
    for chunk_index in (0, 1):
        clock = FakeClock()
        buffer = FakeBuffer(clock, [_frame(0)])
        _capture(buffer, clock, chunk_index=chunk_index)

    assert stems == ["live", "live_chunk_0001"]


def test_process_live_stream_owns_buffer_and_preserves_output(
    monkeypatch, tmp_path, capsys
):
    clock = FakeClock()
    stop_event = threading.Event()
    buffer = FakeBuffer(
        clock,
        [_frame(0)],
        on_get=lambda current: stop_event.set() if not current.items else None,
    )
    monkeypatch.setattr(live_node, "LiveFrameBuffer", lambda *_args, **_kwargs: buffer)
    monkeypatch.setattr(
        live_node,
        "prepare_staging_dirs",
        lambda _output: (tmp_path / "body", tmp_path / "face"),
    )
    monkeypatch.setattr(
        live_node,
        "detect_and_save_frame",
        lambda *_args, **_kwargs: ([{"path": "body.jpg"}], [{"path": "face.jpg"}]),
    )

    import forensics.face_engine.client as face_client
    import forensics.person_creation.models.person_detector as person_detector

    monkeypatch.setattr(face_client, "FaceEngineClient", lambda: object())
    monkeypatch.setattr(person_detector, "get_person_detector", lambda: object())

    raw_uri = "rtsp://private-user:private-password@camera.local/live"
    result = live_node.process_live_stream({
        "camera_uri": raw_uri,
        "camera_id": "lobby",
        "duration_seconds": 10,
        "process_every_n": 1,
        "live_stream_config": {},
        "output_dir": str(tmp_path),
        "_stop_event": stop_event,
    })
    report = json.loads((tmp_path / "stream_report.json").read_text(encoding="utf-8"))
    output = capsys.readouterr().out

    assert buffer.start_calls == 1
    assert buffer.stop_calls == 1
    assert result["body_crops"] == [{"path": "body.jpg"}]
    assert result["face_crops"] == [{"path": "face.jpg"}]
    assert result["camera_uri"] == ""
    assert result["source_type"] == "live_camera"
    assert result["stream_stats"]["frames_read"] == 1
    assert report["source_type"] == "live_camera"
    assert report["duration_seconds"] == 10
    assert report["frames_read"] == 1
    assert report["chunks"][0]["chunk_index"] == 0
    assert report["chunks"][0]["body_detections"] == 1
    assert "private-user" not in json.dumps(report)
    assert "private-password" not in json.dumps(report)
    assert "private-user" not in output
    assert "private-password" not in output


def test_detector_exception_still_stops_owned_buffer(monkeypatch, tmp_path):
    clock = FakeClock()
    buffer = FakeBuffer(clock, [_frame(0)])
    monkeypatch.setattr(live_node, "LiveFrameBuffer", lambda *_args, **_kwargs: buffer)
    monkeypatch.setattr(
        live_node,
        "prepare_staging_dirs",
        lambda _output: (tmp_path / "body", tmp_path / "face"),
    )
    monkeypatch.setattr(
        live_node,
        "detect_and_save_frame",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("detector failed")),
    )

    import forensics.face_engine.client as face_client
    import forensics.person_creation.models.person_detector as person_detector

    monkeypatch.setattr(face_client, "FaceEngineClient", lambda: object())
    monkeypatch.setattr(person_detector, "get_person_detector", lambda: object())

    with pytest.raises(RuntimeError, match="detector failed"):
        live_node.process_live_stream({
            "camera_uri": "rtsp://camera.local/live",
            "duration_seconds": 10,
            "process_every_n": 1,
            "output_dir": str(tmp_path),
        })

    assert buffer.start_calls == 1
    assert buffer.stop_calls == 1


def test_mask_camera_uri_removes_all_user_info():
    masked = mask_camera_uri("rtsp://private-user:private-password@camera.local/live")

    assert masked == "rtsp://****@camera.local/live"
    assert "private-user" not in masked
    assert "private-password" not in masked


def _install_continuous_dependencies(monkeypatch, tmp_path, buffer):
    monkeypatch.setattr(live_node, "LiveFrameBuffer", lambda *_args, **_kwargs: buffer)
    monkeypatch.setattr(
        live_node,
        "prepare_staging_dirs",
        lambda _output: (tmp_path / "body", tmp_path / "face"),
    )

    import forensics.face_engine.client as face_client
    import forensics.person_creation.models.person_detector as person_detector

    monkeypatch.setattr(face_client, "FaceEngineClient", lambda: object())
    monkeypatch.setattr(person_detector, "get_person_detector", lambda: object())


def _chunk_result(index, *, stop=False, empty=False, frame_gap_active=False):
    warnings = (
        ["No camera frames arrived for 5 seconds; the live reader remains active."]
        if frame_gap_active
        else []
    )
    return live_node.LiveChunkResult(
        chunk_index=index,
        started_at=float(index),
        elapsed_seconds=10.0,
        stop_requested=stop,
        frames_read=0 if empty else index + 1,
        frames_processed=0 if empty else 2,
        frames_skipped=0 if empty else 3,
        frames_dropped=0 if empty else 1,
        body_crops=[] if empty else [{"path": f"body-{index}.jpg"}],
        face_crops=[] if empty else [{"path": f"face-{index}.jpg"}],
        body_detection_count=0 if empty else 1,
        face_detection_count=0 if empty else 1,
        warnings=warnings,
        frame_gap_active=frame_gap_active,
    )


def test_continuous_capture_accumulates_until_stop_without_fixed_limit(
    monkeypatch, tmp_path
):
    clock = FakeClock()
    buffer = FakeBuffer(clock)
    stop_event = threading.Event()
    chunk_indices = []
    notifications = []
    _install_continuous_dependencies(monkeypatch, tmp_path, buffer)

    def fake_capture(**kwargs):
        index = kwargs["chunk_index"]
        chunk_indices.append(index)
        assert kwargs["buffer"] is buffer
        assert kwargs["stop_event"] is stop_event
        if index == 7:
            stop_event.set()
        return _chunk_result(index, stop=stop_event.is_set())

    monkeypatch.setattr(live_node, "capture_live_chunk", fake_capture)
    raw_uri = "rtsp://private-user:private-password@camera.local/live"
    result = live_node.process_live_stream({
        "camera_uri": raw_uri,
        "camera_id": "lobby",
        "duration_seconds": 10,
        "process_every_n": 5,
        "live_stream_config": {},
        "output_dir": str(tmp_path),
        "_stop_event": stop_event,
        "_status_callback": lambda status, update=None: notifications.append(
            (status, update)
        ),
    })

    completed = [
        update
        for status, update in notifications
        if status == "processing_live_frames"
        and update
        and "completed_chunks" in update
    ]
    report = json.loads((tmp_path / "stream_report.json").read_text(encoding="utf-8"))
    public_text = json.dumps({"notifications": notifications, "report": report})

    assert chunk_indices == list(range(8))
    assert buffer.start_calls == 1
    assert buffer.stop_calls == 1
    assert [crop["path"] for crop in result["body_crops"]] == [
        f"body-{index}.jpg" for index in range(8)
    ]
    assert [crop["path"] for crop in result["face_crops"]] == [
        f"face-{index}.jpg" for index in range(8)
    ]
    assert result["stream_stats"]["frames_read"] == sum(range(1, 9))
    assert result["stream_stats"]["frames_processed"] == 16
    assert result["stream_stats"]["frames_skipped"] == 24
    assert result["stream_stats"]["frames_dropped"] == 8
    assert result["stream_stats"]["body_detections"] == 8
    assert result["stream_stats"]["face_detections"] == 8
    assert len(completed) == 8
    assert completed[-1]["chunk_index"] == 7
    assert completed[-1]["completed_chunks"] == 8
    assert completed[-1]["session_totals"]["frames_processed"] == 16
    assert report["continuous"] is True
    assert report["duration_seconds"] == 10
    assert report["duration_seconds_per_chunk"] == 10
    assert report["completed_chunks"] == 8
    assert report["stop_requested"] is True
    assert len(report["chunks"]) == 8
    assert "private-user" not in public_text
    assert "private-password" not in public_text
    assert raw_uri not in public_text


def test_returned_accumulated_state_can_enter_next_graph_node(monkeypatch, tmp_path):
    clock = FakeClock()
    buffer = FakeBuffer(clock)
    stop_event = threading.Event()
    _install_continuous_dependencies(monkeypatch, tmp_path, buffer)

    def fake_capture(**kwargs):
        stop_event.set()
        return _chunk_result(kwargs["chunk_index"], stop=True)

    monkeypatch.setattr(live_node, "capture_live_chunk", fake_capture)
    result = live_node.process_live_stream({
        "camera_uri": "rtsp://camera.local/live",
        "duration_seconds": 10,
        "process_every_n": 1,
        "output_dir": str(tmp_path),
        "_stop_event": stop_event,
    })

    downstream_seen = {}

    def mocked_next_node(state):
        downstream_seen.update(state)
        return {"continued": True}

    assert mocked_next_node(result) == {"continued": True}
    assert downstream_seen["body_crops"] == [{"path": "body-0.jpg"}]
    assert downstream_seen["face_crops"] == [{"path": "face-0.jpg"}]
    assert downstream_seen["stream_stats"]["stop_requested"] is True


def test_empty_window_continues_to_next_chunk(monkeypatch, tmp_path):
    clock = FakeClock()
    buffer = FakeBuffer(clock)
    stop_event = threading.Event()
    indices = []
    _install_continuous_dependencies(monkeypatch, tmp_path, buffer)

    def fake_capture(**kwargs):
        index = kwargs["chunk_index"]
        indices.append(index)
        if index == 1:
            stop_event.set()
        return _chunk_result(index, stop=stop_event.is_set(), empty=index == 0)

    monkeypatch.setattr(live_node, "capture_live_chunk", fake_capture)
    result = live_node.process_live_stream({
        "camera_uri": "rtsp://camera.local/live",
        "duration_seconds": 10,
        "process_every_n": 1,
        "output_dir": str(tmp_path),
        "_stop_event": stop_event,
    })

    assert indices == [0, 1]
    assert result["stream_stats"]["completed_chunks"] == 2
    assert "captured no frames; continuing" in " ".join(
        result["stream_stats"]["chunks"][0]["warnings"]
    )
    assert result["stream_stats"]["warnings"] == []


def test_long_outage_keeps_warning_storage_bounded(monkeypatch, tmp_path):
    clock = FakeClock()
    buffer = FakeBuffer(clock)
    stop_event = threading.Event()
    _install_continuous_dependencies(monkeypatch, tmp_path, buffer)

    def fake_capture(**kwargs):
        index = kwargs["chunk_index"]
        if index == 99:
            stop_event.set()
        return _chunk_result(
            index,
            stop=stop_event.is_set(),
            empty=True,
            frame_gap_active=True,
        )

    monkeypatch.setattr(live_node, "capture_live_chunk", fake_capture)
    result = live_node.process_live_stream({
        "camera_uri": "rtsp://camera.local/live",
        "duration_seconds": 5,
        "process_every_n": 1,
        "output_dir": str(tmp_path),
        "_stop_event": stop_event,
    })

    stats = result["stream_stats"]
    stored_chunk_warnings = [
        warning
        for chunk in stats["chunks"]
        for warning in chunk["warnings"]
    ]
    assert stats["completed_chunks"] == 100
    assert stats["consecutive_empty_chunks"] == 100
    assert len(stats["warnings"]) == 2
    assert sum("No camera frames arrived" in item for item in stored_chunk_warnings) == 1
    assert sum("captured no frames" in item for item in stored_chunk_warnings) == 1


def test_outage_warnings_clear_after_frames_resume(monkeypatch, tmp_path):
    clock = FakeClock()
    buffer = FakeBuffer(clock)
    stop_event = threading.Event()
    _install_continuous_dependencies(monkeypatch, tmp_path, buffer)

    def fake_capture(**kwargs):
        index = kwargs["chunk_index"]
        if index < 2:
            return _chunk_result(index, empty=True, frame_gap_active=True)
        stop_event.set()
        return _chunk_result(index, stop=True)

    monkeypatch.setattr(live_node, "capture_live_chunk", fake_capture)
    result = live_node.process_live_stream({
        "camera_uri": "rtsp://camera.local/live",
        "duration_seconds": 5,
        "process_every_n": 1,
        "output_dir": str(tmp_path),
        "_stop_event": stop_event,
    })

    assert result["stream_stats"]["consecutive_empty_chunks"] == 0
    assert not any(
        "No camera frames arrived" in warning
        or "captured no frames" in warning
        for warning in result["stream_stats"]["warnings"]
    )


def test_reconnect_window_preserves_evidence_and_cannot_finish_session(
    monkeypatch,
    tmp_path,
):
    clock = FakeClock()
    buffer = FakeBuffer(clock)
    stop_event = threading.Event()
    chunk_indices = []
    reconnect_updates = []
    _install_continuous_dependencies(monkeypatch, tmp_path, buffer)

    def fake_capture(**kwargs):
        index = kwargs["chunk_index"]
        chunk_indices.append(index)
        if index == 1:
            buffer.stream_state = "reconnecting"
            buffer.stream_reconnect_count = 1
            buffer.stream_warning = "Temporary camera interruption; reconnecting."
            return _chunk_result(index, empty=True)
        buffer.stream_state = "connected"
        buffer.stream_warning = None
        if index == 2:
            stop_event.set()
        return _chunk_result(index, stop=stop_event.is_set())

    monkeypatch.setattr(live_node, "capture_live_chunk", fake_capture)
    result = live_node.process_live_stream({
        "camera_uri": "rtsp://camera.local/live",
        "duration_seconds": 5,
        "process_every_n": 1,
        "output_dir": str(tmp_path),
        "_stop_event": stop_event,
        "_status_callback": lambda _status, update=None: (
            reconnect_updates.append(update)
            if isinstance(update, dict)
            and update.get("stream_stats", {}).get("stream_state") == "reconnecting"
            else None
        ),
    })

    assert chunk_indices == [0, 1, 2]
    assert [item["path"] for item in result["body_crops"]] == [
        "body-0.jpg",
        "body-2.jpg",
    ]
    assert [item["path"] for item in result["face_crops"]] == [
        "face-0.jpg",
        "face-2.jpg",
    ]
    assert reconnect_updates
    assert result["stream_stats"]["stop_requested"] is True
    assert (tmp_path / "stream_report.json").exists()


def test_dead_reader_without_stop_is_job_error_not_normal_completion(
    monkeypatch,
    tmp_path,
):
    clock = FakeClock()
    buffer = FakeBuffer(clock)
    buffer.ended = True
    stop_event = threading.Event()
    _install_continuous_dependencies(monkeypatch, tmp_path, buffer)

    with pytest.raises(OSError, match="reader stopped unexpectedly"):
        live_node.process_live_stream({
            "camera_uri": "rtsp://camera.local/live",
            "duration_seconds": 5,
            "process_every_n": 1,
            "output_dir": str(tmp_path),
            "_stop_event": stop_event,
        })

    assert stop_event.is_set() is False
    assert buffer.stop_calls == 1
    assert not (tmp_path / "stream_report.json").exists()


def test_pre_set_stop_starts_and_stops_once_without_capturing(monkeypatch, tmp_path):
    clock = FakeClock()
    buffer = FakeBuffer(clock)
    stop_event = threading.Event()
    stop_event.set()
    _install_continuous_dependencies(monkeypatch, tmp_path, buffer)
    monkeypatch.setattr(
        live_node,
        "capture_live_chunk",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("chunk started")),
    )

    result = live_node.process_live_stream({
        "camera_uri": "rtsp://camera.local/live",
        "duration_seconds": 10,
        "process_every_n": 1,
        "output_dir": str(tmp_path),
        "_stop_event": stop_event,
    })

    assert buffer.start_calls == 1
    assert buffer.stop_calls == 1
    assert result["body_crops"] == []
    assert result["face_crops"] == []
    assert result["stream_stats"]["completed_chunks"] == 0
    assert result["stream_stats"]["stop_requested"] is True


def test_chunk_exception_stops_buffer_and_propagates(monkeypatch, tmp_path):
    clock = FakeClock()
    buffer = FakeBuffer(clock)
    stop_event = threading.Event()
    _install_continuous_dependencies(monkeypatch, tmp_path, buffer)
    monkeypatch.setattr(
        live_node,
        "capture_live_chunk",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("detector failed")),
    )

    with pytest.raises(RuntimeError, match="detector failed"):
        live_node.process_live_stream({
            "camera_uri": "rtsp://camera.local/live",
            "duration_seconds": 10,
            "process_every_n": 1,
            "output_dir": str(tmp_path),
            "_stop_event": stop_event,
        })

    assert buffer.start_calls == 1
    assert buffer.stop_calls == 1


def test_live_reader_lifecycle_failure_blocks_node_return_and_preserves_staging(
    monkeypatch,
    tmp_path,
):
    clock = FakeClock()
    buffer = FakeBuffer(clock)
    stop_event = threading.Event()
    staging = tmp_path / "_staging"
    staging.mkdir()
    marker = staging / "keep.txt"
    marker.write_text("preserve", encoding="utf-8")
    _install_continuous_dependencies(monkeypatch, tmp_path, buffer)

    def fail_closed_stop():
        buffer.stop_calls += 1
        raise LiveFrameBufferLifecycleError(
            "reader alive; staging must be preserved"
        )

    buffer.stop = fail_closed_stop
    monkeypatch.setattr(
        live_node,
        "capture_live_chunk",
        lambda **kwargs: (
            stop_event.set() or _chunk_result(kwargs["chunk_index"], stop=True, empty=True)
        ),
    )

    with pytest.raises(LiveFrameBufferLifecycleError, match="staging"):
        live_node.process_live_stream({
            "camera_uri": "rtsp://camera.local/live",
            "duration_seconds": 5,
            "process_every_n": 1,
            "output_dir": str(tmp_path),
            "_stop_event": stop_event,
        })

    assert buffer.stop_calls == 1
    assert marker.exists()
    assert not (tmp_path / "stream_report.json").exists()


def test_process_live_stream_does_not_run_downstream_graph_nodes():
    names = set(live_node.process_live_stream.__code__.co_names)

    assert names.isdisjoint({
        "build_graph",
        "filter_quality",
        "embed_all_faces",
        "cluster_identities",
        "assign_bodies_to_clusters",
        "finalize",
    })


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, False), ("0", False), ("false", False), ("1", True), ("true", True)],
)
def test_live_overlap_feature_flag(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("PERSON_CREATION_LIVE_OVERLAP", raising=False)
    else:
        monkeypatch.setenv("PERSON_CREATION_LIVE_OVERLAP", value)
    assert live_node.live_overlap_enabled() is expected


def test_overlap_captures_next_chunk_while_previous_preprocessing_is_blocked(
    monkeypatch, tmp_path
):
    from forensics.person_creation.live_chunk_processing import PreprocessedLiveChunk
    from forensics.person_creation import live_session

    clock = FakeClock()
    buffer = FakeBuffer(clock)
    stop_event = threading.Event()
    preprocessing_started = threading.Event()
    release_preprocessing = threading.Event()
    second_capture_completed = threading.Event()
    processed = []
    outcome = {}
    _install_continuous_dependencies(monkeypatch, tmp_path, buffer)
    monkeypatch.setenv("PERSON_CREATION_LIVE_OVERLAP", "1")
    monkeypatch.setenv("PERSON_CREATION_LIVE_ROLLING_ANALYSIS", "0")

    def fake_preprocess(*, chunk, **_kwargs):
        if chunk.chunk_index == 0:
            preprocessing_started.set()
            assert release_preprocessing.wait(5.0)
        processed.append(chunk.chunk_index)
        return PreprocessedLiveChunk(
            chunk_index=chunk.chunk_index,
            capture_summary=chunk.report_metrics(),
            quality_body_crops=list(chunk.body_crops),
            quality_face_crops=list(chunk.face_crops),
            face_embeddings=[],
            failed_face_embeddings=[],
            warnings=[],
            processing_elapsed_seconds=0.01,
        )

    def fake_capture(**kwargs):
        index = kwargs["chunk_index"]
        if index == 1:
            assert preprocessing_started.wait(1.0)
            second_capture_completed.set()
            stop_event.set()
        return _chunk_result(index, stop=index == 1)

    monkeypatch.setattr(live_session, "preprocess_live_chunk", fake_preprocess)
    monkeypatch.setattr(live_node, "capture_live_chunk", fake_capture)

    def run() -> None:
        try:
            outcome["result"] = live_node.process_live_stream({
                "camera_uri": "rtsp://user:secret@camera.local/live",
                "duration_seconds": 10,
                "process_every_n": 1,
                "live_stream_config": {},
                "output_dir": str(tmp_path),
                "_stop_event": stop_event,
            })
        except Exception as exc:  # pragma: no cover - assertion reports details
            outcome["error"] = exc

    pipeline = threading.Thread(target=run)
    pipeline.start()
    assert preprocessing_started.wait(1.0)
    assert second_capture_completed.wait(2.0)
    assert not release_preprocessing.is_set()
    release_preprocessing.set()
    pipeline.join(3.0)

    assert not pipeline.is_alive()
    assert "error" not in outcome
    assert processed == [0, 1]
    preview = outcome["result"]["stream_stats"]["live_preprocessing"]
    assert preview["capture_completed_chunks"] == 2
    assert preview["preprocessing_completed_chunks"] == 2
    assert not any(
        thread.name == "person-creation-live-preprocessing"
        for thread in threading.enumerate()
    )

    canonical_tail_started = True
    assert canonical_tail_started


def test_disabled_overlap_does_not_construct_preprocessing_session(
    monkeypatch, tmp_path
):
    from forensics.person_creation import live_session

    clock = FakeClock()
    buffer = FakeBuffer(clock)
    stop_event = threading.Event()
    _install_continuous_dependencies(monkeypatch, tmp_path, buffer)
    monkeypatch.setenv("PERSON_CREATION_LIVE_OVERLAP", "0")
    monkeypatch.setattr(
        live_session,
        "LivePreprocessingSession",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("preprocessing session constructed")
        ),
    )

    def fake_capture(**kwargs):
        stop_event.set()
        return _chunk_result(kwargs["chunk_index"], stop=True)

    monkeypatch.setattr(live_node, "capture_live_chunk", fake_capture)
    result = live_node.process_live_stream({
        "camera_uri": "rtsp://camera.local/live",
        "duration_seconds": 10,
        "process_every_n": 1,
        "live_stream_config": {},
        "output_dir": str(tmp_path),
        "_stop_event": stop_event,
    })

    assert "live_preprocessing" not in result["stream_stats"]


def test_preprocessing_failure_stops_capture_and_prevents_return(
    monkeypatch, tmp_path
):
    from forensics.person_creation import live_session
    from forensics.person_creation.live_session import LivePreprocessingSessionError

    clock = FakeClock()
    buffer = FakeBuffer(clock)
    stop_event = threading.Event()
    _install_continuous_dependencies(monkeypatch, tmp_path, buffer)
    monkeypatch.setenv("PERSON_CREATION_LIVE_OVERLAP", "1")
    monkeypatch.setattr(
        live_session,
        "preprocess_live_chunk",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("embedding failed")),
    )
    monkeypatch.setattr(
        live_node,
        "capture_live_chunk",
        lambda **kwargs: _chunk_result(kwargs["chunk_index"], stop=True),
    )

    with pytest.raises(LivePreprocessingSessionError, match="embedding failed"):
        live_node.process_live_stream({
            "camera_uri": "rtsp://camera.local/live",
            "duration_seconds": 10,
            "process_every_n": 1,
            "live_stream_config": {},
            "output_dir": str(tmp_path),
            "_stop_event": stop_event,
        })

    assert stop_event.is_set()
    assert buffer.stop_calls == 1
    assert not (tmp_path / "stream_report.json").exists()


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, False), ("0", False), ("false", False), ("1", True), ("true", True)],
)
def test_live_rolling_analysis_feature_flag(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("PERSON_CREATION_LIVE_ROLLING_ANALYSIS", raising=False)
    else:
        monkeypatch.setenv("PERSON_CREATION_LIVE_ROLLING_ANALYSIS", value)
    assert live_node.live_rolling_analysis_enabled() is expected


def test_rolling_analysis_defaults_to_enabled_overlap_lane(monkeypatch):
    monkeypatch.setenv("PERSON_CREATION_LIVE_OVERLAP", "1")
    monkeypatch.delenv("PERSON_CREATION_LIVE_ROLLING_ANALYSIS", raising=False)

    assert live_node.live_rolling_analysis_enabled() is True


def test_live_chunk_without_usable_faces_keeps_rolling_status_empty(
    monkeypatch,
    tmp_path,
):
    from forensics.person_creation import live_session
    from forensics.person_creation.live_chunk_processing import PreprocessedLiveChunk

    clock = FakeClock()
    buffer = FakeBuffer(clock)
    stop_event = threading.Event()
    rolling_updates = []
    _install_continuous_dependencies(monkeypatch, tmp_path, buffer)
    monkeypatch.setenv("PERSON_CREATION_LIVE_OVERLAP", "1")
    monkeypatch.delenv("PERSON_CREATION_LIVE_ROLLING_ANALYSIS", raising=False)

    def preprocess(*, chunk, **_kwargs):
        return PreprocessedLiveChunk(
            chunk_index=chunk.chunk_index,
            capture_summary=chunk.report_metrics(),
            quality_body_crops=[],
            quality_face_crops=[],
            face_embeddings=[],
            failed_face_embeddings=[],
            warnings=[],
            processing_elapsed_seconds=0.01,
        )

    def capture(**kwargs):
        stop_event.set()
        return _chunk_result(kwargs["chunk_index"], stop=True, empty=True)

    def notify(_status, update=None):
        if isinstance(update, dict) and "rolling_analysis" in update:
            rolling_updates.append(deepcopy(update["rolling_analysis"]))

    monkeypatch.setattr(live_session, "preprocess_live_chunk", preprocess)
    monkeypatch.setattr(live_node, "capture_live_chunk", capture)

    result = live_node.process_live_stream({
        "camera_uri": "rtsp://camera.local/live",
        "duration_seconds": 10,
        "process_every_n": 1,
        "output_dir": str(tmp_path),
        "_stop_event": stop_event,
        "_status_callback": notify,
    })

    assert rolling_updates
    assert all(snapshot == {} for snapshot in rolling_updates)
    assert result["stream_stats"]["rolling_analysis"] == {}
    assert result["live_identity_decisions"] == []


def test_live_embeddings_reach_rolling_identity_decision_and_async_vlm_before_stop(
    monkeypatch,
    tmp_path,
):
    from forensics.global_memory.identity_policy import IdentityPolicyConfig
    from forensics.person_creation import live_analysis, live_session, live_vlm, service
    from forensics.person_creation.live_chunk_processing import PreprocessedLiveChunk

    clock = FakeClock()
    buffer = FakeBuffer(clock)
    stop_event = threading.Event()
    second_capture_started = threading.Event()
    first_identity_ready = threading.Event()
    second_identity_ready = threading.Event()
    no_face_update_ready = threading.Event()
    vlm_received_identity = threading.Event()
    session_root = tmp_path / "person_db" / "session"
    staging_faces = session_root / "_staging" / "face_crops"
    staging_bodies = session_root / "_staging" / "body_crops"
    staging_faces.mkdir(parents=True)
    staging_bodies.mkdir(parents=True)
    database = tmp_path / "memory.db"
    job_id = "job-live-integration"
    rolling_updates = []
    described_jobs = []
    persisted_jobs = []

    _install_continuous_dependencies(monkeypatch, session_root, buffer)
    monkeypatch.setenv("PERSON_CREATION_LIVE_OVERLAP", "1")
    monkeypatch.delenv("PERSON_CREATION_LIVE_ROLLING_ANALYSIS", raising=False)
    monkeypatch.setenv("PERSON_CREATION_MEDIA_ROOT", str(tmp_path / "person_db"))
    monkeypatch.setenv("FORENSICS_MEMORY_DB", str(database))

    def face_record(path, index):
        return {
            "path": str(path),
            "frame_idx": index,
            "video": "camera-source",
            "bbox": [0, 0, 80, 80],
            "sharpness": 100.0 + index,
        }

    def embedding_record(path, index):
        record = face_record(path, index)
        record["crop_path"] = record.pop("path")
        record["embedding"] = [1.0, float(index) / 1000.0]
        return record

    def fake_preprocess(*, chunk, **_kwargs):
        if chunk.chunk_index == 0:
            indices = range(6)
        elif chunk.chunk_index == 1:
            indices = range(6, 7)
        else:
            indices = ()
        faces = []
        embeddings = []
        for index in indices:
            path = staging_faces / f"face-{index}.jpg"
            path.write_bytes(f"face:{index}".encode())
            faces.append(face_record(path, index))
            embeddings.append(embedding_record(path, index))
        bodies = []
        if chunk.chunk_index == 0:
            body = staging_bodies / "body-0.jpg"
            body.write_bytes(b"body:0")
            bodies.append({
                "path": str(body),
                "frame_idx": 0,
                "video": "camera-source",
                "bbox": [0, 0, 80, 160],
                "sharpness": 120.0,
            })
        return PreprocessedLiveChunk(
            chunk_index=chunk.chunk_index,
            capture_summary=chunk.report_metrics(),
            quality_body_crops=bodies,
            quality_face_crops=faces,
            face_embeddings=embeddings,
            failed_face_embeddings=[],
            warnings=[],
            processing_elapsed_seconds=0.01,
        )

    def cluster_all(state):
        records = list(state["all_face_embeddings"])
        if not records:
            return {"identity_clusters": [], "unresolved_faces": []}
        if len(records) == 6:
            assert second_capture_started.wait(5.0)
        return {
            "identity_clusters": [{
                "cluster_id": 0,
                "face_records": records,
                "representative_embedding": [1.0, 0.0],
                "face_count": len(records),
                "confidence": 1.0,
                "low_confidence": False,
            }],
            "unresolved_faces": [],
        }

    def associate(state):
        bodies = list(state.get("quality_body_crops") or [])
        assignments = [{
            "body_crop_path": record["path"],
            "body_sharpness": record["sharpness"],
        } for record in bodies]
        return SimpleNamespace(cluster_assignments={0: assignments})

    real_analysis = live_analysis.LiveRollingAnalysisSession

    def analysis_factory(**kwargs):
        kwargs.update({
            "database_path": database,
            "cluster": cluster_all,
            "associate": associate,
            "identity_decisions": True,
            "policy_config": IdentityPolicyConfig(),
        })
        return real_analysis(**kwargs)

    real_vlm = live_vlm.LiveIdentityVLMCoordinator

    def describe(job):
        described_jobs.append(job)
        vlm_received_identity.set()
        return {
            "per_cluster_clothing": {0: {
                "status": "ok",
                "attempts": 1,
                "top": "black jacket",
                "bottom": "blue jeans",
                "shoes": "white shoes",
                "full": "black jacket, blue jeans, white shoes",
                "failure_reason": None,
            }},
            "clothing_diagnostics": [{
                "cluster_id": 0,
                "attempts": 1,
                "status": "ok",
                "failure_reason": None,
            }],
        }

    def persist(job, _clothing):
        persisted_jobs.append(job)

    def vlm_factory(**kwargs):
        return real_vlm(
            **kwargs,
            media_root=tmp_path / "person_db",
            describe=describe,
            persist=persist,
        )

    monkeypatch.setattr(live_session, "preprocess_live_chunk", fake_preprocess)
    monkeypatch.setattr(live_analysis, "LiveRollingAnalysisSession", analysis_factory)
    monkeypatch.setattr(live_vlm, "LiveIdentityVLMCoordinator", vlm_factory)

    with service._jobs_lock:
        service._jobs[job_id] = service.JobState(
            job_id,
            input_type="camera_uri",
            output_dir=str(session_root),
        )

    def publish_status(status, update=None):
        if not isinstance(update, dict):
            return
        with service._jobs_lock:
            job = service._jobs[job_id]
            job.status = status
            service._merge_job_snapshot(job, update)
        rolling = update.get("rolling_analysis")
        if not isinstance(rolling, dict):
            return
        rolling_updates.append((stop_event.is_set(), deepcopy(rolling)))
        identities = rolling.get("live_identities") or []
        if not identities:
            return
        identity = identities[0]
        if (
            identity.get("face_count") == 6
            and identity.get("decision") == "new_person"
            and identity.get("canonical_person_id")
        ):
            first_identity_ready.set()
        if identity.get("face_count") == 7:
            if int(rolling.get("analysis_version") or 0) >= 3:
                no_face_update_ready.set()
            else:
                second_identity_ready.set()

    def fake_capture(**kwargs):
        index = kwargs["chunk_index"]
        if index == 1:
            second_capture_started.set()
            assert first_identity_ready.wait(5.0)
            assert vlm_received_identity.wait(5.0)
        elif index == 2:
            assert second_identity_ready.wait(5.0)
        elif index == 3:
            assert no_face_update_ready.wait(5.0)
            stop_event.set()
            return _chunk_result(index, stop=True, empty=True)
        return _chunk_result(index, empty=index >= 2)

    monkeypatch.setattr(live_node, "capture_live_chunk", fake_capture)
    try:
        result = live_node.process_live_stream({
            "camera_uri": "rtsp://camera.local/live",
            "duration_seconds": 10,
            "process_every_n": 1,
            "live_stream_config": {"vlm_drain_timeout_seconds": 2.0},
            "output_dir": str(session_root),
            "_job_id": job_id,
            "_stop_event": stop_event,
            "_status_callback": publish_status,
        })
        payload = service.app.test_client().get(
            f"/api/person/status/{job_id}"
        ).get_json()
    finally:
        with service._jobs_lock:
            service._jobs.pop(job_id, None)

    before_stop = [
        snapshot for stopped, snapshot in rolling_updates
        if not stopped and snapshot.get("live_identities")
    ]
    assert before_stop
    assert {item["live_identities"][0]["live_identity_id"] for item in before_stop} == {
        "live_0001"
    }
    assert any(item["live_identities"][0]["face_count"] == 6 for item in before_stop)
    assert any(item["live_identities"][0]["face_count"] == 7 for item in before_stop)
    no_face_snapshot = next(
        item for item in before_stop
        if int(item.get("analysis_version") or 0) >= 3
    )
    assert no_face_snapshot["live_identities"][0]["live_identity_id"] == "live_0001"
    assert no_face_snapshot["live_identities"][0]["face_count"] == 7

    decisions = result["live_identity_decisions"]
    assert decisions and decisions[0]["live_identity_id"] == "live_0001"
    assert decisions[0]["canonical_person_id"].startswith("person_")
    assert described_jobs and persisted_jobs
    assert described_jobs[0].live_identity_id == "live_0001"
    assert described_jobs[0].canonical_person_id == decisions[0]["canonical_person_id"]

    rolling = payload["snapshot"]["rolling_analysis"]
    assert rolling["live_identities"][0]["live_identity_id"] == "live_0001"
    assert rolling["live_identities"][0]["decision"] == "new_person"
    assert rolling["vlm_completed"] >= 1
    assert payload["snapshot"]["live_preprocessing"]["embedded_faces"] == 7


def test_rolling_without_overlap_is_rejected_before_camera_or_staging(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("PERSON_CREATION_LIVE_OVERLAP", "0")
    monkeypatch.setenv("PERSON_CREATION_LIVE_ROLLING_ANALYSIS", "1")
    monkeypatch.setattr(
        live_node,
        "prepare_staging_dirs",
        lambda _path: (_ for _ in ()).throw(AssertionError("staging opened")),
    )

    with pytest.raises(ValueError, match="requires PERSON_CREATION_LIVE_OVERLAP=1"):
        live_node.process_live_stream({
            "camera_uri": "rtsp://camera.local/live",
            "duration_seconds": 5,
            "process_every_n": 1,
            "output_dir": str(tmp_path),
        })


def test_rolling_disabled_does_not_construct_analysis_worker(monkeypatch, tmp_path):
    from forensics.person_creation import live_analysis

    clock = FakeClock()
    buffer = FakeBuffer(clock)
    stop_event = threading.Event()
    _install_continuous_dependencies(monkeypatch, tmp_path, buffer)
    monkeypatch.setenv("PERSON_CREATION_LIVE_OVERLAP", "1")
    monkeypatch.setenv("PERSON_CREATION_LIVE_ROLLING_ANALYSIS", "0")
    monkeypatch.setattr(
        live_analysis,
        "LiveRollingAnalysisSession",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("rolling worker constructed")
        ),
    )
    monkeypatch.setattr(
        live_node,
        "capture_live_chunk",
        lambda **kwargs: (
            stop_event.set() or _chunk_result(kwargs["chunk_index"], stop=True, empty=True)
        ),
    )

    result = live_node.process_live_stream({
        "camera_uri": "rtsp://camera.local/live",
        "duration_seconds": 5,
        "process_every_n": 1,
        "output_dir": str(tmp_path),
        "_stop_event": stop_event,
    })

    assert "rolling_analysis" not in result


def test_hung_analysis_worker_prevents_live_node_return(monkeypatch, tmp_path):
    from forensics.person_creation import live_analysis, live_session

    clock = FakeClock()
    buffer = FakeBuffer(clock)
    stop_event = threading.Event()
    _install_continuous_dependencies(monkeypatch, tmp_path, buffer)
    staging = tmp_path / "_staging"
    staging.mkdir()
    marker = staging / "keep.txt"
    marker.write_text("preserve", encoding="utf-8")
    monkeypatch.setenv("PERSON_CREATION_LIVE_OVERLAP", "1")
    monkeypatch.setenv("PERSON_CREATION_LIVE_ROLLING_ANALYSIS", "1")

    class FakePreprocessing:
        accumulator_version = 1

        def __init__(self, **kwargs):
            self.on_advanced = kwargs["on_accumulator_advanced"]

        def start(self):
            return None

        def submit_chunk(self, _chunk):
            self.on_advanced(1)

        def finish(self):
            return None

        def public_snapshot(self):
            return {"enabled": True}

        def analysis_snapshot(self):
            raise AssertionError("fake analysis owns snapshot behavior")

    class HungAnalysis:
        def __init__(self, **_kwargs):
            pass

        def start(self):
            return None

        def request_version(self, _version):
            return True

        def public_snapshot(self):
            return {"enabled": True}

        def finish(self, _version):
            raise live_analysis.LiveAnalysisLifecycleError(
                "worker alive; staging must be preserved"
            )

        def abort(self):
            return None

    monkeypatch.setattr(live_session, "LivePreprocessingSession", FakePreprocessing)
    monkeypatch.setattr(live_analysis, "LiveRollingAnalysisSession", HungAnalysis)
    monkeypatch.setattr(
        live_node,
        "capture_live_chunk",
        lambda **kwargs: (
            stop_event.set() or _chunk_result(kwargs["chunk_index"], stop=True, empty=True)
        ),
    )

    with pytest.raises(live_analysis.LiveAnalysisLifecycleError, match="staging"):
        live_node.process_live_stream({
            "camera_uri": "rtsp://camera.local/live",
            "duration_seconds": 5,
            "process_every_n": 1,
            "output_dir": str(tmp_path),
            "_stop_event": stop_event,
        })

    assert not (tmp_path / "stream_report.json").exists()
    assert marker.exists()


def test_dead_analysis_worker_waits_for_real_user_stop(monkeypatch, tmp_path):
    from forensics.person_creation import live_analysis, live_session

    clock = FakeClock()
    buffer = FakeBuffer(clock)
    stop_event = threading.Event()
    capture_continued = threading.Event()
    allow_user_stop = threading.Event()
    notifications = []
    outcome = {}
    _install_continuous_dependencies(monkeypatch, tmp_path, buffer)
    monkeypatch.setenv("PERSON_CREATION_LIVE_OVERLAP", "1")
    monkeypatch.setenv("PERSON_CREATION_LIVE_ROLLING_ANALYSIS", "1")

    class FakePreprocessing:
        def __init__(self, **kwargs):
            self.on_advanced = kwargs["on_accumulator_advanced"]
            self.accumulator_version = 0

        def start(self):
            return None

        def submit_chunk(self, _chunk):
            self.accumulator_version += 1
            self.on_advanced(self.accumulator_version)

        def finish(self):
            return None

        def public_snapshot(self):
            return {"enabled": True}

        def analysis_snapshot(self):
            raise AssertionError("dead worker cannot request snapshots")

    class DeadAnalysis:
        def __init__(self, **kwargs):
            assert "request_stop" not in kwargs
            self.requested = 0
            self.failed = False
            self.identity = {
                "session_person_id": "live_0001",
                "status": "provisional",
                "face_count": 3,
                "associated_body_count": 2,
                "memory_match": None,
            }

        def start(self):
            return None

        def request_version(self, version):
            self.requested = version
            self.failed = True
            return False

        def public_snapshot(self):
            return {
                "enabled": True,
                "publication_sequence": 2,
                "requested_version": self.requested,
                "analysis_version": 1,
                "analysis_state": "worker_failed" if self.failed else "ready",
                "analysis_in_progress": False,
                "analyzed_embedding_count": 3,
                "last_completed_preprocessing_chunk": 0,
                "analysis_warning": (
                    "Rolling analysis worker failed."
                    if self.failed
                    else None
                ),
                "live_identities": [self.identity],
                "live_recognition_events": [],
            }

        def finish(self, _version):
            return None

        def abort(self):
            return None

    monkeypatch.setattr(live_session, "LivePreprocessingSession", FakePreprocessing)
    monkeypatch.setattr(live_analysis, "LiveRollingAnalysisSession", DeadAnalysis)

    def fake_capture(**kwargs):
        index = kwargs["chunk_index"]
        if index == 1:
            capture_continued.set()
            assert allow_user_stop.wait(2.0)
        return _chunk_result(index, stop=stop_event.is_set(), empty=True)

    monkeypatch.setattr(live_node, "capture_live_chunk", fake_capture)

    def run_pipeline():
        try:
            outcome["result"] = live_node.process_live_stream({
                "camera_uri": "rtsp://camera.local/live",
                "duration_seconds": 5,
                "process_every_n": 1,
                "output_dir": str(tmp_path),
                "_stop_event": stop_event,
                "_status_callback": lambda status, update=None: notifications.append(
                    (status, update)
                ),
            })
        except Exception as exc:  # pragma: no cover - assertion reports details
            outcome["error"] = exc

    pipeline = threading.Thread(target=run_pipeline)
    pipeline.start()
    try:
        assert capture_continued.wait(1.0)
        assert not stop_event.is_set()
        assert pipeline.is_alive()
        assert outcome == {}
        assert not (tmp_path / "stream_report.json").exists()
        rolling_updates = [
            update["rolling_analysis"]
            for _status, update in notifications
            if isinstance(update, dict) and "rolling_analysis" in update
        ]
        assert rolling_updates[-1]["analysis_state"] == "worker_failed"
        assert rolling_updates[-1]["live_identities"] == [{
            "session_person_id": "live_0001",
            "status": "provisional",
            "face_count": 3,
            "associated_body_count": 2,
            "memory_match": None,
        }]

        stop_event.set()
        allow_user_stop.set()
        pipeline.join(2.0)
    finally:
        stop_event.set()
        allow_user_stop.set()
        pipeline.join(2.0)

    assert not pipeline.is_alive()
    assert "error" not in outcome
    result = outcome["result"]
    assert result["stream_stats"]["completed_chunks"] == 2
    report = json.loads((tmp_path / "stream_report.json").read_text(encoding="utf-8"))
    assert report["rolling_analysis"]["analysis_state"] == "worker_failed"
    assert report["rolling_analysis"]["live_identities"][0][
        "session_person_id"
    ] == "live_0001"


def test_recoverable_rolling_warning_allows_live_node_to_return(
    monkeypatch,
    tmp_path,
):
    from forensics.person_creation import live_analysis, live_session

    clock = FakeClock()
    buffer = FakeBuffer(clock)
    stop_event = threading.Event()
    notifications = []
    _install_continuous_dependencies(monkeypatch, tmp_path, buffer)
    monkeypatch.setenv("PERSON_CREATION_LIVE_OVERLAP", "1")
    monkeypatch.setenv("PERSON_CREATION_LIVE_ROLLING_ANALYSIS", "1")

    class FakePreprocessing:
        accumulator_version = 1

        def __init__(self, **kwargs):
            self.on_advanced = kwargs["on_accumulator_advanced"]

        def start(self):
            return None

        def submit_chunk(self, _chunk):
            self.on_advanced(1)

        def finish(self):
            return None

        def public_snapshot(self):
            return {"enabled": True}

        def analysis_snapshot(self):
            raise AssertionError("fake analysis owns snapshots")

    class WarningAnalysis:
        def __init__(self, **_kwargs):
            self.dead = False

        def start(self):
            return None

        def request_version(self, _version):
            return True

        def public_snapshot(self):
            return {
                "enabled": True,
                "publication_sequence": 2,
                "requested_version": 1,
                "analysis_version": 0,
                "analysis_state": "warning",
                "analysis_in_progress": False,
                "analyzed_embedding_count": 0,
                "last_completed_preprocessing_chunk": 0,
                "analysis_warning": "Rolling analysis failed (RuntimeError).",
                "live_identities": [],
                "live_recognition_events": [],
            }

        def finish(self, _version):
            self.dead = True

        def abort(self):
            self.dead = True

    monkeypatch.setattr(live_session, "LivePreprocessingSession", FakePreprocessing)
    monkeypatch.setattr(live_analysis, "LiveRollingAnalysisSession", WarningAnalysis)
    monkeypatch.setattr(
        live_node,
        "capture_live_chunk",
        lambda **kwargs: (
            stop_event.set() or _chunk_result(kwargs["chunk_index"], stop=True, empty=True)
        ),
    )

    result = live_node.process_live_stream({
        "camera_uri": "rtsp://camera.local/live",
        "duration_seconds": 5,
        "process_every_n": 1,
        "output_dir": str(tmp_path),
        "_stop_event": stop_event,
        "_status_callback": lambda status, update=None: notifications.append(
            (status, update)
        ),
    })

    rolling_updates = [
        update["rolling_analysis"]
        for _status, update in notifications
        if isinstance(update, dict) and "rolling_analysis" in update
    ]
    assert rolling_updates[-1]["analysis_state"] == "warning"
    assert (tmp_path / "stream_report.json").exists()

    assert result["stream_stats"]["completed_chunks"] == 1
    assert (tmp_path / "stream_report.json").exists()
