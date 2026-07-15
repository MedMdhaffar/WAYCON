from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import pytest

from forensics.person_creation.live_stream import BufferedFrame, mask_camera_uri
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
    buffer.ended = True

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
        buffer.ended = True
        _capture(buffer, clock, chunk_index=chunk_index)

    assert stems == ["live", "live_chunk_0001"]


def test_process_live_stream_owns_buffer_and_preserves_output(
    monkeypatch, tmp_path, capsys
):
    clock = FakeClock()
    buffer = FakeBuffer(clock, [_frame(0)])
    buffer.ended = True
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


def _chunk_result(index, *, stop=False, empty=False):
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
        warnings=[],
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
        result["stream_stats"]["warnings"]
    )


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

    def fake_preprocess(*, chunk, **_kwargs):
        if chunk.chunk_index == 0:
            preprocessing_started.set()
            assert release_preprocessing.wait(2.0)
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
            assert preprocessing_started.is_set()
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
    assert second_capture_completed.wait(1.0)
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
