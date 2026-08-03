from copy import deepcopy
from datetime import datetime, timezone

import pytest

from forensics.person_creation import service


@pytest.fixture
def live_job(tmp_path, monkeypatch):
    media_root = tmp_path / "person_db"
    face = media_root / "person_001" / "face_crops" / "face.jpg"
    body = media_root / "person_001" / "body_crops" / "body.jpg"
    face.parent.mkdir(parents=True)
    body.parent.mkdir(parents=True)
    face.write_bytes(b"face")
    body.write_bytes(b"body")
    monkeypatch.setenv("PERSON_CREATION_MEDIA_ROOT", str(media_root))
    job = service.JobState(
        "live-publication",
        input_type="camera_uri",
        status="processing_live_frames",
        output_dir=str(media_root / "_session"),
    )
    with service._jobs_lock:
        service._jobs[job.job_id] = job
    try:
        yield job
    finally:
        with service._jobs_lock:
            service._jobs.pop(job.job_id, None)


def _identity(*, decision="new_person", vlm_status="queued"):
    return {
        "live_identity_id": "live_0001",
        "session_person_id": "live_0001",
        "version": 2,
        "evidence_version": 4,
        "cluster_label": 0,
        "clustering_state": "resolved",
        "decision": decision,
        "reason": "stable_new_person",
        "provisional": vlm_status != "completed",
        "persisted": True,
        "canonical_person_id": "person_001",
        "face_count": 4,
        "observation_count": 4,
        "best_face_path": "person_001/face_crops/face.jpg",
        "representative_face_path": "person_001/face_crops/face.jpg",
        "best_body_path": "person_001/body_crops/body.jpg",
        "selected_body_crop": "person_001/body_crops/body.jpg",
        "vlm_status": vlm_status,
        "vlm_state": {
            "queued": "pending",
            "processing": "running",
            "completed": "completed",
        }.get(vlm_status, vlm_status),
        "clothing_description": (
            "black jacket" if vlm_status == "completed" else ""
        ),
    }


_DEFAULT_IDENTITY = object()


def _publication(
    sequence,
    *,
    identity=_DEFAULT_IDENTITY,
    clothing=None,
    chunk=0,
    evidence_version=4,
):
    identity = _identity() if identity is _DEFAULT_IDENTITY else identity
    rolling = {
        "enabled": True,
        "publication_sequence": sequence,
        "evidence_version": evidence_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "requested_version": evidence_version,
        "analysis_version": evidence_version,
        "analysis_state": "ready",
        "analysis_in_progress": False,
        "analyzed_embedding_count": 4,
        "resolved_cluster_count": 1 if identity else 0,
        "last_completed_preprocessing_chunk": chunk,
        "live_identities": [deepcopy(identity)] if identity else [],
        "live_recognition_events": [],
        "per_cluster_clothing": deepcopy(clothing or {}),
    }
    return {
        "identity_clusters": [{
            "cluster_id": 0,
            "face_count": 4,
            "face_records": [],
        }] if identity else [],
        "per_cluster_profiles": {
            0: {"person_id": "live_0001", "face_crops": []}
        } if identity else {},
        "per_cluster_best_body_crops": {
            0: ["person_001/body_crops/body.jpg"]
        } if identity else {},
        "per_cluster_clothing": deepcopy(clothing or {}),
        "profile": {"person_id": "live_0001"} if identity else {},
        "rolling_analysis": rolling,
    }


def test_canonical_new_person_is_visible_in_real_status_schema_before_stop(
    live_job,
):
    with service._jobs_lock:
        service._merge_job_snapshot(live_job, _publication(1))

    payload = service.app.test_client().get(
        f"/api/person/status/{live_job.job_id}"
    ).get_json()

    assert payload["status"] == "processing_live_frames"
    snapshot = payload["snapshot"]
    assert snapshot["identity_clusters"][0]["face_count"] == 4
    assert snapshot["per_cluster_profiles"]["0"]["person_id"] == "live_0001"
    assert snapshot["per_cluster_best_body_crops"]["0"]
    assert snapshot["profile"]["person_id"] == "live_0001"
    cards = snapshot["rolling_analysis"]["live_identities"]
    assert [card["live_identity_id"] for card in cards] == ["live_0001"]
    assert cards[0]["decision"] == "new_person"
    assert cards[0]["vlm_state"] == "pending"
    assert snapshot["rolling_analysis"]["publication_sequence"] == 1
    assert snapshot["rolling_analysis"]["evidence_version"] == 4
    assert snapshot["rolling_analysis"]["generated_at"]


def test_known_match_uses_the_same_live_status_card_path(live_job):
    known = _identity(decision="attach_existing")
    known["reason"] = "strong_clear_match"
    known["candidate_person_id"] = "person_001"
    known["candidate_similarity"] = 0.91
    with service._jobs_lock:
        service._merge_job_snapshot(
            live_job,
            _publication(1, identity=known),
        )

    payload = service.app.test_client().get(
        f"/api/person/status/{live_job.job_id}"
    ).get_json()
    card = payload["snapshot"]["rolling_analysis"]["live_identities"][0]
    assert card["live_identity_id"] == "live_0001"
    assert card["decision"] == "attach_existing"
    assert card["reason"] == "strong_clear_match"


def test_capture_and_stale_publications_cannot_erase_canonical_identity(live_job):
    completed = _identity(vlm_status="completed")
    clothing = {
        0: {
            "status": "ok",
            "full": "black jacket",
            "live_identity_id": "live_0001",
        },
    }
    with service._jobs_lock:
        service._merge_job_snapshot(live_job, _publication(2))
        service._merge_job_snapshot(
            live_job,
            _publication(3, identity=completed, clothing=clothing),
        )
        service._merge_job_snapshot(live_job, {
            "stream_stats": {"frames_read": 300, "faces_embedded": 12},
            "identity_clusters": [],
            "per_cluster_profiles": {},
            "profile": {},
        })
        service._merge_job_snapshot(
            live_job,
            _publication(1, identity=None),
        )

    payload = service.app.test_client().get(
        f"/api/person/status/{live_job.job_id}"
    ).get_json()
    snapshot = payload["snapshot"]
    assert snapshot["stream_stats"]["frames_read"] == 300
    assert snapshot["identity_clusters"]
    assert snapshot["per_cluster_profiles"]
    assert snapshot["profile"]
    assert snapshot["per_cluster_clothing"]["0"]["full"] == "black jacket"
    rolling = snapshot["rolling_analysis"]
    assert rolling["publication_sequence"] == 3
    assert rolling["live_identities"][0]["vlm_state"] == "completed"


def test_decreasing_core_chunk_identifiers_do_not_freeze_publication(live_job):
    chunks = (-1, -4, -7, -10)
    for sequence, chunk in enumerate(chunks, start=1):
        identity = _identity(vlm_status="completed")
        identity["publication_marker"] = sequence
        clothing = {
            0: {
                "status": "ok",
                "full": f"publication {sequence}",
                "live_identity_id": "live_0001",
                "publication_marker": sequence,
            },
        }
        publication = _publication(
            sequence,
            identity=identity,
            clothing=clothing,
            chunk=chunk,
            evidence_version=sequence,
        )
        publication["identity_clusters"][0]["publication_marker"] = sequence
        publication["per_cluster_profiles"][0]["publication_marker"] = sequence
        publication["per_cluster_best_body_crops"][0] = [
            f"person_001/body_crops/body-{sequence}.jpg"
        ]
        publication["profile"]["publication_marker"] = sequence

        with service._jobs_lock:
            service._merge_job_snapshot(live_job, publication)

        assert live_job.snapshot["rolling_analysis"][
            "publication_sequence"
        ] == sequence
        assert live_job.snapshot["rolling_analysis"][
            "last_completed_preprocessing_chunk"
        ] == chunk

    snapshot = live_job.snapshot
    assert snapshot["identity_clusters"][0]["publication_marker"] == 4
    assert snapshot["per_cluster_profiles"][0]["publication_marker"] == 4
    assert snapshot["per_cluster_best_body_crops"][0] == [
        "person_001/body_crops/body-4.jpg"
    ]
    assert snapshot["per_cluster_clothing"][0]["publication_marker"] == 4
    assert snapshot["profile"]["publication_marker"] == 4
    assert snapshot["rolling_analysis"]["live_identities"][0][
        "publication_marker"
    ] == 4


def test_status_projection_keeps_metadata_but_removes_every_raw_vector(live_job):
    publication = _publication(1)
    publication["identity_clusters"][0].update({
        "representative_embedding": [0.25] * 8,
        "face_records": [{
            "crop_path": "person_001/face_crops/face.jpg",
            "frame_idx": 7,
            "embedding": [0.5] * 8,
        }],
    })
    for profile in (
        publication["per_cluster_profiles"][0],
        publication["profile"],
    ):
        profile["face_embedding"] = [0.75] * 8
        profile["face_embedding_meta"] = {"dim": 8, "norm": "L2"}
        profile["reid"] = {
            "status": "computed",
            "embedding_dim": 8,
            "body_embedding": [0.125] * 8,
        }
    publication["reid_embeddings"] = {0: [0.125] * 8}

    with service._jobs_lock:
        service._merge_job_snapshot(live_job, publication)
    client = service.app.test_client()
    compact = client.get(f"/api/person/status/{live_job.job_id}")

    with service._jobs_lock:
        cluster = live_job.snapshot["identity_clusters"][0]
        cluster["representative_embedding"] = [0.25] * 4096
        cluster["face_records"][0]["embedding"] = [0.5] * 4096
        live_job.snapshot["per_cluster_profiles"][0][
            "face_embedding"
        ] = [0.75] * 4096
        live_job.snapshot["profile"]["face_embedding"] = [0.75] * 4096
        live_job.snapshot["reid_embeddings"] = {0: [0.125] * 4096}
    expanded = client.get(f"/api/person/status/{live_job.job_id}")
    payload = expanded.get_json()

    cluster = payload["snapshot"]["identity_clusters"][0]
    assert cluster["face_count"] == 4
    assert cluster["face_records"] == [{
        "crop_path": "person_001/face_crops/face.jpg",
        "frame_idx": 7,
    }]
    assert payload["snapshot"]["per_cluster_profiles"]["0"][
        "face_embedding_meta"
    ] == {"dim": 8, "norm": "L2"}
    assert payload["snapshot"]["rolling_analysis"]["live_identities"][0][
        "live_identity_id"
    ] == "live_0001"

    def raw_vector_keys(value):
        if isinstance(value, dict):
            for key, item in value.items():
                normalized = str(key).lower()
                if (
                    normalized in {"embedding", "embeddings"}
                    or normalized.endswith("_embedding")
                    or normalized.endswith("_embeddings")
                ):
                    yield key
                yield from raw_vector_keys(item)
        elif isinstance(value, list):
            for item in value:
                yield from raw_vector_keys(item)

    assert list(raw_vector_keys(payload["snapshot"])) == []
    assert "reid_embeddings" not in payload["snapshot"]
    assert len(expanded.data) == len(compact.data)


def test_polling_and_stop_preserve_one_stable_live_card(live_job):
    first = _publication(1)
    stopped = _publication(2)
    with service._jobs_lock:
        service._merge_job_snapshot(live_job, first)
    client = service.app.test_client()
    before = [
        client.get(f"/api/person/status/{live_job.job_id}").get_json()
        for _ in range(3)
    ]
    with service._jobs_lock:
        service._merge_job_snapshot(live_job, stopped)
        live_job.status = "stopping"
    after = client.get(
        f"/api/person/status/{live_job.job_id}"
    ).get_json()

    for payload in [*before, after]:
        cards = payload["snapshot"]["rolling_analysis"]["live_identities"]
        assert len(cards) == 1
        assert cards[0]["live_identity_id"] == "live_0001"


def test_noise_publication_has_no_status_card(live_job):
    with service._jobs_lock:
        service._merge_job_snapshot(
            live_job,
            _publication(1, identity=None),
        )
    payload = service.app.test_client().get(
        f"/api/person/status/{live_job.job_id}"
    ).get_json()
    assert payload["snapshot"]["rolling_analysis"]["live_identities"] == []


_DIAGNOSTIC_KEYS = {
    "representative_similarity_diagnostic",
    "comparisons",
    "medoid_similarity",
    "normalized_mean_similarity",
    "medoid_best_person_id",
    "normalized_mean_best_person_id",
    "similarity_drift",
    "maximum_absolute_similarity_drift",
    "compared_identity_count",
}


def _keys_anywhere(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _keys_anywhere(item)
    elif isinstance(value, list):
        for item in value:
            yield from _keys_anywhere(item)


def _enrolled_roster_diagnostic(enrolled_person_ids):
    """The exact shape the retired in-pass diagnostic used to publish."""
    return {
        "status": "compared",
        "current_strategy": "normalized_medoid",
        "previous_strategy": "normalized_mean",
        "compared_identity_count": len(enrolled_person_ids),
        "medoid_best_person_id": enrolled_person_ids[0],
        "normalized_mean_best_person_id": enrolled_person_ids[0],
        "maximum_absolute_similarity_drift": 0.000002,
        "comparisons": [{
            "person_id": person_id,
            "medoid_similarity": 0.412345,
            "normalized_mean_similarity": 0.412343,
            "similarity_drift": 0.000002,
        } for person_id in enrolled_person_ids],
    }


def test_status_never_exposes_representative_diagnostic_or_enrolled_roster(
    live_job,
):
    """A live pass must not publish medoid/mean comparisons or the roster."""
    with service._jobs_lock:
        service._merge_job_snapshot(live_job, _publication(1))
    payload = service.app.test_client().get(
        f"/api/person/status/{live_job.job_id}"
    ).get_json()

    present = _DIAGNOSTIC_KEYS & set(_keys_anywhere(payload["snapshot"]))
    assert present == set(), f"diagnostic keys leaked into status: {sorted(present)}"
    identity = payload["snapshot"]["rolling_analysis"]["live_identities"][0]
    assert identity["live_identity_id"] == "live_0001"
    assert identity["vlm_state"] == "pending"


def test_status_size_does_not_grow_with_enrolled_identity_count(live_job):
    """Payload must stay bounded as Global Memory grows.

    The retired in-pass diagnostic appended one comparison record per enrolled
    person per identity, so a 20-person deployment paid for 20 extra rows on
    every poll.  Publishing 1 versus 20 versus 200 enrolled identities must now
    produce a byte-identical response.
    """
    client = service.app.test_client()
    sizes = {}
    # Fixed-width, increasing sequence so the payload size cannot vary with the
    # number of digits in publication_sequence itself.
    for step, enrolled in enumerate((1, 20, 200), start=1):
        roster = [f"person_{index:03d}" for index in range(1, enrolled + 1)]
        publication = _publication(100 + step)
        # Simulate the retired behaviour at the publication boundary: if any
        # producer reattaches it, the projection must still drop it.
        publication["rolling_analysis"]["live_identities"][0][
            "representative_similarity_diagnostic"
        ] = _enrolled_roster_diagnostic(roster)
        with service._jobs_lock:
            service._merge_job_snapshot(live_job, publication)
        response = client.get(f"/api/person/status/{live_job.job_id}")
        payload = response.get_json()
        leaked = _DIAGNOSTIC_KEYS & set(_keys_anywhere(payload["snapshot"]))
        assert leaked == set(), (
            f"{enrolled} enrolled identities leaked {sorted(leaked)}"
        )
        assert not any(
            key == "person_id" for key in _keys_anywhere(
                payload["snapshot"]["rolling_analysis"]
            )
        )
        sizes[enrolled] = len(response.data)

    assert sizes[1] == sizes[20] == sizes[200], (
        f"status payload grew with enrolled identity count: {sizes}"
    )


def test_status_keeps_numeric_preprocessing_counters(live_job):
    """Sanitization must not remove scalar diagnostics such as counts."""
    publication = _publication(1)
    publication["live_preprocessing"] = {
        "enabled": True,
        "embedded_faces": 7,
        "failed_face_embeddings": 2,
        "quality_face_crops": 7,
        "duplicate_face_evidence_skipped": 1,
        "face_rejection_counts": {"too_small": 3, "low_sharpness": 1},
    }
    with service._jobs_lock:
        service._merge_job_snapshot(live_job, publication)
    payload = service.app.test_client().get(
        f"/api/person/status/{live_job.job_id}"
    ).get_json()

    preprocessing = payload["snapshot"]["live_preprocessing"]
    assert preprocessing["failed_face_embeddings"] == 2
    assert preprocessing["embedded_faces"] == 7
    assert preprocessing["quality_face_crops"] == 7
    assert preprocessing["duplicate_face_evidence_skipped"] == 1
    assert preprocessing["face_rejection_counts"]["too_small"] == 3


def test_offline_snapshot_merge_remains_unchanged():
    job = service.JobState("offline-publication", input_type="video_file")
    service._merge_job_snapshot(job, {
        "identity_clusters": [{"cluster_id": 0}],
        "per_cluster_profiles": {0: {"person_id": "offline"}},
        "profile": {"person_id": "offline"},
    })
    assert job.snapshot["identity_clusters"] == [{"cluster_id": 0}]
    assert job.snapshot["per_cluster_profiles"][0]["person_id"] == "offline"
    assert job.snapshot["profile"]["person_id"] == "offline"
