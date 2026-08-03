from __future__ import annotations

import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
from urllib.parse import quote

import numpy as np
import pytest

from forensics.global_memory import GlobalMemory
from forensics.person_creation import service


def _person(memory, person_id, values, *, count, name):
    vector = np.asarray(values, dtype=np.float32)
    vector /= np.linalg.norm(vector)
    memory._conn.execute(
        """
        INSERT INTO persons(
            person_id, name, embedding, embedding_count, enrolled_at,
            updated_at, cameras, profile_image, profile_image_source,
            is_active, merged_into_person_id
        ) VALUES(?, ?, ?, ?, '2026-07-01', '2026-07-02', '[]', ?,
                 'auto', 1, NULL)
        """,
        (
            person_id,
            name,
            sqlite3.Binary(vector.tobytes()),
            count,
            f"{person_id}.jpg",
        ),
    )


def _suggestion(memory, source="person_001", candidate="person_002", status="pending"):
    cursor = memory._conn.execute(
        """
        INSERT INTO identity_match_suggestions(
            source_person_id, candidate_person_id, similarity,
            second_similarity, margin, reason, status, created_at
        ) VALUES(?, ?, 0.88, 0.70, 0.18, 'ambiguous_match', ?, '2026-07-20')
        """,
        (source, candidate, status),
    )
    return int(cursor.lastrowid)


@pytest.fixture
def review_api(tmp_path, monkeypatch):
    database = tmp_path / "memory.db"
    media = tmp_path / "media"
    media.mkdir()
    monkeypatch.setenv("FORENSICS_MEMORY_DB", str(database))
    monkeypatch.setenv("PERSON_CREATION_MEDIA_ROOT", str(media))
    memory = GlobalMemory(database, media_root=media)
    for person_id in ("person_001", "person_002", "person_003", "person_004"):
        (media / f"{person_id}.jpg").write_bytes(person_id.encode())
    _person(memory, "person_001", (0, 1, 0), count=3, name="Source")
    _person(memory, "person_002", (1, 0, 0), count=5, name="Candidate")
    _person(memory, "person_003", (0, 0, 1), count=2, name="Other Source")
    _person(memory, "person_004", (1, 1, 0), count=4, name="Other Candidate")
    memory.close()
    service.app.config.update(TESTING=True, GLOBAL_MEMORY_READ_ONLY=False)
    yield service.app.test_client(), database, media
    service.app.config["GLOBAL_MEMORY_READ_ONLY"] = False


def _add(database, media, *args, **kwargs):
    memory = GlobalMemory(database, media_root=media)
    try:
        return _suggestion(memory, *args, **kwargs)
    finally:
        memory.close()


def test_get_queue_success_empty_and_pagination_validation(review_api):
    client, database, media = review_api
    first = _add(database, media)
    response = client.get("/api/identity-reviews?limit=1&offset=0")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["pending_count"] == 1
    assert payload["reviews"][0]["suggestion_id"] == str(first)
    assert payload["reviews"][0]["status"] == "pending"
    assert '"embedding":' not in json.dumps(payload)
    assert client.get("/api/identity-reviews?limit=0").status_code == 400
    assert client.get("/api/identity-reviews?offset=-1").status_code == 400
    assert client.get("/api/identity-reviews?limit=nope").status_code == 400
    assert client.get(
        "/api/identity-reviews?limit=1&offset=9223372036854775807"
    ).status_code == 200

    memory = GlobalMemory(database, media_root=media)
    try:
        memory.resolve_identity_review(first, "reject")
    finally:
        memory.close()
    empty = client.get("/api/identity-reviews").get_json()
    assert empty["pending_count"] == 0
    assert empty["reviews"] == []


@pytest.mark.parametrize(
    "query",
    [
        "limit=1&limit=2",
        "offset=0&offset=1",
        "unknown=value",
        "unknown=a&unknown=b",
        "limit=",
        "offset=",
        "limit=true",
        "offset=false",
        "limit=1.5",
        "offset=1e3",
        "limit=-1",
        "offset=-1",
        "offset=9223372036854775808",
    ],
)
def test_get_queue_rejects_non_contract_query_values(review_api, query):
    client, _, _ = review_api
    response = client.get(f"/api/identity-reviews?{query}")
    assert response.status_code == 400
    assert response.is_json
    assert set(response.get_json()) == {"error"}
    body = response.get_data(as_text=True).lower()
    assert "traceback" not in body
    assert "overflow" not in body


def test_actual_identity_review_component_probe():
    repository = Path(__file__).resolve().parents[1]
    frontend = repository / "forensics/person_creation/frontend"
    node = shutil.which("node")
    if node is None:
        windows_node = Path("/mnt/c/Program Files/nodejs/node.exe")
        node = str(windows_node) if windows_node.is_file() else None
    assert node is not None, "the existing frontend Node runtime is unavailable"
    result = subprocess.run(
        [node, "scripts/validate-identity-review-component.mjs"],
        cwd=frontend,
        check=False,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stderr
    assert "actual-component probe passed" in result.stdout


def test_get_detail_success_missing_and_invalid(review_api):
    client, database, media = review_api
    review_id = _add(database, media)
    response = client.get(f"/api/identity-reviews/{review_id}")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["source_profile"]["person_id"] == "person_001"
    assert payload["candidate_profile"]["person_id"] == "person_002"
    assert '"embedding":' not in json.dumps(payload)
    assert client.get("/api/identity-reviews/999").status_code == 404
    assert client.get("/api/identity-reviews/not-an-id").status_code == 400


@pytest.mark.parametrize(
    "suggestion_id",
    [
        "9" * 4301,
        "9" * 6000,
        "१२३",
        "١٢٣",
        "１２３",
        "𝟙𝟚𝟛",
        "+1",
        "-1",
        "1.0",
        "1e3",
        "0x1",
        " ",
        str(2**63),
    ],
)
def test_suggestion_id_path_contract_is_controlled_atomic_and_unlocks(
    review_api,
    suggestion_id,
):
    client, database, media = review_api
    _add(database, media)
    encoded = quote(suggestion_id, safe="")

    memory = GlobalMemory(database, media_root=media)
    try:
        before = {
            table: [tuple(row) for row in memory._conn.execute(f"SELECT * FROM {table}")]
            for table in (
                "persons",
                "appearances",
                "person_gallery",
                "recognition_log",
                "identity_match_suggestions",
                "identity_merge_audit",
            )
        }
    finally:
        memory.close()

    responses = (
        client.get(f"/api/identity-reviews/{encoded}"),
        client.post(f"/api/identity-reviews/{encoded}/accept"),
        client.post(f"/api/identity-reviews/{encoded}/reject"),
    )
    assert [response.status_code for response in responses] == [400, 400, 400]
    for response in responses:
        assert response.is_json
        assert set(response.get_json()) == {"error"}
        assert "traceback" not in response.get_data(as_text=True).lower()

    memory = GlobalMemory(database, media_root=media)
    try:
        after = {
            table: [tuple(row) for row in memory._conn.execute(f"SELECT * FROM {table}")]
            for table in before
        }
        assert after == before
        memory._conn.execute("BEGIN IMMEDIATE")
        memory._conn.execute("ROLLBACK")
    finally:
        memory.close()


@pytest.mark.parametrize("endpoint", ["detail", "accept", "reject"])
@pytest.mark.parametrize("suggestion_id, expected", [("1", 200), (str(2**63 - 1), 404)])
def test_suggestion_id_ascii_bounds_are_accepted_by_all_routes(
    review_api,
    endpoint,
    suggestion_id,
    expected,
):
    client, database, media = review_api
    assert _add(database, media) == 1
    path = f"/api/identity-reviews/{suggestion_id}"
    response = (
        client.get(path)
        if endpoint == "detail"
        else client.post(f"{path}/{endpoint}")
    )
    assert response.status_code == expected


def test_post_accept_success_replay_and_server_owned_direction(review_api):
    client, database, media = review_api
    review_id = _add(database, media)
    override = client.post(
        f"/api/identity-reviews/{review_id}/accept",
        json={"source_person_id": "person_002", "target_person_id": "person_001"},
    )
    assert override.status_code == 400

    first = client.post(
        f"/api/identity-reviews/{review_id}/accept",
        json={"reason": "same person", "decision_source": "supervisor-a"},
    )
    replay = client.post(f"/api/identity-reviews/{review_id}/accept", json={})
    assert first.status_code == replay.status_code == 200
    assert first.get_json()["source_person_id"] == "person_001"
    assert first.get_json()["target_person_id"] == "person_002"
    assert first.get_json()["idempotent_replay"] is False
    assert replay.get_json()["idempotent_replay"] is True

    memory = GlobalMemory(database, media_root=media)
    try:
        source = memory._conn.execute(
            "SELECT is_active, merged_into_person_id FROM persons WHERE person_id='person_001'"
        ).fetchone()
        assert tuple(source) == (0, "person_002")
        assert memory._conn.execute("SELECT COUNT(*) FROM identity_merge_audit").fetchone()[0] == 1
    finally:
        memory.close()


def test_post_reject_success_replay_and_no_person_mutation(review_api):
    client, database, media = review_api
    review_id = _add(database, media)
    memory = GlobalMemory(database, media_root=media)
    people_before = [tuple(row) for row in memory._conn.execute("SELECT * FROM persons ORDER BY person_id")]
    memory.close()

    first = client.post(
        f"/api/identity-reviews/{review_id}/reject",
        json={"decision_source": "supervisor-b"},
    )
    replay = client.post(f"/api/identity-reviews/{review_id}/reject")
    assert first.status_code == replay.status_code == 200
    assert first.get_json()["status"] == "rejected"
    assert replay.get_json()["idempotent_replay"] is True
    assert client.post(
        f"/api/identity-reviews/{review_id}/reject",
        json={"reason": "schema has no notes"},
    ).status_code == 400
    memory = GlobalMemory(database, media_root=media)
    try:
        assert [tuple(row) for row in memory._conn.execute("SELECT * FROM persons ORDER BY person_id")] == people_before
        assert memory._conn.execute("SELECT COUNT(*) FROM identity_merge_audit").fetchone()[0] == 0
    finally:
        memory.close()


def test_post_conflicts_stale_and_missing_ids(review_api):
    client, database, media = review_api
    accepted = _add(database, media)
    rejected = _add(database, media, "person_003", "person_004")
    stale = _add(database, media, "person_004", "person_002", status="stale")
    assert client.post(f"/api/identity-reviews/{accepted}/accept").status_code == 200
    assert client.post(f"/api/identity-reviews/{accepted}/reject").status_code == 409
    assert client.post(f"/api/identity-reviews/{rejected}/reject").status_code == 200
    assert client.post(f"/api/identity-reviews/{rejected}/accept").status_code == 409
    assert client.post(f"/api/identity-reviews/{stale}/accept").status_code == 409
    assert client.post("/api/identity-reviews/999/accept").status_code == 404
    assert client.post("/api/identity-reviews/nope/reject").status_code == 400


def test_post_malformed_json_and_non_object_are_400(review_api):
    client, database, media = review_api
    review_id = _add(database, media)
    malformed = client.post(
        f"/api/identity-reviews/{review_id}/accept",
        data="{",
        content_type="application/json",
    )
    non_object = client.post(
        f"/api/identity-reviews/{review_id}/reject",
        json=["reject"],
    )
    plain = client.post(
        f"/api/identity-reviews/{review_id}/accept",
        data="reason",
        content_type="text/plain",
    )
    assert malformed.status_code == non_object.status_code == plain.status_code == 400


def test_post_read_only_mode_maps_to_503(review_api):
    client, database, media = review_api
    review_id = _add(database, media)
    service.app.config["GLOBAL_MEMORY_READ_ONLY"] = True
    assert client.get("/api/identity-reviews").status_code == 200
    response = client.post(f"/api/identity-reviews/{review_id}/accept")
    assert response.status_code == 503
    assert response.get_json() == {"error": "identity reviews are read-only"}


def test_controlled_internal_failure_is_generic_and_atomic(review_api, monkeypatch):
    client, database, media = review_api
    review_id = _add(database, media)

    def fail(_self):
        raise RuntimeError("secret traceback detail")

    monkeypatch.setattr(GlobalMemory, "_before_review_commit", fail)
    response = client.post(f"/api/identity-reviews/{review_id}/accept")
    assert response.status_code == 500
    assert response.get_json() == {"error": "identity review operation failed"}
    assert "traceback" not in response.get_data(as_text=True).lower()
    assert "secret" not in response.get_data(as_text=True).lower()
    memory = GlobalMemory(database, media_root=media)
    try:
        assert memory._conn.execute(
            "SELECT status FROM identity_match_suggestions WHERE id=?", (review_id,)
        ).fetchone()[0] == "pending"
        assert memory._conn.execute("SELECT COUNT(*) FROM identity_merge_audit").fetchone()[0] == 0
        assert tuple(memory._conn.execute(
            "SELECT is_active, merged_into_person_id FROM persons WHERE person_id='person_001'"
        ).fetchone()) == (1, None)
    finally:
        memory.close()
