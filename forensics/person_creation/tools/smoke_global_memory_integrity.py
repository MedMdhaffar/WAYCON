from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _fake_profile(name: str, embedding: list[float], appearance_date: str = "2026-07-08") -> dict:
    return {
        "id": name,
        "name": name,
        "cluster_id": 0,
        "cluster_confidence": 0.9,
        "cluster_face_count": 1,
        "face_embedding": embedding,
        "appearance": {"date": appearance_date, "top": "unknown"},
        "appearance_signals": {"color": {}},
        "reid": {"primary_key": "face_embedding"},
        "face_crops": ["face.jpg"],
        "body_crops": ["body.jpg"],
        "best_body_crops": ["body.jpg"],
        "video_sources": ["video.mp4"],
    }


def main() -> int:
    from forensics.person_creation.global_memory import GlobalMemoryStore
    import forensics.person_creation.global_memory.store as store_mod

    tmp = Path(tempfile.mkdtemp(prefix="waycon_gm_smoke_"))
    try:
      # DB path is repo-anchored when env override is absent.
        old_env = os.environ.pop("PERSON_GLOBAL_MEMORY_DB", None)
        a = GlobalMemoryStore()
        db_a = a.db_path
        a.close()
        old_cwd = Path.cwd()
        os.chdir(tmp)
        b = GlobalMemoryStore()
        db_b = b.db_path
        b.close()
        os.chdir(old_cwd)
        if old_env is not None:
            os.environ["PERSON_GLOBAL_MEMORY_DB"] = old_env
        _assert(db_a == db_b and db_a.is_absolute(), "default DB path is not stable across cwd")

        # Rejected suggestions do not block future pending suggestions.
        db = tmp / "memory.sqlite"
        store = GlobalMemoryStore(str(db))
        store_mod.embed_face_photo_records = lambda paths, save_crops_dir=None: {
            "embeddings": [[1.0] + [0.0] * 511],
            "sources": [{"image_path": paths[0], "face_crop_path": None}],
            "skipped": [],
        }
        base = store.register_face_photo_identity_with_result("Base", ["base.jpg"])
        maybe = store.register_profile_with_result(
            _fake_profile("MaybeBaseA", [0.5, 0.8660254] + [0.0] * 510),
            profile_path="maybe_a.json",
        )
        first_id = maybe["suggestion_id"]
        _assert(first_id, "first uncertain match did not create suggestion")
        store.reject_suggestion(first_id)
        second_id = store._create_match_suggestion(
            maybe["person_id"],
            base["person_id"],
            0.5,
            0.65,
            "2026-07-08T00:00:00+00:00",
        )
        _assert(second_id, "rejected suggestion blocked a future pending suggestion")
        _assert(second_id != first_id, "new suggestion reused rejected id")
        store.close()

        # Flask health and output_dir guard. Skipped only when the local smoke
        # environment has not installed Flask yet.
        try:
            from forensics.person_creation.service import app
        except ModuleNotFoundError as exc:
            if exc.name != "flask":
                raise
            print("smoke_global_memory_integrity: skipped Flask API checks (flask not installed)")
        else:
            client = app.test_client()
            health = client.get("/api/health")
            _assert(health.status_code == 200 and "device" in health.get_json(), "/api/health missing device info")

            used = tmp / "used_run"
            (used / "cluster_0").mkdir(parents=True)
            res = client.post("/api/person/start", json={
                "name": "Smoke",
                "video_paths": ["dummy.mp4"],
                "output_dir": str(used),
            })
            _assert(res.status_code == 400, "used output_dir was not rejected")

        from forensics.person_creation.nodes.finalize import finalize

        os.environ["PERSON_GLOBAL_MEMORY_DB"] = str(tmp / "finalize_memory.sqlite")
        out = tmp / "finalize_run"
        result = finalize({
            "output_dir": str(out),
            "video_paths": ["video.mp4"],
            "per_cluster_profiles": {0: _fake_profile("FinalPerson", [1.0] + [0.0] * 511)},
            "identity_clusters": [{"cluster_id": 0}],
            "unresolved_faces": [],
            "unattached_bodies": [],
        })
        _assert("global_memory" in result, "finalize did not return global_memory")
        report = json.loads((out / "session_report.json").read_text(encoding="utf-8"))
        _assert("global_memory" in report and "device" in report, "session report missing global_memory/device")

        print("smoke_global_memory_integrity: ok")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
