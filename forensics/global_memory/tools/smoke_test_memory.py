from __future__ import annotations

import argparse
import json
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path

from forensics.global_memory.store import GlobalMemoryStore


def _count(conn: sqlite3.Connection, table: str) -> int:
    return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test WAYCON Global Memory.")
    parser.add_argument("--profile", required=True, help="Path to a profile.json file.")
    args = parser.parse_args()

    profile_path = Path(args.profile)
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    first_profile = profile["people"][0] if isinstance(profile.get("people"), list) else profile
    person_id = first_profile["id"]
    appearance_date = first_profile.get("appearance", {}).get("date")
    embedding = first_profile["face_embedding"]
    expected_camera = Path(str(first_profile["video_sources"][0]).replace("\\", "/")).stem

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "waycon_memory_test.db"
        store = GlobalMemoryStore(db_path)
        result = store.register_profile_file(profile_path)
        assert person_id in result["person_ids"]

        with closing(sqlite3.connect(db_path)) as conn:
            assert _count(conn, "people") >= 1
            assert _count(conn, "appearances") >= 1
            assert _count(conn, "face_embeddings") >= 1
            assert _count(conn, "crop_references") >= 1

        face_hits = store.search_by_face(embedding, top_k=1)
        assert face_hits and face_hits[0]["person_id"] == person_id

        date_hits = store.get_by_date(appearance_date)
        assert any(row["person_id"] == person_id for row in date_hits)

        camera_hits = store.get_by_camera(expected_camera)
        assert any(row["person_id"] == person_id for row in camera_hits)

        print(json.dumps({
            "status": "ok",
            "db_path": str(db_path),
            "registered": result,
            "top_face_hit": face_hits[0],
            "date_hits": len(date_hits),
            "camera_hits": len(camera_hits),
        }, indent=2))


if __name__ == "__main__":
    main()
