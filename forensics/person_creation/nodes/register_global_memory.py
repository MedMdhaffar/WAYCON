import os
from pathlib import Path

from forensics.global_memory.store import GlobalMemoryStore


def _profile_paths_from_state(state: dict) -> list[Path]:
    paths = [Path(p) for p in state.get("profile_paths", []) if p]
    if paths:
        return paths

    profile_path = state.get("profile_path")
    if profile_path:
        return [Path(profile_path)]

    output_dir = state.get("output_dir")
    if not output_dir:
        return []

    root = Path(output_dir)
    cluster_profiles = sorted(root.glob("cluster_*/profile.json"))
    if cluster_profiles:
        return cluster_profiles
    return [root / "profile.json"]


def register_global_memory(state: dict) -> dict:
    """
    Register finalized profile JSON files into Global Memory.
    """
    strict = os.getenv("WAYCON_GLOBAL_MEMORY_STRICT") == "1"
    store = GlobalMemoryStore()
    registered_ids: list[str] = []
    registered_count = 0

    for profile_path in _profile_paths_from_state(state):
        if not profile_path.exists():
            message = f"[global_memory] profile file missing: {profile_path}"
            if strict:
                raise FileNotFoundError(message)
            print(f"{message} - skipping")
            continue

        result = store.register_profile_file(profile_path)
        registered_ids.extend(result["person_ids"])
        registered_count += int(result["registered_count"])

    print(f"[global_memory] db -> {store.db_path}")
    print(f"[global_memory] registered count -> {registered_count}")
    print(f"[global_memory] registered person ids -> {registered_ids}")

    return {
        "global_memory_db_path": str(store.db_path),
        "global_memory_registered_person_ids": registered_ids,
        "global_memory_registered_count": registered_count,
    }
