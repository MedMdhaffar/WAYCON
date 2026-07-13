import os

DB_PATH: str = os.environ.get(
    "FORENSICS_MEMORY_DB",
    "forensics/global_memory.db",
)

SIMILARITY_THRESHOLD: float = float(os.environ.get(
    "FACE_SIMILARITY_THRESHOLD",
    "0.60",
))
