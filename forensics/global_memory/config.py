import os

DB_PATH: str = os.environ.get(
    "FORENSICS_MEMORY_DB",
    "forensics/global_memory.db",
)

SIMILARITY_THRESHOLD: float = float(os.environ.get(
    "FACE_SIMILARITY_THRESHOLD",
    "0.60",
))

# Tag written onto clothing_jobs rows (and eventually into appearances) so a change
# to the async VLM worker / prompt / model can be told apart from older jobs during
# backfills or debugging. Bump when the clothing-description pipeline changes shape.
CLOTHING_PIPELINE_VERSION: str = os.environ.get(
    "CLOTHING_PIPELINE_VERSION",
    "clothing_v1",
)
