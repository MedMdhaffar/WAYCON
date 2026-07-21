import os

# ─── Postgres connection ───────────────────────────────────────────────────────
#
# GLOBAL_MEMORY_DSN, if set, overrides everything else below (a full
# postgresql://user:pass@host:port/dbname URI or libpq keyword string). Otherwise
# the individual GLOBAL_MEMORY_* parts are assembled into a DSN. See
# .env.example / docker-compose.yml for the matching local-dev defaults.
DSN: str | None = os.environ.get("GLOBAL_MEMORY_DSN")

HOST: str = os.environ.get("GLOBAL_MEMORY_HOST", "localhost")
PORT: int = int(os.environ.get("GLOBAL_MEMORY_PORT", "5432"))
DBNAME: str = os.environ.get("GLOBAL_MEMORY_DB", "waycon_forensics")
USER: str = os.environ.get("GLOBAL_MEMORY_USER", "waycon")
PASSWORD: str = os.environ.get("GLOBAL_MEMORY_PASSWORD", "waycon")

POOL_MIN_SIZE: int = int(os.environ.get("GLOBAL_MEMORY_POOL_MIN_SIZE", "1"))
POOL_MAX_SIZE: int = int(os.environ.get("GLOBAL_MEMORY_POOL_MAX_SIZE", "10"))


def connection_string() -> str:
    if DSN:
        return DSN
    return (
        f"host={HOST} port={PORT} dbname={DBNAME} "
        f"user={USER} password={PASSWORD}"
    )


# Legacy SQLite path -- no longer used by GlobalMemory (see store.py), only read by
# migrate_sqlite_to_postgres.py as the one-time backfill source, and by any
# still-installed copy of the old SQLite-backed tooling.
DB_PATH: str = os.environ.get(
    "FORENSICS_MEMORY_DB",
    "forensics/global_memory.db",
)

SIMILARITY_THRESHOLD: float = float(os.environ.get(
    "FACE_SIMILARITY_THRESHOLD",
    "0.60",
))

# Generic pipeline-version tag: PersonCreationState.pipeline_version (state.py),
# written onto segments.pipeline_version as part of its (segment_start_ts,
# pipeline_version) idempotency key. Bump when the detection/identity pipeline
# changes shape in a way that should be distinguishable during backfills/debugging.
PIPELINE_VERSION: str = os.environ.get("PIPELINE_VERSION", "pipeline_v1")

# Tag written onto clothing_jobs rows (and into appearances) so a change to the
# async VLM worker / prompt / model can be told apart from older jobs during
# backfills or debugging. Bump when the clothing-description pipeline changes shape.
CLOTHING_PIPELINE_VERSION: str = os.environ.get(
    "CLOTHING_PIPELINE_VERSION",
    "clothing_v1",
)

# clothing_jobs.status retry policy for the async VLM worker (vlm_worker.py). Mirrors
# the shape of the segment reliability state machine and gst_stream.py's reconnect
# backoff: PENDING -> PROCESSING -> DONE / FAILED_RETRYABLE (retried up to
# CLOTHING_JOB_MAX_ATTEMPTS, with exponential backoff between attempts, capped) /
# FAILED_FINAL once attempts are exhausted.
CLOTHING_JOB_MAX_ATTEMPTS: int = int(os.environ.get("CLOTHING_JOB_MAX_ATTEMPTS", "3"))
CLOTHING_JOB_RETRY_BACKOFF_SECONDS: tuple[float, ...] = (10.0, 60.0, 300.0)

# segments.status values (see schema_postgres.sql). SUCCEEDED reflects core
# detection/identity completion only -- clothing enrichment is tracked separately in
# clothing_jobs / appearances and must never be conflated with this table (see the
# /api/segments* monitoring endpoints in person_creation/service.py).
SEGMENT_STATUS_CAPTURING = "CAPTURING"
SEGMENT_STATUS_READY = "READY"
SEGMENT_STATUS_PROCESSING = "PROCESSING"
SEGMENT_STATUS_SUCCEEDED = "SUCCEEDED"
SEGMENT_STATUS_FAILED_RETRYABLE = "FAILED_RETRYABLE"
SEGMENT_STATUS_FAILED_FINAL = "FAILED_FINAL"
SEGMENT_MAX_RETRIES: int = int(os.environ.get("SEGMENT_MAX_RETRIES", "3"))

CLOTHING_JOB_STATUS_PENDING = "pending"
CLOTHING_JOB_STATUS_PROCESSING = "processing"
CLOTHING_JOB_STATUS_DONE = "done"
CLOTHING_JOB_STATUS_FAILED_RETRYABLE = "failed_retryable"
CLOTHING_JOB_STATUS_FAILED_FINAL = "failed_final"
