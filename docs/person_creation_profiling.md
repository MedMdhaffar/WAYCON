# Person Creation Profiling

Profiling is disabled by default. Enable it before starting the Flask service or
CLI pipeline:

```powershell
$env:PERSON_CREATION_PROFILE = "1"
$env:PERSON_CREATION_PROFILE_RESOURCE_INTERVAL = "1.0"
```

Optional controls:

- `PERSON_CREATION_PROFILE_CUDA_SYNC=1` enables CUDA synchronization and CUDA
  event timing around marked model calls. It improves GPU timing accuracy but
  adds overhead.
- `PERSON_CREATION_PROFILE_VERBOSE=1` prints every completed timing record.
- `PERSON_CREATION_PROFILE_FRAME_LEVEL=1` adds frame indexes to repeated frame
  records. Core aggregate frame timings are collected whenever profiling is on.
- `PERSON_CREATION_PROFILE_SQL=1` enables redacted SQLite trace output and query
  plan inspection.

Reports are written atomically after success or failure to:

```text
<output_dir>/profiling/
    pipeline_profile.json
    resource_samples.json
    database_profile.json
    profiling_summary.txt
    process_history.jsonl
    sql_trace.log              # only with SQL profiling enabled
```

`psutil` supplies process CPU/RAM sampling. When it is unavailable, timing still
works and process-resource fields are omitted. PyTorch supplies allocated,
reserved, and peak CUDA memory, but it does not expose whole-device utilization,
temperature, or power. Those device metrics are included only when a compatible
`pynvml` package is installed and NVML is available.

The face engine is a separate HTTP service. Client reports can separate JPEG
encoding, request latency, and response postprocessing, but cannot separate the
remote service's preprocessing, inference, and NMS. Ultralytics exposes
preprocess/inference/postprocess internally; the current safe instrumentation
therefore measures its complete `predict` call plus local result conversion.
