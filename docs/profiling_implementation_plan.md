# Person Creation Profiling Implementation Plan

## Inspected execution path

`forensics/person_creation/graph.py` is authoritative. Its linear node order is:

1. `load_models`
2. `process_video`
3. `filter_quality`
4. `embed_all_faces`
5. `cluster_identities`
6. `assign_bodies_to_clusters`
7. `promote_crops`
8. `select_best`
9. `compute_reid`
10. `describe_clothing`
11. `build_profile`
12. `finalize`

The Flask service and CLI import `build_graph`; direct node imports are confined to
`graph.py`. Therefore node timing will be added at graph registration, preserving
all existing public node functions and avoiding recursive wrappers.

`process_video` opens videos with `cv2.VideoCapture`, reads frames with
`cap.read`, applies the stride decision, calls `PersonDetector.detect` and
`FaceEngineClient.detect`, extracts crops with `_crop`, calculates sharpness with
`_sharpness`, writes JPEGs with `cv2.imwrite`, and constructs crop dictionaries.
The person detector wraps one Ultralytics `predict` call; the face detector is a
remote HTTP `/detect` request whose client performs JPEG encoding.

`GlobalMemory` owns one SQLite connection per instance, uses explicit
`BEGIN IMMEDIATE`/`COMMIT`/`ROLLBACK` during registration, and otherwise relies
on autocommit. Important work includes schema initialization, face-search fetch,
BLOB-to-NumPy conversion, cosine similarity and sorting, person insert/update,
appearance upsert, gallery insert/prune, recognition-log insert, reads, and close.

## Minimum-safe implementation

1. Replace the basic profiler with an optional, thread-safe high-resolution
   profiler, active-profiler context binding, statistical summaries, atomic JSON
   writes, run metadata, text summary support, and a daemon resource sampler.
2. Enable profiling only with `PERSON_CREATION_PROFILE=1`; use separate flags for
   CUDA synchronization, sample interval, verbose records, frame-level records,
   and SQL trace/query-plan collection.
3. Wrap the 12 graph callables only when registered. Derive small result metadata
   from node updates without storing images, arrays, embeddings, or full state.
4. Instrument `process_video` around video open/metadata, each decode, stride
   decision, detector calls, crop extraction, sharpness, JPEG writing, metadata,
   per-video total, and aggregate counters. Keep CUDA synchronization opt-in and
   limit per-frame record volume behind the frame-level flag.
5. Instrument GlobalMemory at method/transaction boundaries and split face search
   into fetch, deserialize, cosine, sort, and total measurements without changing
   query results. Add optional SQLite trace and query plans only in profiling mode.
6. Use one profiling session and one sampler per Flask job or CLI invocation.
   Always stop sampling and atomically write reports in `output_dir/profiling`,
   including failure runs. Append a locked process-memory history JSONL entry.
7. Add focused unit tests for success/error records, CPU deltas, percentiles,
   disabled/no-CUDA behavior, sampler lifecycle, database records, and report
   serialization. Run compile checks and existing relevant tests with the
   available project interpreter; run a real video only if the configured models,
   face-engine service, and test video are locally available.

No model call will be duplicated. No thresholds, graph order, algorithms, database
schema/data semantics, crop selection, or output profile contents will be changed.
