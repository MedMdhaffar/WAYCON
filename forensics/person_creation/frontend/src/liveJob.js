const TERMINAL_STATUSES = new Set(['done', 'error'])

const STATUS_LABELS = {
  idle: 'Queued',
  queued: 'Queued',
  loading_models: 'Loading models',
  connecting_camera: 'Connecting camera',
  buffering_stream: 'Buffering camera',
  processing_live_frames: 'Camera running',
  stop_requested: 'Stop requested',
  stopping: 'Stopping camera',
  processing_video: 'Extracting video crops',
  filtering: 'Processing captured evidence',
  embedding: 'Processing captured evidence',
  clustering: 'Processing captured evidence',
  auto_pairing: 'Processing captured evidence',
  promoting_crops: 'Processing captured evidence',
  selecting: 'Processing captured evidence',
  computing_reid: 'Processing captured evidence',
  describing: 'Processing captured evidence',
  building_profile: 'Processing captured evidence',
  finalizing: 'Finalizing captured evidence',
  done: 'Complete',
  error: 'Failed',
}

const DOWNSTREAM_NODES = new Set([
  'filter_quality', 'embed_all_faces', 'cluster_identities',
  'assign_bodies_to_clusters', 'promote_crops', 'select_best',
  'compute_reid', 'describe_clothing', 'build_profile', 'finalize',
])

function objectOrEmpty(value) {
  return value && typeof value === 'object' && !Array.isArray(value) ? value : {}
}

function finiteNumber(value) {
  if (value === null || value === undefined || value === '') return null
  const number = Number(value)
  return Number.isFinite(number) ? number : null
}

export function isTerminalStatus(status) {
  return TERMINAL_STATUSES.has(status)
}

export function isJobActive(jobId, status) {
  return Boolean(jobId) && !isTerminalStatus(status)
}

export function shouldShowStop(jobId, sourceType, status, node, snapshot) {
  if (sourceType !== 'live_camera' || !isJobActive(jobId, status)) return false
  if (!DOWNSTREAM_NODES.has(node)) return true
  return isStopPending(status, snapshot)
}

export function isStopPending(status, snapshot, requestPending = false) {
  return Boolean(requestPending || status === 'stop_requested' || status === 'stopping' || snapshot?.stop_requested)
}

export function statusLabel(status) {
  if (!status) return 'Waiting for status'
  return STATUS_LABELS[status] ?? status.replace(/_/g, ' ')
}

export function cameraPhase(status, node, snapshot) {
  if (status === 'done') return 'Complete'
  if (status === 'error') return 'Failed'
  if (DOWNSTREAM_NODES.has(node)) return 'Processing captured evidence'
  if (status === 'stopping') return 'Stopping camera'
  if (status === 'stop_requested' || snapshot?.stop_requested) return 'Stop requested'
  return 'Running'
}

export function normalizeSnapshot(payload) {
  const nested = objectOrEmpty(payload?.snapshot)
  const topLevelProgress = {}
  for (const key of [
    'chunk_index', 'completed_chunks', 'continuous',
    'duration_seconds_per_chunk', 'stop_requested', 'last_chunk',
    'session_totals', 'stream_stats',
  ]) {
    if (payload?.[key] !== undefined) topLevelProgress[key] = payload[key]
  }
  return { ...topLevelProgress, ...nested }
}

export function liveProgress(snapshot = {}) {
  const stats = objectOrEmpty(snapshot.stream_stats)
  const lastChunk = objectOrEmpty(snapshot.last_chunk ?? stats.last_chunk)
  const totals = objectOrEmpty(snapshot.session_totals ?? stats.session_totals)
  const warnings = [
    ...(Array.isArray(stats.warnings) ? stats.warnings : []),
    ...(Array.isArray(lastChunk.warnings) ? lastChunk.warnings : []),
  ].filter(value => typeof value === 'string' && value.trim())

  return {
    completedWindows: finiteNumber(snapshot.completed_chunks ?? stats.completed_chunks),
    windowIndex: finiteNumber(snapshot.chunk_index ?? stats.chunk_index ?? lastChunk.chunk_index),
    windowDuration: finiteNumber(snapshot.duration_seconds_per_chunk ?? stats.duration_seconds_per_chunk ?? stats.duration_seconds ?? snapshot.duration_seconds),
    windowFramesRead: finiteNumber(lastChunk.frames_read),
    windowFramesProcessed: finiteNumber(lastChunk.frames_processed),
    windowBodyDetections: finiteNumber(lastChunk.body_detections),
    windowFaceDetections: finiteNumber(lastChunk.face_detections),
    totalFramesRead: finiteNumber(totals.frames_read ?? stats.frames_read),
    totalFramesProcessed: finiteNumber(totals.frames_processed ?? stats.frames_processed),
    totalBodyDetections: finiteNumber(totals.body_detections ?? stats.body_detections),
    totalFaceDetections: finiteNumber(totals.face_detections ?? stats.face_detections),
    warnings: [...new Set(warnings)],
  }
}

export function displayMetric(value, suffix = '') {
  return value === null || value === undefined ? '-' : `${value}${suffix}`
}

export function safeErrorMessage(error, fallback = 'Request failed') {
  const raw = typeof error === 'string' ? error : error?.message
  if (!raw) return fallback
  return String(raw).replace(/rtsps?:\/\/[^\s"'<>]+/gi, '<camera-source>')
}

export async function postStopRequest(jobId, fetchImpl = fetch) {
  const response = await fetchImpl(`/api/person/stop/${encodeURIComponent(jobId)}`, { method: 'POST' })
  let data = {}
  try {
    data = await response.json()
  } catch (_) {
    data = {}
  }
  if (data.status === 'already_finished') return data
  if (!response.ok || data.error || data.status === 'not_live_camera' || data.status === 'stop_unavailable') {
    const defaults = {
      not_live_camera: 'Stop is available only for live camera jobs.',
      stop_unavailable: 'The camera job can no longer be stopped.',
    }
    throw new Error(safeErrorMessage(data.error || defaults[data.status], `Stop request failed (HTTP ${response.status})`))
  }
  return data
}

export function runSingleFlight(ref, action) {
  if (ref.current) return ref.current
  const request = Promise.resolve().then(action)
  const guarded = request.finally(() => {
    if (ref.current === guarded) ref.current = null
  })
  ref.current = guarded
  return guarded
}

export function startStatusPolling({
  fetchStatus,
  onTerminal,
  intervalMs = 2000,
  setIntervalFn = setInterval,
  clearIntervalFn = clearInterval,
}) {
  let disposed = false
  let inFlight = false
  let timerId = null

  const stop = () => {
    if (disposed) return
    disposed = true
    if (timerId !== null) clearIntervalFn(timerId)
  }
  const poll = async () => {
    if (disposed || inFlight) return
    inFlight = true
    try {
      const data = await fetchStatus()
      if (data && isTerminalStatus(data.status)) {
        onTerminal?.(data)
        stop()
      }
    } finally {
      inFlight = false
    }
  }

  timerId = setIntervalFn(poll, intervalMs)
  void poll()
  return stop
}
