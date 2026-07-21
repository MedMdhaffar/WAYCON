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

const ROLLING_STATE_LABELS = {
  disabled: 'Disabled',
  waiting: 'Waiting',
  scheduled: 'Waiting',
  analyzing: 'Analyzing',
  ready: 'Ready',
  warning: 'Warning',
  error: 'Error',
}

/**
 * @typedef {Object} RollingMemoryMatch
 * @property {string} personId
 * @property {string} name
 * @property {number|null} similarity
 */

/**
 * @typedef {Object} RollingIdentity
 * @property {string} sessionPersonId
 * @property {number|null} clusterLabel
 * @property {string} status
 * @property {number} faceCount
 * @property {number} associatedBodyCount
 * @property {number|null} firstSeenChunk
 * @property {number|null} lastSeenChunk
 * @property {string} representativeFacePath
 * @property {RollingMemoryMatch|null} memoryMatch
 */

/**
 * @typedef {Object} RollingEvent
 * @property {string} eventId
 * @property {string} type
 * @property {string} sessionPersonId
 * @property {number|null} analysisVersion
 * @property {RollingMemoryMatch|null} memoryMatch
 */

function objectOrEmpty(value) {
  return value && typeof value === 'object' && !Array.isArray(value) ? value : {}
}

function finiteNumber(value) {
  if (value === null || value === undefined || value === '') return null
  const number = Number(value)
  return Number.isFinite(number) ? number : null
}

function nonNegativeInteger(value, fallback = 0) {
  const number = finiteNumber(value)
  return number === null ? fallback : Math.max(0, Math.trunc(number))
}

function stringValue(value, fallback = '') {
  return typeof value === 'string' && value.trim() ? value.trim() : fallback
}

function normalizeMemoryMatch(value) {
  const match = objectOrEmpty(value)
  const personId = stringValue(match.person_id)
  const name = stringValue(match.name)
  const similarity = finiteNumber(match.similarity)
  if (!personId && !name && similarity === null) return null
  return { personId, name, similarity }
}

function rollingState(value, enabled) {
  if (!enabled) return 'disabled'
  const state = stringValue(value, 'idle').toLowerCase()
  if (state === 'running' || state === 'analyzing') return 'analyzing'
  if (state === 'ready') return 'ready'
  if (state === 'scheduled') return 'scheduled'
  if (state === 'warning' || state === 'shutdown_warning') return 'warning'
  if (state === 'error' || state === 'failed' || state === 'worker_failed') return 'error'
  return 'waiting'
}

export function mediaImageUrl(path) {
  const safePath = stringValue(path)
  if (!safePath) return ''
  if (
    safePath.includes('\\')
    || safePath.startsWith('/')
    || /^[A-Za-z]:/.test(safePath)
    || /^[A-Za-z][A-Za-z0-9+.-]*:/.test(safePath)
  ) return ''
  const segments = safePath.split('/')
  if (segments.some(segment => !segment || segment === '.' || segment === '..')) return ''
  return `/api/images?path=${encodeURIComponent(safePath)}`
}

export function identityImageUrl(path) {
  return mediaImageUrl(path)
}

export function formatSimilarity(value) {
  const similarity = finiteNumber(value)
  if (similarity === null) return null
  return `${Math.round(Math.min(1, Math.max(0, similarity)) * 100)}%`
}

export function normalizeRollingAnalysis(value) {
  const rolling = objectOrEmpty(value)
  const enabled = rolling.enabled === true
  const rawIdentities = Array.isArray(rolling.live_identities) ? rolling.live_identities : []
  const identityIds = new Set()
  const identities = []
  for (const rawValue of rawIdentities) {
    const raw = objectOrEmpty(rawValue)
    const sessionPersonId = stringValue(raw.session_person_id)
    if (!sessionPersonId || identityIds.has(sessionPersonId)) continue
    identityIds.add(sessionPersonId)
    identities.push({
      sessionPersonId,
      clusterLabel: finiteNumber(raw.cluster_label),
      status: stringValue(raw.status, 'provisional'),
      faceCount: nonNegativeInteger(raw.face_count),
      associatedBodyCount: nonNegativeInteger(raw.associated_body_count),
      firstSeenChunk: finiteNumber(raw.first_seen_chunk),
      lastSeenChunk: finiteNumber(raw.last_seen_chunk),
      representativeFacePath: stringValue(raw.representative_face_path),
      memoryMatch: normalizeMemoryMatch(raw.memory_match),
    })
  }

  const rawEvents = Array.isArray(rolling.live_recognition_events)
    ? rolling.live_recognition_events
    : []
  const eventIds = new Set()
  const events = []
  for (let index = rawEvents.length - 1; index >= 0 && events.length < 20; index -= 1) {
    const raw = objectOrEmpty(rawEvents[index])
    const type = stringValue(raw.type ?? raw.event_type, 'unknown')
    const sessionPersonId = stringValue(raw.session_person_id)
    const fallbackId = `legacy-${type}-${sessionPersonId}-${finiteNumber(raw.analysis_version) ?? index}`
    const eventId = stringValue(raw.event_id, fallbackId)
    if (eventIds.has(eventId)) continue
    eventIds.add(eventId)
    events.push({
      eventId,
      type,
      sessionPersonId,
      analysisVersion: finiteNumber(raw.analysis_version),
      memoryMatch: normalizeMemoryMatch(raw.memory_match),
    })
  }

  const reportedState = rollingState(rolling.analysis_state, enabled)
  const state = rolling.analysis_in_progress === true && !['warning', 'error'].includes(reportedState)
    ? 'analyzing'
    : reportedState
  return {
    enabled,
    publicationSequence: nonNegativeInteger(rolling.publication_sequence),
    requestedVersion: nonNegativeInteger(rolling.requested_version),
    analysisVersion: nonNegativeInteger(rolling.analysis_version),
    state,
    stateLabel: ROLLING_STATE_LABELS[state],
    analysisInProgress: rolling.analysis_in_progress === true,
    analyzedEmbeddingCount: nonNegativeInteger(rolling.analyzed_embedding_count),
    lastCompletedChunk: finiteNumber(rolling.last_completed_preprocessing_chunk),
    warning: stringValue(rolling.analysis_warning),
    identities,
    events,
    faceEvidenceCount: identities.reduce((total, identity) => total + identity.faceCount, 0),
  }
}

export function mergeJobStatus(previous, incoming) {
  if (!incoming || typeof incoming !== 'object') return previous
  if (incoming.status === 'done') return incoming
  const previousRolling = objectOrEmpty(previous?.snapshot?.rolling_analysis)
  const incomingSnapshot = objectOrEmpty(incoming.snapshot)
  const previousMediaVersion = nonNegativeInteger(previous?.snapshot?.media_lifecycle_version)
  const incomingMediaVersion = nonNegativeInteger(incomingSnapshot.media_lifecycle_version)
  if (incomingMediaVersion !== previousMediaVersion) return incoming
  const incomingRolling = objectOrEmpty(incomingSnapshot.rolling_analysis)
  if (previousRolling.enabled !== true) return incoming

  const previousSequence = nonNegativeInteger(previousRolling.publication_sequence)
  const incomingSequence = nonNegativeInteger(incomingRolling.publication_sequence)
  if (incomingRolling.enabled === true && incomingSequence < previousSequence) {
    return previous
  }
  if (incomingRolling.enabled !== true) {
    return {
      ...incoming,
      snapshot: { ...incomingSnapshot, rolling_analysis: previousRolling },
    }
  }
  const state = rollingState(incomingRolling.analysis_state, true)
  if (state !== 'warning' && state !== 'error') return incoming
  return {
    ...incoming,
    snapshot: {
      ...incomingSnapshot,
      rolling_analysis: {
        ...incomingRolling,
        live_identities: Array.isArray(incomingRolling.live_identities) && incomingRolling.live_identities.length
          ? incomingRolling.live_identities
          : previousRolling.live_identities,
        live_recognition_events: Array.isArray(incomingRolling.live_recognition_events) && incomingRolling.live_recognition_events.length
          ? incomingRolling.live_recognition_events
          : previousRolling.live_recognition_events,
      },
    },
  }
}

export function createStatusRequestGuard() {
  let sequence = 0
  let controller = null
  return {
    start(jobId) {
      sequence += 1
      controller?.abort()
      controller = new AbortController()
      return { jobId, sequence, signal: controller.signal }
    },
    isCurrent(request) {
      return request.sequence === sequence && !request.signal.aborted
    },
    cancel() {
      sequence += 1
      controller?.abort()
      controller = null
    },
  }
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

export function shouldShowRollingIdentityPanel(status) {
  return status !== 'done'
}

export function isStopPending(status, snapshot, requestPending = false) {
  return Boolean(requestPending || status === 'stop_requested' || status === 'stopping' || snapshot?.stop_requested)
}

export function isCanonicalFinalizing(sourceType, status, node, snapshot, requestPending = false) {
  if (sourceType !== 'live_camera' || isTerminalStatus(status)) return false
  return isStopPending(status, snapshot, requestPending) || DOWNSTREAM_NODES.has(node)
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
  const streamState = stringValue(stats.stream_state).toLowerCase()
  const normalizedStreamState = ['connected', 'reconnecting', 'stopped', 'error'].includes(streamState)
    ? streamState
    : ''
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
    streamState: normalizedStreamState,
    streamReconnectCount: finiteNumber(stats.stream_reconnect_count),
    streamWarning: stringValue(stats.stream_warning),
    lastFrameAgeSeconds: finiteNumber(stats.last_frame_age_seconds),
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
      if (disposed) return
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
