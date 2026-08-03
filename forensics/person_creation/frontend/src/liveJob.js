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

const LIVE_IDENTITY_STATES = new Set([
  'observing', 'provisional', 'new_person', 'attach_existing', 'review_required',
])
const LIVE_IDENTITY_DECISIONS = new Set([
  'new_person', 'attach_existing', 'review_required',
])
const LIVE_IDENTITY_STATE_RANK = {
  observing: 0,
  provisional: 1,
  new_person: 2,
  attach_existing: 2,
  review_required: 2,
}
const VLM_STATES = new Set([
  'not_started', 'queued', 'processing', 'completed', 'failed', 'timed_out',
])
const VLM_STATE_RANK = {
  not_started: 0,
  queued: 1,
  processing: 2,
  completed: 3,
  failed: 3,
  timed_out: 3,
}
const VLM_ERRORS = new Set([
  'image_decode_failed', 'inference_error', 'invalid_output', 'empty_output',
  'no_valid_body_crop', 'persistence_error', 'queue_capacity', 'timeout',
])
const VLM_COUNTER_FIELDS = [
  'vlm_completed', 'vlm_failed', 'vlm_dropped', 'vlm_timed_out',
]
const VLM_IDENTITY_FIELDS = [
  'vlm_status', 'clothing_description', 'clothing_diagnostics', 'vlm_error',
  'vlm_version', 'selected_body_crop',
]

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

export function canonicalLiveCropPath(path, cropType) {
  const safePath = stringValue(path)
  const folder = cropType === 'face' ? 'face_crops' : cropType === 'body' ? 'body_crops' : ''
  if (!folder || !mediaImageUrl(safePath)) return ''
  const match = safePath.match(/^person_[0-9]+\/(face_crops|body_crops)\/([^/]+)$/)
  if (match && match[1] === folder && /\.(?:jpe?g|png|webp|bmp)$/i.test(match[2])) {
    return safePath
  }
  const parts = safePath.split('/')
  const filename = parts.at(-1) ?? ''
  const staging = (
    parts.length >= 4
    && parts.at(-3) === '_staging'
    && parts.at(-2) === folder
    && parts.slice(0, -3).every(part => /^[A-Za-z0-9_.-]+$/.test(part))
    && /\.(?:jpe?g|png|webp|bmp)$/i.test(filename)
  )
  return staging ? safePath : ''
}

export function sanitizeVlmError(value) {
  const error = stringValue(value).toLowerCase()
  if (!error) return ''
  return VLM_ERRORS.has(error) ? error : 'inference_error'
}

export function formatSimilarity(value) {
  const similarity = finiteNumber(value)
  if (similarity === null) return null
  return `${Math.round(Math.min(1, Math.max(0, similarity)) * 100)}%`
}

export function markStatusResponseReceived(
  data,
  receivedMonotonic = performance.now(),
  receivedEpochMs = Date.now(),
) {
  const identities = data?.snapshot?.rolling_analysis?.live_identities
  if (!Array.isArray(identities)) return data
  for (const identity of identities) {
    if (!identity || typeof identity !== 'object') continue
    identity.latency_metrics = {
      ...objectOrEmpty(identity.latency_metrics),
      frontend_received_monotonic: receivedMonotonic,
      frontend_received_epoch_ms: receivedEpochMs,
    }
  }
  return data
}

export function completeCardLatency(
  value,
  renderedMonotonic = performance.now(),
  renderedEpochMs = Date.now(),
) {
  const latency = objectOrEmpty(value)
  const received = finiteNumber(latency.frontendReceivedMonotonic)
  const sourceTimestamp = stringValue(latency.sourceFrameTimestamp)
  const sourceEpochMs = Date.parse(sourceTimestamp)
  const statusToCard = received === null
    ? null
    : Math.max(0, renderedMonotonic - received)
  const captureToCard = Number.isFinite(sourceEpochMs)
    ? Math.max(0, renderedEpochMs - sourceEpochMs)
    : (
      finiteNumber(latency.captureToStatusMs) !== null && statusToCard !== null
        ? finiteNumber(latency.captureToStatusMs) + statusToCard
        : null
    )
  return {
    ...latency,
    cardRenderedMonotonic: renderedMonotonic,
    statusToCardMs: statusToCard,
    captureToCardMs: captureToCard,
  }
}

export function normalizeRollingAnalysis(value) {
  const rolling = objectOrEmpty(value)
  const enabled = rolling.enabled === true
  const rawIdentities = Array.isArray(rolling.live_identities) ? rolling.live_identities : []
  const identityIds = new Set()
  const identities = []
  for (const rawValue of rawIdentities) {
    const raw = objectOrEmpty(rawValue)
    const liveIdentityId = stringValue(raw.live_identity_id ?? raw.session_person_id)
    if (!liveIdentityId || identityIds.has(liveIdentityId)) continue
    const bestFacePath = canonicalLiveCropPath(
      raw.best_face_path ?? raw.representative_face_path,
      'face',
    )
    const clusteringState = stringValue(
      raw.clustering_state,
      'resolved',
    ).toLowerCase() === 'unresolved' ? 'unresolved' : 'resolved'
    const reasonValue = stringValue(raw.reason)
    if (!bestFacePath) continue
    if (clusteringState === 'unresolved' && reasonValue !== 'strong_clear_match') continue
    identityIds.add(liveIdentityId)
    const sessionPersonId = stringValue(raw.session_person_id, liveIdentityId)
    const decisionValue = stringValue(raw.decision).toLowerCase()
    const decision = LIVE_IDENTITY_DECISIONS.has(decisionValue) ? decisionValue : ''
    const stateValue = stringValue(raw.state).toLowerCase()
    const state = LIVE_IDENTITY_STATES.has(stateValue)
      ? stateValue
      : decision || (raw.provisional === true ? 'provisional' : 'observing')
    const rawVlmValue = stringValue(
      raw.vlm_status ?? raw.vlm_state,
      'not_started',
    ).toLowerCase()
    const vlmValue = rawVlmValue === 'pending'
      ? 'queued'
      : rawVlmValue === 'running'
        ? 'processing'
        : rawVlmValue
    const vlmStatus = VLM_STATES.has(vlmValue) ? vlmValue : 'failed'
    const rawLatency = objectOrEmpty(raw.latency_metrics)
    identities.push({
      liveIdentityId,
      sessionPersonId,
      version: nonNegativeInteger(raw.version),
      clusterLabel: finiteNumber(raw.cluster_label),
      status: stringValue(raw.status, 'provisional'),
      clusteringState,
      state,
      decision,
      reason: reasonValue,
      provisional: raw.provisional === true,
      persisted: raw.persisted === true || (
        raw.provisional === false && Boolean(raw.canonical_person_id)
      ),
      canonicalPersonId: stringValue(raw.canonical_person_id),
      faceCount: nonNegativeInteger(raw.face_count),
      observationCount: nonNegativeInteger(raw.observation_count ?? raw.face_count),
      bodyCount: nonNegativeInteger(raw.body_count ?? raw.associated_body_count),
      associatedBodyCount: nonNegativeInteger(raw.body_count ?? raw.associated_body_count),
      candidatePersonId: stringValue(raw.candidate_person_id),
      candidateSimilarity: finiteNumber(raw.candidate_similarity),
      secondCandidatePersonId: stringValue(raw.second_candidate_person_id),
      secondCandidateSimilarity: finiteNumber(raw.second_candidate_similarity),
      margin: finiteNumber(raw.margin),
      evidenceVersion: nonNegativeInteger(raw.evidence_version ?? raw.decision_version),
      evidenceSignature: stringValue(
        raw.evidence_signature ?? raw.last_evidence_signature,
      ),
      firstSeenChunk: finiteNumber(raw.first_seen_chunk),
      lastSeenChunk: finiteNumber(raw.last_seen_chunk),
      bestFacePath,
      representativeFacePath: bestFacePath,
      bestBodyPath: canonicalLiveCropPath(raw.best_body_path, 'body'),
      vlmStatus,
      selectedBodyCrop: canonicalLiveCropPath(raw.selected_body_crop, 'body'),
      clothingDescription: stringValue(raw.clothing_description),
      vlmError: sanitizeVlmError(raw.vlm_error),
      vlmVersion: finiteNumber(raw.vlm_version),
      memoryMatch: normalizeMemoryMatch(raw.memory_match),
      comparisonTimestamp: stringValue(raw.comparison_timestamp),
      latencyMetrics: {
        sourceFrameTimestamp: stringValue(rawLatency.source_frame_timestamp),
        captureToFaceMs: finiteNumber(rawLatency.capture_to_face_ms),
        captureToEmbeddingMs: finiteNumber(rawLatency.capture_to_embedding_ms),
        embeddingToComparisonMs: finiteNumber(
          rawLatency.embedding_to_comparison_ms,
        ),
        comparisonToStatusMs: finiteNumber(
          rawLatency.comparison_to_status_ms,
        ),
        captureToStatusMs: finiteNumber(rawLatency.capture_to_status_ms),
        frontendReceivedMonotonic: finiteNumber(
          rawLatency.frontend_received_monotonic,
        ),
        frontendReceivedEpochMs: finiteNumber(
          rawLatency.frontend_received_epoch_ms,
        ),
      },
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
    unresolvedEmbeddingCount: nonNegativeInteger(
      rolling.unresolved_embedding_count,
    ),
    resolvedClusterCount: nonNegativeInteger(rolling.resolved_cluster_count),
    lastCompletedChunk: finiteNumber(rolling.last_completed_preprocessing_chunk),
    warning: stringValue(rolling.analysis_warning),
    vlmQueueDepth: nonNegativeInteger(rolling.vlm_queue_depth),
    vlmQueueCapacity: nonNegativeInteger(rolling.vlm_queue_capacity, 2),
    vlmActiveIdentity: stringValue(rolling.vlm_active_identity),
    vlmCompleted: nonNegativeInteger(rolling.vlm_completed),
    vlmFailed: nonNegativeInteger(rolling.vlm_failed),
    vlmDropped: nonNegativeInteger(rolling.vlm_dropped),
    vlmTimedOut: nonNegativeInteger(rolling.vlm_timed_out),
    identities,
    events,
    faceEvidenceCount: identities.reduce((total, identity) => total + identity.faceCount, 0),
  }
}

function rawLiveIdentityId(value) {
  const raw = objectOrEmpty(value)
  return stringValue(raw.live_identity_id ?? raw.session_person_id)
}

function rawStateRank(value) {
  const raw = objectOrEmpty(value)
  const decision = stringValue(raw.decision).toLowerCase()
  const state = stringValue(raw.state).toLowerCase()
  const normalized = LIVE_IDENTITY_STATES.has(state)
    ? state
    : LIVE_IDENTITY_DECISIONS.has(decision) ? decision : 'observing'
  return LIVE_IDENTITY_STATE_RANK[normalized]
}

function preserveVlm(previousIdentity, incomingIdentity) {
  const merged = { ...incomingIdentity }
  for (const field of VLM_IDENTITY_FIELDS) {
    if (Object.hasOwn(previousIdentity, field)) merged[field] = previousIdentity[field]
  }
  return merged
}

function mergeLiveIdentity(previousValue, incomingValue, staleVlm) {
  const previous = objectOrEmpty(previousValue)
  const incoming = objectOrEmpty(incomingValue)
  const previousVersion = nonNegativeInteger(previous.version)
  const incomingVersion = nonNegativeInteger(incoming.version)
  if (incomingVersion < previousVersion) return previous

  let merged = { ...previous, ...incoming }
  if (
    incomingVersion === previousVersion
    && rawStateRank(incoming) < rawStateRank(previous)
  ) {
    for (const field of [
      'state', 'decision', 'provisional', 'canonical_person_id', 'suggestion_id',
      'persisted', 'reason', 'candidate_person_id', 'candidate_similarity',
      'second_candidate_person_id', 'second_candidate_similarity', 'margin',
      'observation_count', 'evidence_version', 'evidence_signature',
    ]) {
      if (Object.hasOwn(previous, field)) merged[field] = previous[field]
    }
  }

  const previousVlmVersion = finiteNumber(previous.vlm_version)
  const incomingVlmVersion = finiteNumber(incoming.vlm_version)
  const previousCrop = stringValue(previous.selected_body_crop)
  const incomingCrop = stringValue(incoming.selected_body_crop)
  const previousVlmState = stringValue(previous.vlm_status, 'not_started').toLowerCase()
  const incomingVlmState = stringValue(incoming.vlm_status, 'not_started').toLowerCase()
  const versionRegressed = (
    previousVlmVersion !== null
    && incomingVlmVersion !== null
    && incomingVlmVersion < previousVlmVersion
  )
  const stateRegressed = (
    previousVlmVersion === incomingVlmVersion
    && previousCrop === incomingCrop
    && (VLM_STATE_RANK[incomingVlmState] ?? 0) < (VLM_STATE_RANK[previousVlmState] ?? 0)
  )
  if (staleVlm || versionRegressed || stateRegressed) {
    merged = preserveVlm(previous, merged)
  }

  const previousCompleted = previousVlmState === 'completed'
  if (
    previousCompleted
    && !stringValue(merged.clothing_description)
    && (incomingVlmVersion === null || incomingVlmVersion === previousVlmVersion)
  ) {
    merged.clothing_description = previous.clothing_description
    if (!stringValue(incoming.vlm_status)) merged.vlm_status = previous.vlm_status
    if (!stringValue(incoming.selected_body_crop)) {
      merged.selected_body_crop = previous.selected_body_crop
    }
  }
  return merged
}

function mergeRollingPayload(previousRolling, incomingRolling) {
  const countersRegressed = VLM_COUNTER_FIELDS.some(field => (
    finiteNumber(incomingRolling[field]) !== null
    && finiteNumber(previousRolling[field]) !== null
    && Number(incomingRolling[field]) < Number(previousRolling[field])
  ))
  const merged = { ...previousRolling, ...incomingRolling }
  const previousIdentities = new Map(
    (Array.isArray(previousRolling.live_identities) ? previousRolling.live_identities : [])
      .map(identity => [rawLiveIdentityId(identity), identity])
      .filter(([identityId]) => identityId),
  )
  const currentlyActiveIdentityIds = new Set(
    (Array.isArray(incomingRolling.live_identities)
      ? incomingRolling.live_identities
      : [])
      .map(rawLiveIdentityId)
      .filter(Boolean),
  )
  const retiredIdentityIds = new Set([
    ...(Array.isArray(previousRolling.retired_live_identity_ids)
      ? previousRolling.retired_live_identity_ids.map(stringValue).filter(Boolean)
      : []),
    ...(Array.isArray(incomingRolling.retired_live_identity_ids)
      ? incomingRolling.retired_live_identity_ids.map(stringValue).filter(Boolean)
      : []),
  ])
  for (const identityId of currentlyActiveIdentityIds) {
    retiredIdentityIds.delete(identityId)
  }
  merged.retired_live_identity_ids = [...retiredIdentityIds]
  if (Array.isArray(incomingRolling.live_identities)) {
    const seen = new Set()
    merged.live_identities = []
    for (const identity of incomingRolling.live_identities) {
      const identityId = rawLiveIdentityId(identity)
      if (!identityId || seen.has(identityId) || retiredIdentityIds.has(identityId)) continue
      seen.add(identityId)
      const previous = previousIdentities.get(identityId)
      merged.live_identities.push(
        previous ? mergeLiveIdentity(previous, identity, countersRegressed) : identity,
      )
    }
    for (const [identityId, identity] of previousIdentities) {
      if (seen.has(identityId) || retiredIdentityIds.has(identityId)) continue
      seen.add(identityId)
      merged.live_identities.push(identity)
    }
  }
  for (const field of VLM_COUNTER_FIELDS) {
    const previous = finiteNumber(previousRolling[field])
    const incoming = finiteNumber(incomingRolling[field])
    if (previous !== null || incoming !== null) merged[field] = Math.max(previous ?? 0, incoming ?? 0)
  }
  if (countersRegressed) {
    merged.vlm_queue_depth = previousRolling.vlm_queue_depth
    merged.vlm_queue_capacity = previousRolling.vlm_queue_capacity
    merged.vlm_active_identity = previousRolling.vlm_active_identity
  }
  return merged
}

export function mergeJobStatus(previous, incoming) {
  if (!incoming || typeof incoming !== 'object') return previous
  if (incoming.status === 'done') return incoming
  const previousRolling = objectOrEmpty(previous?.snapshot?.rolling_analysis)
  const incomingSnapshot = objectOrEmpty(incoming.snapshot)
  const previousMediaVersion = nonNegativeInteger(previous?.snapshot?.media_lifecycle_version)
  const incomingMediaVersion = nonNegativeInteger(incomingSnapshot.media_lifecycle_version)
  if (incomingMediaVersion < previousMediaVersion) return previous
  if (incomingMediaVersion > previousMediaVersion) return incoming
  const incomingRolling = objectOrEmpty(incomingSnapshot.rolling_analysis)
  if (previousRolling.enabled !== true) return incoming

  const monotonicRollingFields = [
    'publication_sequence', 'evidence_version', 'analysis_version',
    'requested_version',
  ]
  if (incomingRolling.enabled === true && monotonicRollingFields.some(field => (
    finiteNumber(incomingRolling[field]) !== null
    && finiteNumber(previousRolling[field]) !== null
    && Number(incomingRolling[field]) < Number(previousRolling[field])
  ))) {
    return previous
  }
  if (incomingRolling.enabled !== true) {
    return {
      ...incoming,
      snapshot: { ...incomingSnapshot, rolling_analysis: previousRolling },
    }
  }
  const state = rollingState(incomingRolling.analysis_state, true)
  const mergedRolling = mergeRollingPayload(previousRolling, incomingRolling)
  if (state !== 'warning' && state !== 'error') {
    return {
      ...incoming,
      snapshot: { ...incomingSnapshot, rolling_analysis: mergedRolling },
    }
  }
  return {
    ...incoming,
    snapshot: {
      ...incomingSnapshot,
      rolling_analysis: {
        ...mergedRolling,
        live_identities: Array.isArray(incomingRolling.live_identities) && incomingRolling.live_identities.length
          ? mergedRolling.live_identities
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
  const preprocessing = objectOrEmpty(
    snapshot.live_preprocessing ?? stats.live_preprocessing,
  )
  const rolling = objectOrEmpty(
    snapshot.rolling_analysis ?? stats.rolling_analysis,
  )
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
    samplingIntervalFrames: finiteNumber(
      snapshot.sampling_interval_frames
      ?? stats.sampling_interval_frames
      ?? lastChunk.sampling_interval_frames,
    ),
    acceptedFaces: finiteNumber(preprocessing.quality_face_crops),
    embeddedFaces: finiteNumber(preprocessing.embedded_faces),
    unresolvedEmbeddings: finiteNumber(rolling.unresolved_embedding_count),
    resolvedClusters: finiteNumber(rolling.resolved_cluster_count),
    maximumQueueDepth: finiteNumber(preprocessing.maximum_queue_depth),
    totalDroppedFrames: finiteNumber(totals.frames_dropped ?? stats.frames_dropped),
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
  intervalMs = 500,
  terminalIntervalMs = 2000,
  setTimeoutFn = setTimeout,
  clearTimeoutFn = clearTimeout,
}) {
  let disposed = false
  let inFlight = false
  let timerId = null
  let terminalNotified = false

  const stop = () => {
    if (disposed) return
    disposed = true
    if (timerId !== null) clearTimeoutFn(timerId)
    timerId = null
  }
  const schedule = delay => {
    if (disposed) return
    timerId = setTimeoutFn(() => {
      timerId = null
      void poll()
    }, delay)
  }
  const poll = async () => {
    if (disposed || inFlight) return
    inFlight = true
    let terminal = false
    try {
      const data = await fetchStatus()
      if (disposed) return
      terminal = Boolean(data && isTerminalStatus(data.status))
      if (terminal && !terminalNotified) {
        terminalNotified = true
        onTerminal?.(data)
      }
    } finally {
      inFlight = false
      schedule(terminal ? terminalIntervalMs : intervalMs)
    }
  }

  void poll()
  stop.diagnostics = Object.freeze({
    activeIntervalMs: intervalMs,
    terminalIntervalMs,
    singleFlight: true,
  })
  return stop
}
