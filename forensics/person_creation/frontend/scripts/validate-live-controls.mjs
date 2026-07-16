import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'

import {
  cameraPhase,
  createStatusRequestGuard,
  displayMetric,
  formatSimilarity,
  identityImageUrl,
  isCanonicalFinalizing,
  isJobActive,
  isTerminalStatus,
  isStopPending,
  liveProgress,
  mergeJobStatus,
  normalizeRollingAnalysis,
  postStopRequest,
  runSingleFlight,
  safeErrorMessage,
  shouldShowRollingIdentityPanel,
  shouldShowStop,
  startStatusPolling,
  statusLabel,
} from '../src/liveJob.js'

assert.equal(shouldShowStop('job-1', 'live_camera', 'processing_live_frames'), true)
assert.equal(shouldShowStop('job-1', 'video_file', 'processing_video'), false)
assert.equal(shouldShowStop('job-1', 'live_camera', 'filtering', 'filter_quality', {}), false)
assert.equal(shouldShowStop('job-1', 'live_camera', 'stopping', 'filter_quality', { stop_requested: true }), true)
assert.equal(shouldShowRollingIdentityPanel('processing_live_frames'), true)
assert.equal(shouldShowRollingIdentityPanel('stop_requested'), true)
assert.equal(shouldShowRollingIdentityPanel('finalizing'), true)
assert.equal(shouldShowRollingIdentityPanel('error'), true)
assert.equal(shouldShowRollingIdentityPanel('done'), false)
assert.equal(isJobActive('job-1', 'stop_requested'), true)
assert.equal(isJobActive('job-1', 'done'), false)
assert.equal(isTerminalStatus('stop_requested'), false)
assert.equal(isTerminalStatus('stopping'), false)
assert.equal(isTerminalStatus('done'), true)
assert.equal(isTerminalStatus('error'), true)
assert.equal(isStopPending('stop_requested', {}, false), true)
assert.equal(statusLabel('processing_live_frames'), 'Camera running')
assert.equal(statusLabel('filtering'), 'Processing captured evidence')
assert.equal(cameraPhase('filtering', 'filter_quality', {}), 'Processing captured evidence')
assert.equal(cameraPhase('stopping', 'filter_quality', { stop_requested: true }), 'Processing captured evidence')

const progress = liveProgress({
  completed_chunks: 5,
  chunk_index: 4,
  duration_seconds_per_chunk: 10,
  last_chunk: {
    frames_read: 320,
    frames_processed: 21,
    body_detections: 2,
    face_detections: 1,
  },
  session_totals: {
    frames_read: 1580,
    frames_processed: 104,
    body_detections: 8,
    face_detections: 5,
  },
})
assert.deepEqual(
  [progress.completedWindows, progress.windowFramesProcessed, progress.totalFramesRead, progress.totalFaceDetections],
  [5, 21, 1580, 5],
)
assert.equal(displayMetric(liveProgress({}).totalFramesRead), '-')
const reconnectProgress = liveProgress({
  stream_stats: {
    stream_state: 'reconnecting',
    stream_reconnect_count: 2,
    stream_warning: 'Temporary camera interruption; reconnecting.',
    last_frame_age_seconds: 3.2,
  },
})
assert.equal(reconnectProgress.streamState, 'reconnecting')
assert.equal(reconnectProgress.streamReconnectCount, 2)
assert.equal(reconnectProgress.lastFrameAgeSeconds, 3.2)
assert.equal(reconnectProgress.streamWarning, 'Temporary camera interruption; reconnecting.')
assert.equal(liveProgress({ stream_stats: { stream_state: 'unexpected' } }).streamState, '')

const rollingPayload = {
  enabled: true,
  publication_sequence: 12,
  requested_version: 11,
  analysis_version: 10,
  analysis_state: 'ready',
  analysis_in_progress: false,
  analyzed_embedding_count: 37,
  last_completed_preprocessing_chunk: 9,
  live_identities: [
    {
      session_person_id: 'live_0001',
      cluster_label: 0,
      status: 'provisional',
      face_count: 8,
      associated_body_count: 7,
      first_seen_chunk: 2,
      last_seen_chunk: 9,
      representative_face_path: 'folder/face one.jpg',
      memory_match: { person_id: 'person_006', name: 'Malek', similarity: 0.84 },
    },
    {
      session_person_id: 'live_0002',
      face_count: 4,
      associated_body_count: 3,
      memory_match: null,
    },
  ],
  live_recognition_events: [
    { event_id: 'event-1', type: 'identity_created', session_person_id: 'live_0001', analysis_version: 8 },
    { event_id: 'event-2', type: 'memory_match_found', session_person_id: 'live_0001', analysis_version: 10, memory_match: { person_id: 'person_006', name: 'Malek', similarity: 0.84 } },
    { event_id: 'event-2', type: 'memory_match_found', session_person_id: 'live_0001', analysis_version: 10 },
    { event_id: 'event-3', type: 'future_event', session_person_id: 'live_0002', analysis_version: 11 },
  ],
}
const rolling = normalizeRollingAnalysis(rollingPayload)
assert.equal(normalizeRollingAnalysis(undefined).enabled, false)
assert.equal(normalizeRollingAnalysis({ enabled: true, analysis_state: 'idle' }).state, 'waiting')
assert.equal(normalizeRollingAnalysis({ enabled: true, analysis_state: 'ready', analysis_in_progress: true }).state, 'analyzing')
assert.equal(rolling.identities.length, 2)
assert.equal(rolling.identities[0].memoryMatch.name, 'Malek')
assert.equal(rolling.identities[1].memoryMatch, null)
const missingName = normalizeRollingAnalysis({
  enabled: true,
  live_identities: [{
    session_person_id: 'live_0003',
    memory_match: { person_id: 'person_009', similarity: 0.7 },
  }],
})
assert.equal(missingName.identities[0].memoryMatch.name, '')
assert.equal(rolling.faceEvidenceCount, 12)
assert.equal(formatSimilarity(0.84), '84%')
assert.equal(formatSimilarity(4), '100%')
assert.equal(formatSimilarity(-1), '0%')
assert.equal(formatSimilarity('malformed'), null)
assert.equal(identityImageUrl('folder/face one.jpg'), '/api/images?path=folder%2Fface%20one.jpg')
assert.equal(identityImageUrl(''), '')
assert.deepEqual(rolling.events.map(event => event.eventId), ['event-3', 'event-2', 'event-1'])
assert.equal(rolling.events[0].type, 'future_event')
assert.equal(rolling.events[1].memoryMatch, null)
assert.equal(normalizeRollingAnalysis({
  enabled: true,
  live_recognition_events: Array.from({ length: 30 }, (_, index) => ({
    event_id: `bounded-${index}`,
    event_type: 'identity_created',
  })),
}).events.length, 20)
assert.equal(isCanonicalFinalizing('live_camera', 'stop_requested', 'process_live_stream', {}), true)
assert.equal(isCanonicalFinalizing('live_camera', 'done', 'finalize', { stop_requested: true }), false)

const previousStatus = { status: 'processing_live_frames', snapshot: { rolling_analysis: rollingPayload } }
const warningStatus = {
  status: 'processing_live_frames',
  snapshot: {
    rolling_analysis: {
      ...rollingPayload,
      publication_sequence: 13,
      analysis_state: 'warning',
      analysis_warning: 'Temporary warning',
      live_identities: [],
      live_recognition_events: [],
    },
  },
}
const retainedWarning = mergeJobStatus(previousStatus, warningStatus)
assert.equal(retainedWarning.snapshot.rolling_analysis.live_identities.length, 2)
assert.equal(retainedWarning.snapshot.rolling_analysis.live_recognition_events.length, 4)
assert.equal(mergeJobStatus(previousStatus, { status: 'done', snapshot: {} }).snapshot.rolling_analysis, undefined)
assert.equal(mergeJobStatus(previousStatus, { status: 'processing_live_frames', snapshot: {} }).snapshot.rolling_analysis.enabled, true)
const reconnectStatus = mergeJobStatus(previousStatus, {
  status: 'processing_live_frames',
  snapshot: {
    stream_stats: {
      stream_state: 'reconnecting',
      stream_warning: 'Temporary camera interruption; reconnecting.',
    },
  },
})
assert.equal(reconnectStatus.snapshot.rolling_analysis.live_identities.length, 2)
assert.equal(reconnectStatus.snapshot.stream_stats.stream_state, 'reconnecting')
assert.equal(mergeJobStatus(previousStatus, null), previousStatus)

const requestGuard = createStatusRequestGuard()
const oldRequest = requestGuard.start('old-job')
const newRequest = requestGuard.start('new-job')
assert.equal(oldRequest.signal.aborted, true)
assert.equal(requestGuard.isCurrent(oldRequest), false)
assert.equal(requestGuard.isCurrent(newRequest), true)
requestGuard.cancel()
assert.equal(newRequest.signal.aborted, true)

let stopUrl = ''
const stopResponse = await postStopRequest('job id', async (url, options) => {
  stopUrl = url
  assert.equal(options.method, 'POST')
  return { ok: true, status: 200, json: async () => ({ status: 'stop_requested' }) }
})
assert.equal(stopUrl, '/api/person/stop/job%20id')
assert.equal(stopResponse.status, 'stop_requested')

let actionCalls = 0
let releaseAction
const ref = { current: null }
const first = runSingleFlight(ref, () => {
  actionCalls += 1
  return new Promise(resolve => { releaseAction = resolve })
})
const second = runSingleFlight(ref, () => { actionCalls += 1 })
assert.equal(first, second)
await Promise.resolve()
assert.equal(actionCalls, 1)
releaseAction()
await first

let scheduledPoll
let cleared = 0
let terminalCalls = 0
const statuses = ['stop_requested', 'stopping', 'done']
const cleanup = startStatusPolling({
  fetchStatus: async () => ({ status: statuses.shift() }),
  onTerminal: () => { terminalCalls += 1 },
  setIntervalFn: (callback, interval) => {
    assert.equal(interval, 2000)
    scheduledPoll = callback
    return 7
  },
  clearIntervalFn: id => { assert.equal(id, 7); cleared += 1 },
})
await new Promise(resolve => setTimeout(resolve, 0))
assert.equal(cleared, 0)
await scheduledPoll()
assert.equal(cleared, 0)
await scheduledPoll()
assert.equal(cleared, 1)
assert.equal(terminalCalls, 1)
cleanup()

let resolvePendingPoll
let terminalAfterCleanup = 0
const pendingCleanup = startStatusPolling({
  fetchStatus: () => new Promise(resolve => { resolvePendingPoll = resolve }),
  onTerminal: () => { terminalAfterCleanup += 1 },
  setIntervalFn: () => 9,
  clearIntervalFn: () => {},
})
await Promise.resolve()
pendingCleanup()
resolvePendingPoll({ status: 'done' })
await new Promise(resolve => setTimeout(resolve, 0))
assert.equal(terminalAfterCleanup, 0)

const privateUri = 'rtsp://private-user:private-password@camera.local/live?token=secret'
const safeMessage = safeErrorMessage(`Failed to stop ${privateUri}`)
assert.equal(safeMessage.includes('private-user'), false)
assert.equal(safeMessage.includes('private-password'), false)
assert.equal(safeMessage.includes('rtsp://'), false)

const appSource = await readFile(new URL('../src/App.jsx', import.meta.url), 'utf8')
const formSource = await readFile(new URL('../src/components/StartForm.jsx', import.meta.url), 'utf8')
const panelSource = await readFile(new URL('../src/components/LiveIdentityPanel.jsx', import.meta.url), 'utf8')
const cardSource = await readFile(new URL('../src/components/LiveIdentityCard.jsx', import.meta.url), 'utf8')
const imageSource = await readFile(new URL('../src/components/SafeIdentityImage.jsx', import.meta.url), 'utf8')
const streamStatsSource = await readFile(new URL('../src/components/LiveStreamStats.jsx', import.meta.url), 'utf8')
assert.match(appSource, /LiveJobControls/)
assert.match(appSource, /LiveIdentityPanel/)
assert.match(appSource, /shouldShowRollingIdentityPanel\(jobStatus\?\.status\)/)
assert.match(appSource, /statusRequestGuardRef\.current\.cancel\(\)/)
assert.match(appSource, /setTab\(2\)/)
assert.match(appSource, /jobStatus\?\.status === 'error'/)
assert.match(appSource, /activeJob=\{activeJob\}/)
assert.match(formSource, /Processing window duration/)
assert.match(formSource, /continues running until Stop is pressed/)
assert.match(panelSource, /Live Identity Analysis/)
assert.match(panelSource, /Finalizing canonical profiles/)
assert.match(cardSource, /Known person detected/)
assert.match(cardSource, /Unknown person detected/)
assert.match(cardSource, /Known person'/)
assert.match(imageSource, /onError=\{\(\) => setFailed\(true\)\}/)
assert.match(imageSource, /No face image/)
assert.match(streamStatsSource, /Camera connection interrupted\. Reconnecting/)
assert.match(streamStatsSource, /streamReconnectCount/)
assert.match(streamStatsSource, /lastFrameAgeSeconds/)
assert.equal(/console\.(log|error|warn)\s*\(/.test(`${appSource}\n${formSource}`), false)

console.log('live camera frontend validation: rolling identity and polling behaviors passed')
