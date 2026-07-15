import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'

import {
  cameraPhase,
  displayMetric,
  isJobActive,
  isTerminalStatus,
  isStopPending,
  liveProgress,
  postStopRequest,
  runSingleFlight,
  safeErrorMessage,
  shouldShowStop,
  startStatusPolling,
  statusLabel,
} from '../src/liveJob.js'

assert.equal(shouldShowStop('job-1', 'live_camera', 'processing_live_frames'), true)
assert.equal(shouldShowStop('job-1', 'video_file', 'processing_video'), false)
assert.equal(shouldShowStop('job-1', 'live_camera', 'filtering', 'filter_quality', {}), false)
assert.equal(shouldShowStop('job-1', 'live_camera', 'stopping', 'filter_quality', { stop_requested: true }), true)
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
  setIntervalFn: callback => { scheduledPoll = callback; return 7 },
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

const privateUri = 'rtsp://private-user:private-password@camera.local/live?token=secret'
const safeMessage = safeErrorMessage(`Failed to stop ${privateUri}`)
assert.equal(safeMessage.includes('private-user'), false)
assert.equal(safeMessage.includes('private-password'), false)
assert.equal(safeMessage.includes('rtsp://'), false)

const appSource = await readFile(new URL('../src/App.jsx', import.meta.url), 'utf8')
const formSource = await readFile(new URL('../src/components/StartForm.jsx', import.meta.url), 'utf8')
assert.match(appSource, /LiveJobControls/)
assert.match(appSource, /activeJob=\{activeJob\}/)
assert.match(formSource, /Processing window duration/)
assert.match(formSource, /continues running until Stop is pressed/)
assert.equal(/console\.(log|error|warn)\s*\(/.test(`${appSource}\n${formSource}`), false)

console.log('live camera frontend validation: 15 behaviors passed')
