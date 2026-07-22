import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import { existsSync } from 'node:fs'
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import process from 'node:process'

import { build } from 'esbuild'

import {
  canonicalLiveCropPath,
  createStatusRequestGuard,
  mergeJobStatus,
  normalizeRollingAnalysis,
  sanitizeVlmError,
  startStatusPolling,
} from '../src/liveJob.js'

const identity = (overrides = {}) => ({
  live_identity_id: 'live_0001',
  session_person_id: 'live_0001',
  version: 2,
  state: 'provisional',
  decision: 'attach_existing',
  provisional: true,
  canonical_person_id: null,
  face_count: 5,
  body_count: 3,
  candidate_person_id: 'person_004',
  candidate_similarity: 0.86,
  margin: 0.12,
  best_face_path: 'person_004/face_crops/face.jpg',
  best_body_path: 'person_004/body_crops/body.jpg',
  vlm_status: 'queued',
  selected_body_crop: 'person_004/body_crops/body.jpg',
  clothing_description: '',
  vlm_error: null,
  vlm_version: 2,
  ...overrides,
})

const rolling = (identities, overrides = {}) => ({
  enabled: true,
  publication_sequence: 8,
  requested_version: 8,
  analysis_version: 8,
  last_completed_preprocessing_chunk: 7,
  analysis_state: 'ready',
  live_identities: identities,
  vlm_queue_depth: 1,
  vlm_queue_capacity: 2,
  vlm_active_identity: null,
  vlm_completed: 0,
  vlm_failed: 0,
  vlm_dropped: 0,
  vlm_timed_out: 0,
  ...overrides,
})

const status = rollingAnalysis => ({
  status: 'processing_live_frames',
  snapshot: { media_lifecycle_version: 0, rolling_analysis: rollingAnalysis },
})

const normalized = normalizeRollingAnalysis(rolling([
  identity(),
  identity({ state: 'observing', face_count: 1 }),
  identity({ live_identity_id: 'live_0002', session_person_id: 'live_0002' }),
]))
assert.equal(normalized.identities.length, 2, 'duplicate live identities are removed')
assert.equal(normalized.identities[0].liveIdentityId, 'live_0001')
assert.equal(normalized.identities[0].bodyCount, 3)
assert.equal(normalized.identities[0].candidatePersonId, 'person_004')
assert.equal(normalized.identities[0].candidateSimilarity, 0.86)
assert.equal(normalized.identities[0].vlmStatus, 'queued')
assert.equal(normalized.vlmQueueCapacity, 2)
for (const stateValue of [
  'observing', 'provisional', 'new_person', 'attach_existing', 'review_required',
]) {
  const item = normalizeRollingAnalysis(rolling([identity({ state: stateValue })])).identities[0]
  assert.equal(item.state, stateValue, `live identity state ${stateValue} is supported`)
}
for (const vlmValue of [
  'not_started', 'queued', 'processing', 'completed', 'failed', 'timed_out',
]) {
  const item = normalizeRollingAnalysis(rolling([identity({ vlm_status: vlmValue })])).identities[0]
  assert.equal(item.vlmStatus, vlmValue, `VLM state ${vlmValue} is supported`)
}
assert.equal(canonicalLiveCropPath('person_004/face_crops/face.jpg', 'face'), 'person_004/face_crops/face.jpg')
for (const unsafe of [
  '/private/face.jpg',
  'C:\\private\\face.jpg',
  '../person_004/face_crops/face.jpg',
  '_staging/face_crops/face.jpg',
  'person_004/cluster_2/face.jpg',
  'person_004/session/face.jpg',
  'person_004/body_crops/face.jpg',
  'person_004/face_crops/not-an-image.txt',
]) {
  assert.equal(canonicalLiveCropPath(unsafe, 'face'), '', `unsafe live path accepted: ${unsafe}`)
}
assert.equal(sanitizeVlmError('timeout'), 'timeout')
assert.equal(sanitizeVlmError('rtsp://user:secret@camera/live C:\\private'), 'inference_error')

const provisional = status(rolling([identity()]))
const persisted = status(rolling([identity({
  version: 3,
  state: 'attach_existing',
  provisional: false,
  canonical_person_id: 'person_004',
})], { publication_sequence: 9, requested_version: 9, analysis_version: 9 }))
const persistedResult = normalizeRollingAnalysis(
  mergeJobStatus(provisional, persisted).snapshot.rolling_analysis,
)
assert.equal(persistedResult.identities[0].liveIdentityId, 'live_0001')
assert.equal(persistedResult.identities[0].state, 'attach_existing')
assert.equal(persistedResult.identities[0].provisional, false)

const processing = status(rolling([identity({ vlm_status: 'processing' })]))
const completed = status(rolling([identity({
  vlm_status: 'completed',
  clothing_description: 'black jacket, blue jeans',
})], { vlm_completed: 1, vlm_queue_depth: 0 }))
const completedStatus = mergeJobStatus(processing, completed)
const temporarilyOmitted = status(rolling([identity({
  vlm_status: undefined,
  clothing_description: undefined,
  selected_body_crop: undefined,
})], { vlm_completed: 1, vlm_queue_depth: 0 }))
const preservedStatus = mergeJobStatus(completedStatus, temporarilyOmitted)
const preservedIdentity = normalizeRollingAnalysis(
  preservedStatus.snapshot.rolling_analysis,
).identities[0]
assert.equal(preservedIdentity.vlmStatus, 'completed')
assert.equal(preservedIdentity.clothingDescription, 'black jacket, blue jeans')
assert.equal(preservedIdentity.selectedBodyCrop, 'person_004/body_crops/body.jpg')

const stale = status(rolling([identity({ vlm_status: 'processing' })], {
  publication_sequence: 7,
  requested_version: 7,
  analysis_version: 7,
}))
assert.equal(mergeJobStatus(completedStatus, stale), completedStatus)

const requestGuard = createStatusRequestGuard()
const oldRequest = requestGuard.start('job-live')
const currentRequest = requestGuard.start('job-live')
assert.equal(oldRequest.signal.aborted, true)
assert.equal(requestGuard.isCurrent(oldRequest), false)
assert.equal(requestGuard.isCurrent(currentRequest), true)
requestGuard.cancel()

let scheduledPoll
let cleared = 0
let terminalCalls = 0
const pollingCleanup = startStatusPolling({
  fetchStatus: async () => ({ status: 'done' }),
  onTerminal: () => { terminalCalls += 1 },
  setIntervalFn: callback => { scheduledPoll = callback; return 23 },
  clearIntervalFn: id => { assert.equal(id, 23); cleared += 1 },
})
await new Promise(resolve => setTimeout(resolve, 0))
assert.equal(typeof scheduledPoll, 'function')
assert.equal(cleared, 1)
assert.equal(terminalCalls, 1)
pollingCleanup()

const frontendRoot = path.resolve(import.meta.dirname, '..')
const panelPath = path.join(frontendRoot, 'src', 'components', 'LiveIdentityPanel.jsx')
const safeImagePath = path.join(frontendRoot, 'src', 'components', 'SafeImage.jsx')
const panelSource = await readFile(panelPath, 'utf8')
const cardSource = await readFile(
  path.join(frontendRoot, 'src', 'components', 'LiveIdentityCard.jsx'),
  'utf8',
)
assert.match(panelSource, /key=\{identity\.liveIdentityId\}/)
assert.match(cardSource, /SafeImage/)
assert.doesNotMatch(cardSource, /<img\b/)

const temporary = await mkdtemp(path.join(os.tmpdir(), 'waycon-live-identities-'))
const entryPath = path.join(temporary, 'probe.jsx')
const htmlPath = path.join(temporary, 'probe.html')
const bundlePath = path.join(temporary, 'probe.js')
const entry = String.raw`
import React from 'react'
import { createRoot } from 'react-dom/client'
import LiveIdentityPanel from ${JSON.stringify(panelPath.replaceAll('\\', '/'))}
import { clearFailedMediaUrls } from ${JSON.stringify(safeImagePath.replaceAll('\\', '/'))}

const output = document.getElementById('root')
const root = createRoot(output)
let imageMounts = 0
const observer = new MutationObserver(records => {
  for (const record of records) {
    for (const node of record.addedNodes) {
      if (node.nodeName === 'IMG') imageMounts += 1
      if (node.querySelectorAll) imageMounts += node.querySelectorAll('img').length
    }
  }
})
observer.observe(output, { childList: true, subtree: true })
const wait = ms => new Promise(resolve => setTimeout(resolve, ms))
const waitFor = async (test, message) => {
  const started = performance.now()
  while (!test()) {
    if (performance.now() - started > 2500) throw new Error('Timed out: ' + message)
    await wait(10)
  }
}
const assert = (condition, message) => { if (!condition) throw new Error(message) }
const identity = (id, overrides = {}) => ({
  live_identity_id: id,
  session_person_id: id,
  version: 2,
  state: 'provisional',
  provisional: true,
  face_count: 4,
  body_count: 2,
  best_face_path: 'person_004/face_crops/face.jpg',
  best_body_path: 'person_004/body_crops/body.jpg',
  selected_body_crop: 'person_004/body_crops/body.jpg',
  vlm_status: 'queued',
  vlm_version: 2,
  ...overrides,
})
const rolling = identities => ({
  enabled: true,
  publication_sequence: 8,
  requested_version: 8,
  analysis_version: 8,
  analysis_state: 'ready',
  live_identities: identities,
  vlm_queue_depth: 1,
  vlm_queue_capacity: 2,
  vlm_active_identity: 'live_0001',
  vlm_completed: 0,
  vlm_failed: 0,
  vlm_dropped: 0,
  vlm_timed_out: 0,
})
const render = value => root.render(<LiveIdentityPanel rollingAnalysis={value} />)

async function run() {
  clearFailedMediaUrls()
  render(rolling([
    identity('live_0001'),
    identity('live_0001', { face_count: 99 }),
    identity('live_0002', {
      state: 'observing',
      best_face_path: '_staging/face.jpg',
      best_body_path: '../private.jpg',
      selected_body_crop: '',
      vlm_status: 'not_started',
    }),
  ]))
  await waitFor(() => output.querySelectorAll('[data-live-identity-id]').length === 2, 'two unique cards')
  const stableCard = output.querySelector('[data-live-identity-id="live_0001"]')
  assert(stableCard, 'first live identity card exists')
  assert(output.textContent.includes('No face image'), 'missing face placeholder is stable')
  assert(output.textContent.includes('No body image'), 'missing body placeholder is stable')
  await waitFor(() => imageMounts === 2, 'canonical face and body requests')
  await waitFor(() => stableCard.textContent.includes('No face image'), 'failed canonical face placeholder')

  render(rolling([identity('live_0001', { vlm_status: 'processing' })]))
  await waitFor(() => output.querySelector('[data-vlm-status="processing"]'), 'processing VLM state')
  assert(output.querySelector('[data-live-identity-id="live_0001"]') === stableCard, 'React card identity remains stable')
  render(rolling([identity('live_0001', {
    version: 3,
    state: 'attach_existing',
    decision: 'attach_existing',
    provisional: false,
    canonical_person_id: 'person_004',
    vlm_status: 'completed',
    clothing_description: 'black jacket, blue jeans',
  })]))
  await waitFor(() => output.querySelector('[data-vlm-status="completed"]'), 'completed VLM state')
  assert(output.querySelector('[data-live-identity-id="live_0001"]') === stableCard, 'persisted transition keeps card node')
  assert(stableCard.textContent.includes('attach existing'), 'persisted identity state is shown')
  assert(stableCard.textContent.includes('black jacket, blue jeans'), 'completed clothing is shown')
  assert(output.textContent.includes('1 / 2 queued'), 'queue diagnostics are shown')

  for (let index = 0; index < 8; index += 1) {
    render(rolling([identity('live_0001', {
      version: 3,
      state: 'attach_existing',
      canonical_person_id: 'person_004',
      vlm_status: 'completed',
      clothing_description: 'black jacket, blue jeans',
    })]))
    await wait(0)
  }
  assert(imageMounts === 2, 'failed live image URLs are not retried during polling rerenders')
  observer.disconnect()
  root.unmount()
  output.textContent = 'LIVE_IDENTITY_COMPONENT_PROBE_PASS 12'
}

run().catch(error => {
  document.body.dataset.probe = 'fail'
  output.textContent = 'LIVE_IDENTITY_COMPONENT_PROBE_FAIL ' + (error?.stack || error)
})
`
const html = '<!doctype html><html><body><div id="root"></div><script type="module" src="./probe.js"></script></body></html>'
const browserCandidates = process.platform === 'win32'
  ? [
      'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
      'C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe',
      'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
    ]
  : ['/usr/bin/microsoft-edge', '/usr/bin/google-chrome', '/usr/bin/chromium']

try {
  await writeFile(entryPath, entry, 'utf8')
  await writeFile(htmlPath, html, 'utf8')
  await build({
    entryPoints: [entryPath],
    bundle: true,
    outfile: bundlePath,
    format: 'esm',
    platform: 'browser',
    jsx: 'automatic',
    absWorkingDir: frontendRoot,
    nodePaths: [path.join(frontendRoot, 'node_modules')],
    define: { 'process.env.NODE_ENV': '"production"' },
    logLevel: 'silent',
  })
  const browser = browserCandidates.find(existsSync)
  if (!browser) throw new Error('No installed headless Edge/Chrome executable was found')
  const pageUrl = new URL('file:///' + htmlPath.replaceAll('\\', '/')).href
  const result = spawnSync(browser, [
    '--headless=new', '--disable-gpu', '--disable-extensions', '--no-first-run',
    '--no-default-browser-check', '--allow-file-access-from-files',
    '--run-all-compositor-stages-before-draw', '--virtual-time-budget=5000',
    '--dump-dom', pageUrl,
  ], { encoding: 'utf8', timeout: 15000, maxBuffer: 4 * 1024 * 1024 })
  if (result.error) throw result.error
  if (result.status !== 0 || !result.stdout.includes('LIVE_IDENTITY_COMPONENT_PROBE_PASS')) {
    throw new Error(`Actual live identity component probe failed.\n${result.stdout}\n${result.stderr}`)
  }
  console.log('Live identity card validation passed (data, polling, and 12 browser assertions).')
} finally {
  await rm(temporary, { recursive: true, force: true })
}
