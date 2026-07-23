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
  second_candidate_person_id: 'person_009',
  second_candidate_similarity: 0.74,
  margin: 0.12,
  reason: 'insufficient_face_observations',
  observation_count: 5,
  evidence_version: 2,
  evidence_signature: 'face-signature-2',
  persisted: false,
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
assert.equal(normalized.identities[0].secondCandidatePersonId, 'person_009')
assert.equal(normalized.identities[0].secondCandidateSimilarity, 0.74)
assert.equal(normalized.identities[0].observationCount, 5)
assert.equal(normalized.identities[0].evidenceVersion, 2)
assert.equal(normalized.identities[0].persisted, false)
assert.equal(normalized.identities[0].vlmStatus, 'queued')
assert.equal(normalized.vlmQueueCapacity, 2)
const tenNormalized = normalizeRollingAnalysis(rolling([
  ...Array.from({ length: 10 }, (_, index) => identity({
    live_identity_id: `live_${String(index + 1).padStart(4, '0')}`,
    session_person_id: `live_${String(index + 1).padStart(4, '0')}`,
  })),
  identity({ live_identity_id: 'live_0005', session_person_id: 'live_0005' }),
]))
assert.equal(tenNormalized.identities.length, 10, 'all ten unique identities are retained')
assert.deepEqual(
  tenNormalized.identities.map(item => item.liveIdentityId),
  Array.from({ length: 10 }, (_, index) => `live_${String(index + 1).padStart(4, '0')}`),
)
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
assert.equal(persistedResult.identities[0].persisted, true)

const noFaceUpdate = status(rolling([], {
  publication_sequence: 10,
  requested_version: 10,
  analysis_version: 10,
}))
const noFaceResult = normalizeRollingAnalysis(
  mergeJobStatus(persisted, noFaceUpdate).snapshot.rolling_analysis,
)
assert.equal(noFaceResult.identities.length, 1, 'no-face update retains known live cards')
assert.equal(noFaceResult.identities[0].liveIdentityId, 'live_0001')

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
const appSource = await readFile(path.join(frontendRoot, 'src', 'App.jsx'), 'utf8')
const liveJobSource = await readFile(path.join(frontendRoot, 'src', 'liveJob.js'), 'utf8')
const cssSource = await readFile(path.join(frontendRoot, 'src', 'App.css'), 'utf8')
const cardSource = await readFile(
  path.join(frontendRoot, 'src', 'components', 'LiveIdentityCard.jsx'),
  'utf8',
)
assert.match(panelSource, /key=\{identity\.liveIdentityId\}/)
assert.match(panelSource, /Live Identities — \{analysis\.identities\.length\}/)
assert.doesNotMatch(panelSource, /\.slice\s*\([^)]*4/)
assert.doesNotMatch(liveJobSource, /\.slice\s*\([^)]*4/)
assert.match(appSource, /className="live-identities-section"/)
assert.match(cssSource, /\.live-identities-section,\s*\.live-identity-panel\s*\{[^}]*grid-column:\s*1\s*\/\s*-1[^}]*height:\s*auto[^}]*max-height:\s*none[^}]*overflow:\s*visible/s)
assert.match(cssSource, /\.live-identity-grid\s*\{[^}]*grid-template-columns:\s*repeat\(auto-fill,\s*minmax\(280px,\s*1fr\)\)[^}]*gap:\s*16px[^}]*height:\s*auto[^}]*max-height:\s*none[^}]*overflow:\s*visible[^}]*align-items:\s*start/s)
assert.match(cssSource, /#root\s*\{[^}]*min-height:\s*100vh[^}]*height:\s*auto[^}]*overflow-y:\s*auto/s)
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
const render = value => root.render(
  <>
    <header className="header"><h1>Forensics — Person Creation</h1></header>
    <nav className="tabs"><button className="tab-btn active">Progress & Crops</button></nav>
    <main className="tab-content">
      <section className="card" data-layout-predecessor="camera-controls">Camera controls</section>
      <section className="card" data-layout-predecessor="processing-status">Processing status</section>
      <div className="live-identities-section">
        <LiveIdentityPanel rollingAnalysis={value} />
      </div>
    </main>
  </>,
)

async function run() {
  clearFailedMediaUrls()
  render(rolling([identity('live_0001', {
    face_count: 1,
    observation_count: 1,
    evidence_version: 1,
    evidence_signature: 'first-face',
    decision: 'review_required',
    reason: 'insufficient_face_observations',
    provisional: true,
    persisted: false,
  })]))
  await waitFor(() => output.querySelectorAll('[data-live-identity-id]').length === 1, 'one-face provisional card')
  const firstFaceCard = output.querySelector('[data-live-identity-id="live_0001"]')
  assert(firstFaceCard.textContent.includes('Provisional comparison'), 'one-face result is labelled provisional')
  assert(firstFaceCard.textContent.includes('PersistedNo'), 'one-face result is not labelled persisted')
  assert(firstFaceCard.textContent.includes('1 face observation'), 'one-face observation count is displayed')

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

  const completedIdentity = identity('live_0001', {
    version: 3,
    state: 'attach_existing',
    decision: 'attach_existing',
    provisional: false,
    canonical_person_id: 'person_004',
    vlm_status: 'completed',
    clothing_description: 'black jacket, blue jeans',
  })
  const identityWithoutMedia = index => identity(
    'live_' + String(index).padStart(4, '0'),
    {
      best_face_path: '',
      best_body_path: '',
      selected_body_crop: '',
      vlm_status: 'not_started',
    },
  )
  const firstFour = [completedIdentity, ...[2, 3, 4].map(identityWithoutMedia)]
  render(rolling(firstFour))
  await waitFor(() => output.querySelectorAll('[data-live-identity-id]').length === 4, 'first four cards')
  const existingNodes = new Map(firstFour.map(item => [
    item.live_identity_id,
    output.querySelector('[data-live-identity-id="' + item.live_identity_id + '"]'),
  ]))

  const allSeven = [...firstFour, ...[5, 6, 7].map(identityWithoutMedia)]
  render(rolling(allSeven))
  await waitFor(() => output.querySelectorAll('[data-live-identity-id]').length === 7, 'all seven cards')
  const seventhCard = output.querySelector('[data-live-identity-id="live_0007"]')
  assert(output.textContent.includes('Live Identities — 7'), 'header reports the actual seven-card count')
  assert(seventhCard.getBoundingClientRect().width > 0 && seventhCard.getBoundingClientRect().height > 0, 'seventh card has non-zero geometry')

  const allTen = [...firstFour, ...[5, 6, 7, 8, 9, 10].map(identityWithoutMedia)]
  render(rolling([...allTen, identityWithoutMedia(5)]))
  await waitFor(() => output.querySelectorAll('[data-live-identity-id]').length === 10, 'all ten unique cards')
  const renderedCards = [...output.querySelectorAll('[data-live-identity-id]')]
  assert(new Set(renderedCards.map(node => node.dataset.liveIdentityId)).size === 10, 'no duplicate cards render')
  for (const [identityId, node] of existingNodes) {
    assert(output.querySelector('[data-live-identity-id="' + identityId + '"]') === node, 'existing card DOM identity remains stable')
  }
  assert(output.textContent.includes('Live Identities — 10'), 'header reports the actual ten-card count')
  assert(getComputedStyle(renderedCards[0].parentElement).display === 'grid', 'identity cards use a CSS grid')

  window.scrollTo(0, 0)
  await wait(25)
  const scrollingElement = document.scrollingElement
  assert(scrollingElement.scrollHeight > scrollingElement.clientHeight, 'page scrolls for later identity rows')
  const documentHeight = scrollingElement.scrollHeight
  const clippingReason = card => {
    const cardBounds = card.getBoundingClientRect()
    for (let ancestor = card.parentElement; ancestor && ancestor !== document.documentElement; ancestor = ancestor.parentElement) {
      if (ancestor === document.body) continue
      const style = getComputedStyle(ancestor)
      const overflowValues = [style.overflow, style.overflowX, style.overflowY]
      if (overflowValues.some(value => value === 'hidden' || value === 'clip')) {
        return ancestor.className || ancestor.id || ancestor.tagName
      }
      if (overflowValues.some(value => value === 'auto' || value === 'scroll')) {
        const ancestorBounds = ancestor.getBoundingClientRect()
        if (cardBounds.top < ancestorBounds.top - 1 || cardBounds.bottom > ancestorBounds.bottom + 1) {
          return ancestor.className || ancestor.id || ancestor.tagName
        }
      }
    }
    return ''
  }
  assert(output.querySelector('[data-live-identity-id="live_0007"]') === seventhCard, 'seventh card node is preserved')
  assert(!clippingReason(seventhCard), 'seventh card has no clipping ancestor')
  seventhCard.scrollIntoView({ block: 'center' })
  await wait(25)
  {
    const bounds = seventhCard.getBoundingClientRect()
    assert(bounds.bottom > 0 && bounds.top < window.innerHeight, 'scrolling reveals the seventh card')
  }
  for (let index = 4; index < 10; index += 1) {
    const card = renderedCards[index]
    const style = getComputedStyle(card)
    const bounds = card.getBoundingClientRect()
    const documentTop = bounds.top + window.scrollY
    const documentBottom = bounds.bottom + window.scrollY
    assert(style.display !== 'none' && style.visibility !== 'hidden', 'card ' + (index + 1) + ' is visually displayed')
    assert(bounds.width > 0 && bounds.height > 0, 'card ' + (index + 1) + ' has non-zero geometry')
    assert(!clippingReason(card), 'card ' + (index + 1) + ' is not clipped by an ancestor')
    assert(documentTop >= 0 && documentBottom <= documentHeight + 1, 'card ' + (index + 1) + ' is inside the scrollable document')
  }

  const rowRepresentatives = []
  for (const card of renderedCards) {
    const top = card.getBoundingClientRect().top + window.scrollY
    if (!rowRepresentatives.some(item => Math.abs(item.top - top) < 2)) {
      rowRepresentatives.push({ top, card })
    }
  }
  assert(rowRepresentatives.length >= 3, 'ten cards create at least three visual rows')
  for (const { card } of rowRepresentatives.slice(1)) {
    card.scrollIntoView({ block: 'center' })
    await wait(25)
    const bounds = card.getBoundingClientRect()
    assert(bounds.bottom > 0 && bounds.top < window.innerHeight, 'scrolling reveals each later identity row')
  }
  assert(window.scrollY > 0, 'the document vertically scrolled to reveal later rows')

  const tenCardNodes = new Map(renderedCards.map(card => [card.dataset.liveIdentityId, card]))
  const allEleven = [...allTen, identityWithoutMedia(11)]
  render(rolling([...allEleven, identityWithoutMedia(5)]))
  await waitFor(() => output.querySelectorAll('[data-live-identity-id]').length === 11, 'eleventh card appends')
  const elevenCards = [...output.querySelectorAll('[data-live-identity-id]')]
  assert(new Set(elevenCards.map(node => node.dataset.liveIdentityId)).size === 11, 'eleven cards remain unique')
  for (const [identityId, node] of tenCardNodes) {
    assert(output.querySelector('[data-live-identity-id="' + identityId + '"]') === node, 'appending preserves existing card DOM nodes')
  }
  assert(output.textContent.includes('Live Identities — 11'), 'header updates to the actual eleven-card count')

  for (let index = 0; index < 8; index += 1) {
    render(rolling(allEleven))
    await wait(0)
  }
  assert(imageMounts === 2, 'failed live image URLs are not retried during polling rerenders')
  observer.disconnect()
  root.unmount()
  output.textContent = 'LIVE_IDENTITY_COMPONENT_PROBE_PASS VISUAL_ROWS_' + rowRepresentatives.length
}

run().catch(error => {
  document.body.dataset.probe = 'fail'
  output.textContent = 'LIVE_IDENTITY_COMPONENT_PROBE_FAIL ' + (error?.stack || error)
})
`
const html = `<!doctype html><html><head><style>${cssSource}</style></head><body><div id="root"></div><script type="module" src="./probe.js"></script></body></html>`
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
  const visualRows = result.stdout.match(/LIVE_IDENTITY_COMPONENT_PROBE_PASS VISUAL_ROWS_(\d+)/)?.[1]
  if (result.status !== 0 || !visualRows) {
    throw new Error(`Actual live identity component probe failed.\n${result.stdout}\n${result.stderr}`)
  }
  console.log(`Live identity card validation passed (${visualRows} visual rows, 10-card geometry, ancestor clipping, scrolling, and stable 11-card append).`)
} finally {
  await rm(temporary, { recursive: true, force: true })
}
