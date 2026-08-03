import { spawnSync } from 'node:child_process'
import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import process from 'node:process'

import { build } from 'esbuild'

const frontendRoot = path.resolve(import.meta.dirname, '..')
const componentPath = path.join(frontendRoot, 'src', 'components', 'IdentityReviewView.jsx')
const temporary = await mkdtemp(path.join(os.tmpdir(), 'waycon-identity-review-'))
const entryPath = path.join(temporary, 'probe.jsx')
const bundlePath = path.join(temporary, 'probe.js')
const htmlPath = path.join(temporary, 'probe.html')

const entry = String.raw`
import React from 'react'
import { createRoot } from 'react-dom/client'
import IdentityReviewView from ${JSON.stringify(componentPath.replaceAll('\\', '/'))}

const output = document.getElementById('root')
const results = []
const assert = (condition, message) => {
  if (!condition) throw new Error(message)
  results.push(message)
}
const deferred = () => {
  let resolve
  let reject
  const promise = new Promise((yes, no) => { resolve = yes; reject = no })
  return { promise, resolve, reject }
}
const response = (data, status = 200) => new Response(JSON.stringify(data), {
  status,
  headers: { 'Content-Type': 'application/json' },
})
const summary = id => ({
  suggestion_id: id,
  source_person_id: 'source-' + id,
  candidate_person_id: 'target-' + id,
  source_name: 'Source ' + id,
  candidate_name: 'Target ' + id,
  source_preview: null,
  similarity: 0.88,
  margin: 0.1,
  status: 'pending',
  created_at: '2026-07-21T10:00:00Z',
})
const queue = ids => ({ reviews: ids.map(summary), pending_count: ids.length })
const detail = id => ({
  suggestion: {
    suggestion_id: id,
    status: 'pending',
    similarity: 0.88,
    margin: 0.1,
    created_at: '2026-07-21T10:00:00Z',
  },
  source_profile: {
    person_id: 'source-' + id,
    name: 'Source ' + id,
    is_active: true,
    cameras: [],
    profile_image: null,
  },
  candidate_profile: {
    person_id: 'target-' + id,
    name: 'Target ' + id,
    is_active: true,
    cameras: [],
    profile_image: null,
  },
  source_gallery: [], candidate_gallery: [],
  source_appearances: [], candidate_appearances: [],
  source_recognition_events: [], candidate_recognition_events: [],
})
const decision = id => ({
  suggestion_id: id,
  source_person_id: 'source-' + id,
  target_person_id: 'target-' + id,
  idempotent_replay: false,
  status: 'resolved',
})
const wait = ms => new Promise(resolve => setTimeout(resolve, ms))
const waitFor = async (test, message, timeout = 2500) => {
  const started = performance.now()
  while (!test()) {
    if (performance.now() - started > timeout) throw new Error('Timed out: ' + message)
    await wait(10)
  }
}
const button = label => [...output.querySelectorAll('button')].find(
  item => item.textContent.trim().includes(label),
)
const click = label => {
  const target = button(label)
  if (!target) throw new Error('Missing button: ' + label)
  target.click()
}
const mount = handler => {
  globalThis.fetch = handler
  const root = createRoot(output)
  root.render(<IdentityReviewView />)
  return root
}
const reset = root => {
  root.unmount()
  output.replaceChildren()
}

async function raceAndBannerScenario() {
  const delayedA = deferred()
  const submissions = []
  let queueCalls = 0
  let detailACalls = 0
  const root = mount(async (url, options = {}) => {
    if (url.startsWith('/api/identity-reviews?')) {
      queueCalls += 1
      return response(queueCalls === 1 ? queue(['A', 'B']) : queue(['A', 'C']))
    }
    if (url === '/api/identity-reviews/A') {
      detailACalls += 1
      return detailACalls === 1 ? delayedA.promise : response(detail('A'))
    }
    if (url === '/api/identity-reviews/B') return response(detail('B'))
    if (url === '/api/identity-reviews/C') return response(detail('C'))
    if (url === '/api/identity-reviews/B/accept' && options.method === 'POST') {
      submissions.push(['accept', 'B'])
      return response(decision('B'))
    }
    throw new Error('Unexpected request ' + url)
  })
  await waitFor(() => button('Source B'), 'queue B')
  await waitFor(() => output.textContent.includes('Loading source'), 'delayed A detail')
  click('Source B')
  assert(!output.textContent.includes('Suggestion A'), 'selection change clears old detail')
  await waitFor(() => output.textContent.includes('Suggestion B'), 'B detail')
  delayedA.resolve(response(detail('A')))
  await wait(30)
  assert(output.textContent.includes('Suggestion B'), 'late A success cannot overwrite B')
  click('Accept merge')
  await waitFor(() => button('Confirm accept merge'), 'accept confirmation')
  click('Confirm accept merge')
  await waitFor(() => submissions.length === 1, 'B accept submission')
  assert(JSON.stringify(submissions) === JSON.stringify([['accept', 'B']]), 'Accept submits displayed B')
  await waitFor(() => output.textContent.includes('Suggestion A'), 'automatic queue advance')
  assert(output.textContent.includes('Accepted review B:'), 'success banner survives automatic advance and names B')
  click('Source C')
  await waitFor(() => output.textContent.includes('Suggestion C'), 'manual C selection')
  assert(!output.textContent.includes('Accepted review B:'), 'manual selection clears success banner')
  reset(root)
}

async function lateErrorScenario() {
  const delayedA = deferred()
  const root = mount(async url => {
    if (url.startsWith('/api/identity-reviews?')) return response(queue(['A', 'B']))
    if (url === '/api/identity-reviews/A') return delayedA.promise
    if (url === '/api/identity-reviews/B') return response(detail('B'))
    throw new Error('Unexpected request ' + url)
  })
  await waitFor(() => button('Source B'), 'late-error queue')
  click('Source B')
  await waitFor(() => output.textContent.includes('Suggestion B'), 'late-error B detail')
  delayedA.resolve(response({ error: 'late A failure' }, 500))
  await wait(30)
  assert(output.textContent.includes('Suggestion B'), 'late A error cannot overwrite B')
  assert(!output.textContent.includes('late A failure'), 'late A error remains hidden')
  reset(root)
}

async function rejectScenario() {
  const submissions = []
  let queueCalls = 0
  const root = mount(async (url, options = {}) => {
    if (url.startsWith('/api/identity-reviews?')) {
      queueCalls += 1
      return response(queueCalls === 1 ? queue(['B']) : queue([]))
    }
    if (url === '/api/identity-reviews/B') return response(detail('B'))
    if (url === '/api/identity-reviews/B/reject' && options.method === 'POST') {
      submissions.push(['reject', 'B'])
      return response(decision('B'))
    }
    throw new Error('Unexpected request ' + url)
  })
  await waitFor(() => output.textContent.includes('Suggestion B'), 'reject B detail')
  click('Reject match')
  await waitFor(() => submissions.length === 1, 'B reject submission')
  assert(JSON.stringify(submissions) === JSON.stringify([['reject', 'B']]), 'Reject submits displayed B')
  await waitFor(() => output.textContent.includes('Rejected review B:'), 'reject banner')
  reset(root)
}

async function mismatchScenario() {
  const submissions = []
  const root = mount(async (url, options = {}) => {
    if (url.startsWith('/api/identity-reviews?')) return response(queue(['B']))
    if (url === '/api/identity-reviews/B') return response(detail('A'))
    if (options.method === 'POST') submissions.push(url)
    throw new Error('Unexpected request ' + url)
  })
  await waitFor(() => output.textContent.includes('malformed'), 'mismatch error')
  assert(!button('Accept merge') && !button('Reject match'), 'mismatched detail ID refuses actions')
  assert(submissions.length === 0, 'mismatched detail submits nothing')
  reset(root)
}

async function currentFailureScenario() {
  const root = mount(async url => {
    if (url.startsWith('/api/identity-reviews?')) return response(queue(['F']))
    if (url === '/api/identity-reviews/F') return response({ error: 'current detail failed' }, 500)
    throw new Error('Unexpected request ' + url)
  })
  await waitFor(() => output.textContent.includes('current detail failed'), 'current failure')
  assert(!output.textContent.includes('Loading source'), 'current request failure clears loading')
  reset(root)
}

async function unmountScenario() {
  const late = deferred()
  const root = mount(async url => {
    if (url.startsWith('/api/identity-reviews?')) return response(queue(['U']))
    if (url === '/api/identity-reviews/U') return late.promise
    throw new Error('Unexpected request ' + url)
  })
  await waitFor(() => output.textContent.includes('Loading source'), 'unmount pending detail')
  reset(root)
  late.resolve(response(detail('U')))
  await wait(30)
  assert(output.childNodes.length === 0, 'unmount prevents late updates')
}

async function timeoutScenario() {
  const root = mount(async (url, options = {}) => {
    if (url.startsWith('/api/identity-reviews?')) return response(queue(['T']))
    if (url === '/api/identity-reviews/T') {
      return new Promise((_resolve, reject) => {
        options.signal.addEventListener('abort', () => {
          reject(new DOMException('aborted', 'AbortError'))
        }, { once: true })
      })
    }
    throw new Error('Unexpected request ' + url)
  })
  await waitFor(() => output.textContent.includes('timed out'), 'request timeout', 14000)
  assert(!output.textContent.includes('Loading source'), 'timeout clears loading without stale detail')
  reset(root)
}

async function run() {
  await raceAndBannerScenario()
  await lateErrorScenario()
  await rejectScenario()
  await mismatchScenario()
  await currentFailureScenario()
  await unmountScenario()
  await timeoutScenario()
  document.body.dataset.probe = 'pass'
  document.body.dataset.assertions = String(results.length)
  output.textContent = 'IDENTITY_REVIEW_COMPONENT_PROBE_PASS ' + results.length
}

run().catch(error => {
  document.body.dataset.probe = 'fail'
  output.textContent = 'IDENTITY_REVIEW_COMPONENT_PROBE_FAIL ' + (error?.stack || error)
})
`

const html = `<!doctype html><html><body><div id="root"></div><script type="module" src="./probe.js"></script></body></html>`

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

  const { existsSync } = await import('node:fs')
  const browser = browserCandidates.find(existsSync)
  if (!browser) throw new Error('No installed headless Edge/Chrome executable was found')
  const pageUrl = new URL('file:///' + htmlPath.replaceAll('\\', '/')).href
  const result = spawnSync(browser, [
    '--headless=new',
    '--disable-gpu',
    '--disable-extensions',
    '--no-first-run',
    '--no-default-browser-check',
    '--allow-file-access-from-files',
    '--run-all-compositor-stages-before-draw',
    '--virtual-time-budget=18000',
    '--dump-dom',
    pageUrl,
  ], { encoding: 'utf8', timeout: 30000, maxBuffer: 4 * 1024 * 1024 })
  if (result.error) throw result.error
  if (result.status !== 0 || !result.stdout.includes('IDENTITY_REVIEW_COMPONENT_PROBE_PASS')) {
    throw new Error(`Actual-component probe failed.\n${result.stdout}\n${result.stderr}`)
  }
  const match = result.stdout.match(/IDENTITY_REVIEW_COMPONENT_PROBE_PASS (\d+)/)
  console.log(`IdentityReviewView actual-component probe passed (${match?.[1] || '?'} assertions).`)
} finally {
  await rm(temporary, { recursive: true, force: true })
}
