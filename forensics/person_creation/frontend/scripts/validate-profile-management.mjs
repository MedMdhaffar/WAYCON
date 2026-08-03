import { spawnSync } from 'node:child_process'
import { existsSync } from 'node:fs'
import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import process from 'node:process'

import { build } from 'esbuild'

const frontendRoot = path.resolve(import.meta.dirname, '..')
const componentPath = path.join(frontendRoot, 'src', 'components', 'ProfileManager.jsx')
const temporary = await mkdtemp(path.join(os.tmpdir(), 'waycon-profile-management-'))
const entryPath = path.join(temporary, 'probe.jsx')
const htmlPath = path.join(temporary, 'probe.html')
const bundlePath = path.join(temporary, 'probe.js')

const entry = String.raw`
import React from 'react'
import { createRoot } from 'react-dom/client'
import ProfileManager, { proposedNameFromFilename } from ${JSON.stringify(componentPath.replaceAll('\\', '/'))}

const output = document.getElementById('root')
const assert = (condition, message) => { if (!condition) throw new Error(message) }
const wait = ms => new Promise(resolve => setTimeout(resolve, ms))
const waitFor = async (test, message) => {
  const started = performance.now()
  for (;;) {
    let value
    try { value = test() } catch { value = false }
    if (value) return value
    if (performance.now() - started > 3000) throw new Error('Timed out: ' + message)
    await wait(10)
  }
}
const byText = (selector, text) =>
  [...output.querySelectorAll(selector)].find(node => node.textContent.trim() === text)
const includesText = (selector, text) =>
  [...output.querySelectorAll(selector)].find(node => node.textContent.includes(text))
const nativeValue = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set
const nativeSelectValue = Object.getOwnPropertyDescriptor(window.HTMLSelectElement.prototype, 'value').set

let createdUrls = 0
let revokedUrls = 0
const realCreate = URL.createObjectURL.bind(URL)
const realRevoke = URL.revokeObjectURL.bind(URL)
URL.createObjectURL = blob => { createdUrls += 1; return realCreate(blob) }
URL.revokeObjectURL = url => { revokedUrls += 1; return realRevoke(url) }

let blockingDialogs = 0
window.confirm = () => { blockingDialogs += 1; return true }
window.alert = () => { blockingDialogs += 1 }
window.prompt = () => { blockingDialogs += 1; return '' }

const requests = []
const BATCH_ID = 'batch-abc'
const IDENTITY_OK = 'identity-ok'
const IDENTITY_BAD = 'identity-bad'

const PROFILES = [
  {
    person_id: 'person_001', name: 'Khalifa Bouneb', is_active: true,
    merged_into_person_id: null, notes: '', identity_source: 'phone+video',
    profile_image: 'person_001/face_crops/stale.jpg',
    effective_profile_image: 'person_001/face_crops/phone_new.jpg',
    profile_image_origin: 'supervisor_phone_crop',
    image_candidates: ['person_001/face_crops/phone_new.jpg'],
    phone_photo_count: 2, video_evidence_count: 1,
  },
  {
    person_id: 'person_002', name: 'Archived Person', is_active: false,
    merged_into_person_id: null, notes: '', identity_source: 'video',
    profile_image: null, effective_profile_image: 'person_002/face_crops/video.jpg',
    profile_image_origin: 'video_crop', image_candidates: ['person_002/face_crops/video.jpg'],
    phone_photo_count: 0, video_evidence_count: 4,
  },
]

const DETAIL = {
  person_id: 'person_001', name: 'Khalifa Bouneb', notes: 'A note',
  identity_source: 'phone+video', enrolled_at: '2026-01-01', updated_at: '2026-02-02',
  is_active: true, merged_into_person_id: null, state: 'active',
  profile_image: 'person_001/face_crops/stale.jpg',
  effective_profile_image: 'person_001/face_crops/phone_new.jpg',
  profile_image_origin: 'supervisor_phone_crop',
  image_candidates: ['person_001/face_crops/phone_new.jpg'],
  phone_photos: [
    {
      source_id: 'src-new', face_crop_path: 'person_001/face_crops/phone_new.jpg',
      original_image_path: 'person_001/phone_originals/phone_new.jpg',
      source_filename: 'newest.jpg', created_at: '2026-02-02', is_primary: true,
      is_supervisor_selected: true, quality: { sharpness: 900 }, face_bbox: [1, 2, 3, 4],
    },
    {
      source_id: 'src-old', face_crop_path: 'person_001/face_crops/phone_old.jpg',
      original_image_path: 'person_001/phone_originals/phone_old.jpg',
      source_filename: 'older.jpg', created_at: '2026-01-05', is_primary: false,
      is_supervisor_selected: false, quality: { sharpness: 500 }, face_bbox: [1, 2, 3, 4],
    },
  ],
  video_evidence: [{ path: 'person_001/face_crops/video.jpg', session_date: '2026-01-01' }],
  recent_appearances: [{ id: 1, event_type: 'phone_evidence_appended', ts: '2026-02-02' }],
  pending_review_suggestions: [
    {
      kind: 'phone_import', review_key: 'review-1', candidate_person_id: 'person_001',
      proposed_name: 'Maybe Khalifa', similarity: 0.73, reason: 'similarity_between_thresholds',
      evidence: [{ face_crop_path: '_profile_reviews/review-1/crop.jpg' }],
    },
  ],
}

const READY_BATCH = {
  batch_id: BATCH_ID, state: 'ready', created_at: '2026-02-02', can_commit: true,
  progress: { processed: 3, total: 3, valid: 2, existing_matches: 1, new_profiles: 0, review_required: 0, failed: 1 },
  items: [],
  identities: [
    {
      identity_id: IDENTITY_OK,
      source_ids: ['a', 'b'],
      photos: [
        { source_id: 'a', source_filename: 'khalifa_1.jpg', original_photo: 'u/a.jpg', face_crop: 'c/a.jpg', state: 'valid' },
        { source_id: 'b', source_filename: 'khalifa_2.jpg', original_photo: 'u/b.jpg', face_crop: 'c/b.jpg', state: 'valid' },
      ],
      primary_source_id: 'a', proposed_name: 'Khalifa 1', quality: { sharpness: 812 },
      grouped_photo_count: 2, similarity: 0.94,
      memory_match: { reason: 'strong_clear_match' },
      existing_candidate: { person_id: 'person_001', name: 'Khalifa Bouneb', profile_image: 'person_001/face_crops/phone_new.jpg' },
      proposed_action: 'attach_existing', observation_count: 2,
      identity_source: 'phone_supervised', trusted_enrolment_applied: false,
      validation_error: null,
    },
    {
      identity_id: IDENTITY_BAD,
      source_ids: ['c'],
      photos: [{ source_id: 'c', source_filename: 'blurry.jpg', original_photo: 'u/c.jpg', state: 'no_face' }],
      primary_source_id: 'c', proposed_name: 'Blurry', quality: null,
      grouped_photo_count: 1, similarity: null,
      memory_match: { reason: 'no_face' }, existing_candidate: null,
      proposed_action: 'skip', observation_count: 0,
      identity_source: 'phone_supervised', trusted_enrolment_applied: false,
      validation_error: 'exactly one face is required',
    },
  ],
}

window.fetch = async (url, options = {}) => {
  const method = (options.method || 'GET').toUpperCase()
  let body = null
  if (options.body && typeof options.body === 'string') body = JSON.parse(options.body)
  requests.push({ url, method, body, form: options.body instanceof FormData ? options.body : null })
  const json = payload => new Response(JSON.stringify(payload), {
    status: 200, headers: { 'content-type': 'application/json' },
  })
  if (url.startsWith('/api/profiles?state=')) return json({ profiles: PROFILES })
  if (url === '/api/profiles/reviews') return json({ reviews: DETAIL.pending_review_suggestions })
  if (url === '/api/profiles/import/preview') return json({ ...READY_BATCH, state: 'processing', identities: [] })
  if (url.includes('/import/') && url.endsWith('/status')) return json(READY_BATCH)
  if (url.includes('/import/') && url.endsWith('/commit')) {
    return json({ batch_id: BATCH_ID, state: 'committed', results: [], summary: {
      profiles_created: 0, profiles_updated: 1, items_sent_to_review: 0, skipped_items: 1, failed_items: 0,
    } })
  }
  if (url.includes('/import/') && url.endsWith('/cancel')) return json({ batch_id: BATCH_ID, state: 'cancelled' })
  if (url === '/api/profiles/reviews/review-1/resolve') {
    return json({ review_key: 'review-1', action: body.action, status: body.action === 'skip' ? 'skipped' : 'updated' })
  }
  if (url === '/api/profiles/person_001' || url.endsWith('/primary-photo')) return json(DETAIL)
  return json({})
}

const root = createRoot(output)
const imageSources = []
new MutationObserver(records => {
  for (const record of records) {
    for (const node of record.addedNodes) {
      if (node.nodeName === 'IMG') imageSources.push(node.getAttribute('src'))
      if (node.querySelectorAll) {
        for (const image of node.querySelectorAll('img')) imageSources.push(image.getAttribute('src'))
      }
    }
  }
}).observe(output, { childList: true, subtree: true })

function fileList(names) {
  const transfer = new DataTransfer()
  for (const name of names) {
    transfer.items.add(new File([new Uint8Array([1, 2, 3, 4])], name, { type: 'image/jpeg' }))
  }
  return transfer.files
}

async function run() {
  let assertions = 0
  const check = (condition, message) => { assert(condition, message); assertions += 1 }

  check(proposedNameFromFilename('Hadil_Karous.jpg') === 'Hadil Karous', 'filename proposal')
  check(proposedNameFromFilename('../../evil name.PNG') === 'Evil Name', 'filename proposal strips path')

  root.render(<ProfileManager />)
  await waitFor(() => output.querySelectorAll('[role="tab"]').length === 3, 'three tabs render')
  const tabs = [...output.querySelectorAll('[role="tab"]')].map(node => node.textContent.trim())
  check(
    tabs.join('|') === 'Batch Import|Create Manually|Manage Profiles',
    'tab labels wrong: ' + tabs.join('|'),
  )
  await waitFor(() => requests.some(entry => entry.url.startsWith('/api/profiles?state=')), 'profiles load')

  const folderInput = output.querySelector('[data-testid="batch-folder-input"]')
  check(Boolean(folderInput), 'folder input exists')
  check(folderInput.hasAttribute('webkitdirectory'), 'folder input uses webkitdirectory')
  check(folderInput.multiple === true, 'folder input accepts multiple files')

  folderInput.files = fileList(['khalifa_1.jpg', 'khalifa_2.jpg'])
  folderInput.dispatchEvent(new Event('change', { bubbles: true }))
  await waitFor(() => createdUrls === 2, 'local previews created')
  await waitFor(
    () => requests.some(entry => entry.url === '/api/profiles/import/preview' && entry.method === 'POST'),
    'preview upload posted',
  )
  const upload = requests.find(entry => entry.url === '/api/profiles/import/preview')
  check(upload.form instanceof FormData, 'preview upload uses multipart FormData')
  check(upload.form.getAll('images').length === 2, 'preview upload posts the "images" field')

  await waitFor(() => output.querySelector('[data-testid="batch-review-rows"]'), 'review rows render')
  check(
    output.querySelectorAll('[data-testid="batch-review-rows"] article').length === 2,
    'both grouped and failed identities render',
  )
  check(Boolean(includesText('.manage-facts', 'Grouped photos')), 'grouped_photo_count is shown')
  check(
    Boolean(includesText('.manage-validation', 'exactly one face is required')),
    'failed item keeps its validation_error visible',
  )
  check(
    output.querySelectorAll('.manage-review-card.is-invalid').length === 1,
    'failed item is flagged but not removed',
  )
  check(Boolean(includesText('.manage-progress', 'Processed')), 'progress metrics render')

  const consentSelector = '[data-testid="rename-consent-' + IDENTITY_OK + '"]'
  const consent = output.querySelector(consentSelector)
  check(Boolean(consent), 'rename-consent checkbox exists for attach_existing')
  check(consent.disabled === true, 'rename consent is disabled until the name is edited')

  const confirmButton = byText('button', 'Confirm Batch')
  check(Boolean(confirmButton), 'Confirm Batch button exists')
  confirmButton.click()
  await waitFor(() => requests.some(entry => entry.url.endsWith('/commit')), 'commit posted')
  let commit = requests.find(entry => entry.url.endsWith('/commit'))
  let sent = commit.body.identities.find(row => row.identity_id === IDENTITY_OK)
  check(sent !== undefined, 'commit payload carries identity_id')
  check(sent.update_existing_name === false, 'unedited name must NOT request a rename')
  check(sent.action === 'attach_existing', 'commit payload carries action')
  check(sent.existing_person_id === 'person_001', 'commit payload carries existing_person_id')
  check(sent.primary_source_id === 'a', 'commit payload carries primary_source_id')
  check(
    commit.body.identities.find(row => row.identity_id === IDENTITY_BAD).skip === true,
    'invalid identity is sent as skip',
  )
  await waitFor(() => includesText('.manage-commit-summary', 'Profiles updated'), 'commit summary renders')
  assertions += 1

  const nameInput = [...output.querySelectorAll('label')]
    .find(label => label.textContent.startsWith('Name'))
    .querySelector('input')
  nativeValue.call(nameInput, 'Khalifa B.')
  nameInput.dispatchEvent(new Event('input', { bubbles: true }))
  await waitFor(() => !output.querySelector(consentSelector).disabled, 'consent unlocks after an edit')
  output.querySelector(consentSelector).click()
  await waitFor(() => output.querySelector(consentSelector).checked, 'consent can be ticked')
  const commitsBefore = requests.filter(entry => entry.url.endsWith('/commit')).length
  byText('button', 'Confirm Batch').click()
  await waitFor(
    () => requests.filter(entry => entry.url.endsWith('/commit')).length > commitsBefore,
    'second commit posted',
  )
  commit = requests.filter(entry => entry.url.endsWith('/commit')).at(-1)
  sent = commit.body.identities.find(row => row.identity_id === IDENTITY_OK)
  check(sent.update_existing_name === true, 'edited name + explicit consent requests the rename')
  check(sent.name === 'Khalifa B.', 'edited name is sent')

  const revokedBefore = revokedUrls
  folderInput.files = fileList(['single.jpg'])
  folderInput.dispatchEvent(new Event('change', { bubbles: true }))
  await waitFor(() => revokedUrls >= revokedBefore + 2, 'previous object URLs are revoked on reselection')
  check(revokedUrls >= 2, 'object URLs are revoked, not leaked')

  ;[...output.querySelectorAll('[role="tab"]')][2].click()
  await waitFor(() => output.querySelector('.profiles-sidebar'), 'profiles sidebar renders')
  check(Boolean(output.querySelector('input[type="search"]')), 'search box renders')
  const stateSelect = output.querySelector('.profiles-sidebar select')
  check(
    [...stateSelect.options].map(option => option.value).join(',') === 'active,archived,all',
    'active/archived/all filters render',
  )
  check(
    output.querySelectorAll('.profile-list button').length === 1,
    'the active filter hides the archived profile',
  )
  check(
    imageSources.some(src => src && src.includes('phone_new.jpg')),
    'sidebar renders the backend-resolved effective image',
  )
  check(
    !imageSources.some(src => src && src.includes('stale.jpg')),
    'the stale persons.profile_image is never requested',
  )

  nativeSelectValue.call(stateSelect, 'archived')
  stateSelect.dispatchEvent(new Event('change', { bubbles: true }))
  await waitFor(
    () => [...output.querySelectorAll('.profile-list button')]
      .some(node => node.textContent.includes('Archived Person')),
    'archived filter reveals the archived profile',
  )
  nativeSelectValue.call(stateSelect, 'active')
  stateSelect.dispatchEvent(new Event('change', { bubbles: true }))
  await waitFor(() => output.querySelectorAll('.profile-list button').length === 1, 'active filter restored')

  output.querySelector('.profile-list button').click()
  await waitFor(() => output.querySelector('.profile-detail'), 'profile detail renders')
  check(
    output.querySelector('[data-testid="profile-image-origin"]').textContent.includes('supervisor phone crop'),
    'the resolved image origin is surfaced',
  )
  const titles = [...output.querySelectorAll('.card-title')].map(node => node.textContent)
  const phoneIndex = titles.findIndex(text => text.includes('Phone-photo history'))
  const videoIndex = titles.findIndex(text => text.includes('Video-crop history'))
  check(phoneIndex >= 0 && videoIndex >= 0, 'phone and video history cards render')
  check(phoneIndex < videoIndex, 'phone crops render before video crops')
  check(Boolean(byText('span', 'Primary (supervisor)')), 'supervisor-selected primary is labelled')
  check(
    Boolean(output.querySelector('[data-testid="phone-import-review"]')),
    'durable phone-import reviews appear under uncertain suggestions',
  )
  const resolutionControls = output.querySelector('[data-testid="review-resolution-controls"]')
  check(Boolean(resolutionControls), 'review resolution controls render inline')
  check(Boolean(byText('button', 'Attach to existing')), 'attach-existing review action renders')
  check(Boolean(byText('button', 'Create new')), 'create-new review action renders')
  const resolveSkip = [...resolutionControls.querySelectorAll('button')]
    .find(node => node.textContent.trim() === 'Skip')
  check(Boolean(resolveSkip), 'skip review action renders')
  resolveSkip.click()
  await waitFor(
    () => requests.some(entry => entry.url === '/api/profiles/reviews/review-1/resolve'),
    'review resolution posted',
  )
  const resolutionRequest = requests.find(
    entry => entry.url === '/api/profiles/reviews/review-1/resolve',
  )
  check(resolutionRequest.body.action === 'skip', 'review resolution payload carries action')
  await waitFor(() => includesText('.manage-banner.is-success', 'Review skipped.'), 'review success renders')
  assertions += 1

  const makePrimary = byText('button', 'Make primary')
  check(Boolean(makePrimary), 'non-primary phone photo offers "Make primary"')
  makePrimary.click()
  await waitFor(() => requests.some(entry => entry.url.endsWith('/primary-photo')), 'primary-photo posted')
  check(
    requests.find(entry => entry.url.endsWith('/primary-photo')).body.source_id === 'src-old',
    'primary-photo posts source_id',
  )

  const mergeButton = byText('button', 'Merge into target')
  check(Boolean(mergeButton), 'merge control renders')
  check(mergeButton.disabled === true, 'merge is blocked until target and confirmation are given')
  check(
    Boolean(includesText('label', 'I confirm these profiles represent the same person.')),
    'merge confirmation is an inline control',
  )
  check(blockingDialogs === 0, 'no blocking browser dialog was used')

  root.unmount()
  document.body.dataset.probe = 'pass'
  output.textContent = 'PROFILE_MANAGEMENT_COMPONENT_PROBE_PASS ' + assertions
}

run().catch(error => {
  document.body.dataset.probe = 'fail'
  output.textContent = 'PROFILE_MANAGEMENT_COMPONENT_PROBE_FAIL ' + (error?.stack || error)
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
    '--run-all-compositor-stages-before-draw', '--virtual-time-budget=8000',
    '--dump-dom', pageUrl,
  ], { encoding: 'utf8', timeout: 30000, maxBuffer: 8 * 1024 * 1024 })
  if (result.error) throw result.error
  if (result.status !== 0 || !result.stdout.includes('PROFILE_MANAGEMENT_COMPONENT_PROBE_PASS')) {
    throw new Error(`ProfileManager actual-component probe failed.\n${result.stdout}\n${result.stderr}`)
  }
  const count = result.stdout.match(/PROFILE_MANAGEMENT_COMPONENT_PROBE_PASS (\d+)/)?.[1]
  console.log(`ProfileManager actual-component probe passed (${count} assertions).`)
} finally {
  await rm(temporary, { recursive: true, force: true })
}
