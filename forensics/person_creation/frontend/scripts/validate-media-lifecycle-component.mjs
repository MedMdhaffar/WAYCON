import { spawnSync } from 'node:child_process'
import { existsSync } from 'node:fs'
import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import process from 'node:process'

import { build } from 'esbuild'

const frontendRoot = path.resolve(import.meta.dirname, '..')
const componentPath = path.join(frontendRoot, 'src', 'components', 'SafeImage.jsx')
const liveJobPath = path.join(frontendRoot, 'src', 'liveJob.js')
const temporary = await mkdtemp(path.join(os.tmpdir(), 'waycon-media-lifecycle-'))
const entryPath = path.join(temporary, 'probe.jsx')
const htmlPath = path.join(temporary, 'probe.html')
const bundlePath = path.join(temporary, 'probe.js')

const entry = String.raw`
import React from 'react'
import { createRoot } from 'react-dom/client'
import SafeImage, { clearFailedMediaUrls } from ${JSON.stringify(componentPath.replaceAll('\\', '/'))}
import { mediaImageUrl } from ${JSON.stringify(liveJobPath.replaceAll('\\', '/'))}

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
    if (performance.now() - started > 2000) throw new Error('Timed out: ' + message)
    await wait(10)
  }
}
const assert = (condition, message) => { if (!condition) throw new Error(message) }
const render = (path, version, paths = []) => root.render(
  <SafeImage path={path} paths={paths} version={version} alt="crop" placeholder="stable-placeholder" />,
)

async function run() {
  clearFailedMediaUrls()
  assert(mediaImageUrl('/mnt/c/private.jpg') === '', 'absolute POSIX path was accepted')
  assert(mediaImageUrl('C:\\private\\face.jpg') === '', 'Windows path was accepted')
  assert(mediaImageUrl('file:///private.jpg') === '', 'file URL was accepted')
  assert(mediaImageUrl('../private.jpg') === '', 'traversal path was accepted')

  render('malek/_staging/face.jpg', 1)
  await waitFor(() => imageMounts === 1, 'first staging image mount')
  await waitFor(() => output.textContent.includes('stable-placeholder'), 'staging failure placeholder')
  for (let index = 0; index < 10; index += 1) {
    render('malek/_staging/face.jpg', 1)
    await wait(0)
  }
  assert(imageMounts === 1, 'unchanged failed URL was retried during polling rerenders')

  render('malek/cluster_0/face.jpg', 2)
  await waitFor(() => imageMounts === 2, 'promoted image mount')
  await waitFor(() => output.textContent.includes('stable-placeholder'), 'promoted failure placeholder')
  render('malek/cluster_0/face.jpg', 2, ['malek/cluster_0/face.jpg'])
  await wait(20)
  assert(imageMounts === 2, 'duplicate equivalent candidate caused another request')

  render('person_016/face_crops/face.jpg', 3)
  await waitFor(() => imageMounts === 3, 'new canonical path mount')
  assert(imageMounts === 3, 'changed canonical version was not requestable immediately')
  await waitFor(() => output.textContent.includes('stable-placeholder'), 'canonical failure placeholder')

  render('/mnt/c/private/face.jpg', 4)
  await wait(20)
  assert(imageMounts === 3, 'absolute candidate created an image request')
  assert(output.textContent.includes('stable-placeholder'), 'invalid candidate lacked stable placeholder')

  observer.disconnect()
  root.unmount()
  document.body.dataset.probe = 'pass'
  output.textContent = 'MEDIA_LIFECYCLE_COMPONENT_PROBE_PASS 9'
}

run().catch(error => {
  document.body.dataset.probe = 'fail'
  output.textContent = 'MEDIA_LIFECYCLE_COMPONENT_PROBE_FAIL ' + (error?.stack || error)
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
    '--run-all-compositor-stages-before-draw', '--virtual-time-budget=4000',
    '--dump-dom', pageUrl,
  ], { encoding: 'utf8', timeout: 15000, maxBuffer: 4 * 1024 * 1024 })
  if (result.error) throw result.error
  if (result.status !== 0 || !result.stdout.includes('MEDIA_LIFECYCLE_COMPONENT_PROBE_PASS')) {
    throw new Error(`Actual media component probe failed.\n${result.stdout}\n${result.stderr}`)
  }
  console.log('SafeImage actual media-lifecycle probe passed (9 assertions).')
} finally {
  await rm(temporary, { recursive: true, force: true })
}
