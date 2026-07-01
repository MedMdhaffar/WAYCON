import { useState, useEffect, useCallback } from 'react'

const fieldLabel = { fontSize: 12, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 6 }
const statRow = { display: 'flex', gap: 24, flexWrap: 'wrap' }
const statBox = { background: '#0f1117', border: '1px solid #1e2330', borderRadius: 8, padding: '12px 16px', minWidth: 140 }
const statNum = { fontSize: 22, fontWeight: 600, color: '#e2e8f0' }
const statLabel = { fontSize: 11, color: '#94a3b8', marginTop: 4, textTransform: 'uppercase', letterSpacing: '0.04em' }
const resultPanel = { background: '#0f1117', border: '1px solid #1e2330', borderRadius: 8, padding: 12, fontSize: 13, fontFamily: 'monospace', color: '#cbd5e1', whiteSpace: 'pre-wrap', marginTop: 12 }

export default function ProfileManager() {
  const [profiles, setProfiles] = useState([])
  const [selected, setSelected] = useState('')
  const [detail, setDetail] = useState(null)
  const [loadingDetail, setLoadingDetail] = useState(false)
  const [error, setError] = useState(null)

  const [imagesDir, setImagesDir] = useState('')
  const [replace, setReplace] = useState(false)

  const [cleanupResult, setCleanupResult] = useState(null)
  const [addResult, setAddResult] = useState(null)
  const [busy, setBusy] = useState(false)

  const fetchProfiles = useCallback(async () => {
    setError(null)
    try {
      const res = await fetch('/api/profiles')
      const data = await res.json()
      setProfiles(data.profiles || [])
      if (data.profiles?.length && !selected) {
        setSelected(data.profiles[0].id)
      }
    } catch (e) {
      setError(`failed to load profiles: ${e.message}`)
    }
  }, [selected])

  const fetchDetail = useCallback(async (name) => {
    if (!name) { setDetail(null); return }
    setLoadingDetail(true)
    setError(null)
    try {
      const res = await fetch(`/api/profiles/${encodeURIComponent(name)}`)
      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err.error || `HTTP ${res.status}`)
      }
      setDetail(await res.json())
    } catch (e) {
      setDetail(null)
      setError(`failed to load profile detail: ${e.message}`)
    } finally {
      setLoadingDetail(false)
    }
  }, [])

  useEffect(() => { fetchProfiles() }, [])
  useEffect(() => { fetchDetail(selected) }, [selected, fetchDetail])

  const runCleanup = async (dryRun) => {
    if (!selected || busy) return
    setBusy(true); setError(null)
    try {
      const res = await fetch(`/api/profiles/${encodeURIComponent(selected)}/cleanup`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ dry_run: dryRun }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`)
      setCleanupResult(data)
      if (!dryRun) fetchDetail(selected)
    } catch (e) {
      setError(`cleanup failed: ${e.message}`)
    } finally {
      setBusy(false)
    }
  }

  const runAddPhotos = async (dryRun) => {
    if (!selected || busy) return
    if (!imagesDir.trim()) { setError('images dir is required'); return }
    setBusy(true); setError(null)
    try {
      const res = await fetch(`/api/profiles/${encodeURIComponent(selected)}/add-face-photos`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ images_dir: imagesDir.trim(), replace, dry_run: dryRun }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`)
      setAddResult(data)
      if (!dryRun) fetchDetail(selected)
    } catch (e) {
      setError(`add-face-photos failed: ${e.message}`)
    } finally {
      setBusy(false)
    }
  }

  return (
    <>
      <div className="card">
        <div className="card-title">Profile</div>
        <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
          <select
            value={selected}
            onChange={(e) => { setSelected(e.target.value); setCleanupResult(null); setAddResult(null) }}
            style={{
              flex: 1, padding: '9px 12px', background: '#0f1117',
              border: '1px solid #1e2330', borderRadius: 6, color: '#e2e8f0', fontSize: 14,
            }}
          >
            {profiles.length === 0 && <option value="">(no profiles)</option>}
            {profiles.map(p => (
              <option key={p.id} value={p.id}>
                {p.id} — {p.name} · face {p.face_crop_count} · body {p.body_crop_count}
              </option>
            ))}
          </select>
          <button className="btn btn-ghost" onClick={() => { fetchProfiles(); fetchDetail(selected) }} disabled={busy}>
            Refresh
          </button>
        </div>
        {error && <div style={{ color: '#ef4444', fontSize: 13, marginTop: 12 }}>{error}</div>}
      </div>

      {detail && (
        <div className="card">
          <div className="card-title">Stats — {detail.name}</div>
          <div style={statRow}>
            <div style={statBox}>
              <div style={statNum}>{detail.face_crops_referenced}</div>
              <div style={statLabel}>face crops (referenced)</div>
              <div style={{ ...statLabel, color: '#64748b', textTransform: 'none' }}>
                {detail.face_crops_on_disk} on disk · {detail.orphan_face} orphan
              </div>
            </div>
            <div style={statBox}>
              <div style={statNum}>{detail.body_crops_referenced}</div>
              <div style={statLabel}>body crops (referenced)</div>
              <div style={{ ...statLabel, color: '#64748b', textTransform: 'none' }}>
                {detail.body_crops_on_disk} on disk · {detail.orphan_body} orphan
              </div>
            </div>
            <div style={statBox}>
              <div style={statNum}>{detail.best_body_crops}</div>
              <div style={statLabel}>best body crops</div>
            </div>
            <div style={statBox}>
              <div style={statNum}>{detail.has_iphone_photos ? 'yes' : 'no'}</div>
              <div style={statLabel}>iphone photos</div>
            </div>
          </div>
        </div>
      )}

      {detail && (
        <div className="card">
          <div className="card-title">Cleanup orphan crops</div>
          <p style={{ fontSize: 13, color: '#94a3b8', marginBottom: 12 }}>
            Removes files in <code>body_crops/</code> and <code>face_crops/</code> that aren't referenced in profile.json. Preview first to see how many would be deleted.
          </p>
          <div style={{ display: 'flex', gap: 8 }}>
            <button className="btn btn-ghost" onClick={() => runCleanup(true)} disabled={busy}>Preview (dry-run)</button>
            <button className="btn btn-danger" onClick={() => runCleanup(false)} disabled={busy || (detail.orphan_face + detail.orphan_body === 0)}>Apply</button>
          </div>
          {cleanupResult && (
            <pre style={resultPanel}>{JSON.stringify(cleanupResult, null, 2)}</pre>
          )}
        </div>
      )}

      {detail && (
        <div className="card">
          <div className="card-title">Add pro-cam face photos</div>
          <p style={{ fontSize: 13, color: '#94a3b8', marginBottom: 12 }}>
            Point at a server-side folder of high-quality face photos (iPhone, DSLR). For each image the largest detected face is cropped, saved to <code>face_crops/</code>, and the profile's <code>face_embedding</code> is recomputed from all current crops.
          </p>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            <div>
              <div style={fieldLabel}>Images dir (on server)</div>
              <input
                type="text"
                value={imagesDir}
                onChange={(e) => setImagesDir(e.target.value)}
                placeholder="/mnt/c/Users/malek/Desktop/iphone_face/"
                style={{
                  width: '100%', padding: '9px 12px', background: '#0f1117',
                  border: '1px solid #1e2330', borderRadius: 6, color: '#e2e8f0', fontSize: 13,
                  fontFamily: 'monospace',
                }}
                disabled={busy}
              />
            </div>
            <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 13, color: '#cbd5e1' }}>
              <input type="checkbox" checked={replace} onChange={(e) => setReplace(e.target.checked)} disabled={busy} />
              Replace existing face crops (drops everything not prefixed <code>iphone_</code>)
            </label>
            <div style={{ display: 'flex', gap: 8 }}>
              <button className="btn btn-ghost" onClick={() => runAddPhotos(true)} disabled={busy || !imagesDir.trim()}>Preview (dry-run)</button>
              <button className="btn btn-primary" onClick={() => runAddPhotos(false)} disabled={busy || !imagesDir.trim()}>Apply</button>
            </div>
          </div>
          {addResult && (
            <pre style={resultPanel}>{JSON.stringify(addResult, null, 2)}</pre>
          )}
        </div>
      )}

      {!detail && !loadingDetail && (
        <div className="card" style={{ color: '#64748b', fontSize: 14 }}>
          Select a profile above to manage it.
        </div>
      )}
    </>
  )
}
