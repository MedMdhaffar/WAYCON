import { useCallback, useEffect, useState } from 'react'

const inputStyle = {
  width: '100%',
  padding: '9px 12px',
  background: '#0f1117',
  border: '1px solid #1e2330',
  borderRadius: 6,
  color: '#e2e8f0',
  fontSize: 14,
}

const labelStyle = {
  fontSize: 12,
  color: '#94a3b8',
  textTransform: 'uppercase',
  letterSpacing: '0.05em',
  marginBottom: 6,
  display: 'block',
}

const mutedText = { fontSize: 13, color: '#94a3b8', lineHeight: 1.5 }
const panelStyle = { background: '#0f1117', border: '1px solid #1e2330', borderRadius: 8, padding: 12 }

function sourceLabel(source) {
  if (source === 'phone_photo') return 'phone_photo - profile photos only'
  if (source === 'video_profile') return 'video_profile - video enrollment only'
  if (source === 'mixed') return 'mixed - phone photos + video enrollment'
  return source || 'unknown'
}

async function apiJson(path, options) {
  const res = await fetch(path, options)
  const text = await res.text()
  let data = null
  try {
    data = text ? JSON.parse(text) : {}
  } catch (_) {
    const hint = text.trim().startsWith('<')
      ? 'received HTML instead of JSON; make sure the Flask backend is running on port 5009 and restart it after code changes'
      : 'received a non-JSON response'
    throw new Error(`${hint} (${res.status})`)
  }
  if (!res.ok || data.ok === false) throw new Error(data.error || `HTTP ${res.status}`)
  return data
}

function Message({ message }) {
  if (!message) return null
  const color = message.type === 'error' ? '#ef4444' : '#22c55e'
  const background = message.type === 'error' ? '#1e1015' : '#0f1d14'
  return (
    <div style={{ color, background, borderRadius: 6, padding: '9px 12px', fontSize: 13, marginTop: 12 }}>
      {message.text}
    </div>
  )
}

function imageUrl(path) {
  return path ? `/api/images?path=${encodeURIComponent(path)}` : ''
}

function Thumbnail({ path, size = 52 }) {
  const [failed, setFailed] = useState(false)
  if (!path || failed) {
    return (
      <div style={{
        width: size, height: size, borderRadius: 6, background: '#1e2330',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        color: '#64748b', fontSize: 11,
      }}>
        no image
      </div>
    )
  }
  return (
    <img
      src={imageUrl(path)}
      alt=""
      onError={() => setFailed(true)}
      style={{
        width: size, height: size, objectFit: 'cover', borderRadius: 6,
        border: '1px solid #1e2330', background: '#0f1117',
      }}
    />
  )
}

function ImageGallery({ title, items, pathKey = 'media_path', totalCount = null, missingCount = 0 }) {
  const list = items || []
  const total = totalCount ?? list.length
  return (
    <div style={panelStyle}>
      <div style={labelStyle}>{title} ({list.length})</div>
      {missingCount > 0 && (
        <div style={{ color: '#f59e0b', fontSize: 12, marginBottom: 8 }}>
          {total} saved references, {list.length} files found ({missingCount} missing hidden)
        </div>
      )}
      {list.length === 0 ? (
        <div style={mutedText}>No valid images found</div>
      ) : (
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          {list.slice(0, 24).map((item, idx) => {
            const path = item[pathKey] || item.face_crop_path || item.image_path
            return <Thumbnail key={`${path || idx}-${idx}`} path={path} size={72} />
          })}
        </div>
      )}
    </div>
  )
}

export default function GlobalMemory() {
  const [persons, setPersons] = useState([])
  const [suggestions, setSuggestions] = useState([])
  const [dbPath, setDbPath] = useState('')
  const [thresholds, setThresholds] = useState({ review: 0.45, auto: 0.65 })
  const [selectedId, setSelectedId] = useState('')
  const [detail, setDetail] = useState(null)
  const [message, setMessage] = useState(null)
  const [busy, setBusy] = useState(false)

  const [name, setName] = useState('')
  const [notes, setNotes] = useState('')
  const [photos, setPhotos] = useState([])

  const [editName, setEditName] = useState('')
  const [editNotes, setEditNotes] = useState('')
  const [mergeTargetId, setMergeTargetId] = useState('')
  const [mergeName, setMergeName] = useState('')
  const [detailPhotos, setDetailPhotos] = useState([])

  const [profilePath, setProfilePath] = useState('')
  const [compareResult, setCompareResult] = useState(null)

  const fetchPersons = useCallback(async () => {
    try {
      const data = await apiJson('/api/global-memory/persons')
      setPersons(data.persons || [])
      setDbPath(data.db_path || '')
      setThresholds({
        review: data.review_threshold ?? 0.45,
        auto: data.auto_match_threshold ?? 0.65,
      })
      if (!selectedId && data.persons?.length) setSelectedId(data.persons[0].person_id)
    } catch (err) {
      setMessage({ type: 'error', text: `failed to load Global Memory: ${err.message}` })
    }
  }, [selectedId])

  const fetchSuggestions = useCallback(async () => {
    try {
      const data = await apiJson('/api/global-memory/suggestions')
      setSuggestions(data.suggestions || [])
      setThresholds({
        review: data.review_threshold ?? 0.45,
        auto: data.auto_match_threshold ?? 0.65,
      })
    } catch (err) {
      setMessage({ type: 'error', text: `failed to load match suggestions: ${err.message}` })
    }
  }, [])

  const fetchDetail = useCallback(async (personId) => {
    if (!personId) { setDetail(null); return }
    try {
      const data = await apiJson(`/api/global-memory/persons/${encodeURIComponent(personId)}`)
      setDetail(data.person)
      setEditName(data.person?.person?.name || '')
      setEditNotes(data.person?.person?.notes || '')
    } catch (err) {
      setDetail(null)
      setMessage({ type: 'error', text: `failed to load person detail: ${err.message}` })
    }
  }, [])

  const refreshAll = useCallback(async () => {
    await fetchPersons()
    await fetchSuggestions()
    if (selectedId) await fetchDetail(selectedId)
  }, [fetchPersons, fetchSuggestions, fetchDetail, selectedId])

  useEffect(() => { refreshAll() }, [])
  useEffect(() => { fetchDetail(selectedId) }, [selectedId, fetchDetail])

  const registerPhotos = async (event) => {
    event.preventDefault()
    setMessage(null)
    setCompareResult(null)
    if (!name.trim()) {
      setMessage({ type: 'error', text: 'name is required' })
      return
    }
    if (!photos.length) {
      setMessage({ type: 'error', text: 'upload at least one clear face photo' })
      return
    }

    const form = new FormData()
    form.append('name', name.trim())
    form.append('notes', notes.trim())
    photos.forEach((photo) => form.append('photos', photo))

    setBusy(true)
    try {
      const data = await apiJson('/api/global-memory/register-face-photos', {
        method: 'POST',
        body: form,
      })
      const duplicate = data.possible_duplicate && data.best_match
        ? ` Possible duplicate: ${data.name} looks similar to ${data.best_match.name}, similarity ${Number(data.best_match.similarity).toFixed(4)}. Created a separate identity; merge manually if this is the same person.`
        : ''
      setMessage({ type: 'success', text: `${data.message}: ${data.name} (${data.identity_source}).${duplicate}` })
      setName('')
      setNotes('')
      setPhotos([])
      setSelectedId(data.person_id)
      await refreshAll()
    } catch (err) {
      setMessage({ type: 'error', text: err.message })
    } finally {
      setBusy(false)
    }
  }

  const savePerson = async () => {
    if (!selectedId) return
    setBusy(true)
    setMessage(null)
    try {
      await apiJson(`/api/global-memory/persons/${encodeURIComponent(selectedId)}`, {
        method: 'PATCH',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ name: editName, notes: editNotes }),
      })
      setMessage({ type: 'success', text: 'person details updated' })
      await refreshAll()
    } catch (err) {
      setMessage({ type: 'error', text: `update failed: ${err.message}` })
    } finally {
      setBusy(false)
    }
  }

  const mergeSelected = async () => {
    if (!selectedId || !mergeTargetId) return
    setBusy(true)
    setMessage(null)
    try {
      await apiJson('/api/global-memory/persons/merge', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          source_person_id: selectedId,
          target_person_id: mergeTargetId,
          new_name: mergeName,
        }),
      })
      setMessage({ type: 'success', text: 'duplicate merged into selected target identity' })
      setSelectedId(mergeTargetId)
      setMergeTargetId('')
      setMergeName('')
      await refreshAll()
    } catch (err) {
      setMessage({ type: 'error', text: `merge failed: ${err.message}` })
    } finally {
      setBusy(false)
    }
  }

  const resolveSuggestion = async (suggestionId, action) => {
    setBusy(true)
    setMessage(null)
    try {
      await apiJson(`/api/global-memory/suggestions/${encodeURIComponent(suggestionId)}/${action}`, {
        method: 'POST',
      })
      setMessage({ type: 'success', text: action === 'accept' ? 'suggestion accepted and merged' : 'suggestion rejected' })
      await refreshAll()
    } catch (err) {
      setMessage({ type: 'error', text: `suggestion ${action} failed: ${err.message}` })
    } finally {
      setBusy(false)
    }
  }

  const addPhotosToSelected = async (event) => {
    event.preventDefault()
    if (!selectedId) return
    if (!detailPhotos.length) {
      setMessage({ type: 'error', text: 'choose at least one photo to add' })
      return
    }

    const form = new FormData()
    detailPhotos.forEach((photo) => form.append('photos', photo))

    setBusy(true)
    setMessage(null)
    try {
      const data = await apiJson(`/api/global-memory/persons/${encodeURIComponent(selectedId)}/add-face-photos`, {
        method: 'POST',
        body: form,
      })
      setMessage({ type: 'success', text: `${data.message}: ${data.name} (${data.identity_source})` })
      setDetailPhotos([])
      await refreshAll()
    } catch (err) {
      setMessage({ type: 'error', text: `add photos failed: ${err.message}` })
    } finally {
      setBusy(false)
    }
  }

  const compareProfile = async (event) => {
    event.preventDefault()
    setMessage(null)
    setCompareResult(null)
    if (!profilePath.trim()) {
      setMessage({ type: 'error', text: 'profile path is required' })
      return
    }

    setBusy(true)
    try {
      const data = await apiJson('/api/global-memory/compare-profile', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ profile_path: profilePath.trim() }),
      })
      setCompareResult(data)
    } catch (err) {
      setMessage({ type: 'error', text: `compare failed: ${err.message}` })
    } finally {
      setBusy(false)
    }
  }

  const selectedPerson = detail?.person || null
  const mergeTargets = persons.filter((p) => p.person_id !== selectedId)

  return (
    <>
      <div className="card">
        <div className="card-title">Global Memory</div>
        <p style={mutedText}>
          Clear front-facing profile photos create the permanent face-embedding identity anchor.
          Clothing and body descriptions are daily appearance signals added later from approved video enrollment.
        </p>
        <p style={{ ...mutedText, marginTop: 8 }}>
          Face match bands: below {thresholds.review} = no match; {thresholds.review} to {thresholds.auto} = supervisor review; above {thresholds.auto} = automatic match.
        </p>
        {dbPath && <p style={{ ...mutedText, marginTop: 8 }}>Database: <code>{dbPath}</code></p>}
        <Message message={message} />
      </div>

      {suggestions.length > 0 && (
        <div className="card">
          <div className="card-title">Match suggestions</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {suggestions.map((s) => (
              <div key={s.id} style={{ ...panelStyle, display: 'flex', gap: 12, justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap' }}>
                <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
                  <Thumbnail path={s.new_profile_image_path} size={54} />
                  <Thumbnail path={s.candidate_profile_image_path} size={54} />
                </div>
                <div style={{ flex: '1 1 360px' }}>
                  <strong>{s.new_person_name}</strong> may be <strong>{s.candidate_person_name}</strong>
                  <span style={{ color: '#94a3b8' }}> - similarity {Number(s.similarity).toFixed(4)} / auto {s.auto_match_threshold}</span>
                  <div style={{ ...mutedText, marginTop: 4 }}>
                    Source duplicate: <code>{s.new_person_id}</code> | Target candidate: <code>{s.candidate_person_id}</code>
                  </div>
                </div>
                <div style={{ display: 'flex', gap: 8 }}>
                  <button className="btn btn-success" disabled={busy} onClick={() => resolveSuggestion(s.id, 'accept')}>Accept merge</button>
                  <button className="btn btn-ghost" disabled={busy} onClick={() => resolveSuggestion(s.id, 'reject')}>Reject</button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="card">
        <div className="card-title">Register from profile photos</div>
        <form onSubmit={registerPhotos} style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div>
            <label style={labelStyle}>Name</label>
            <input style={inputStyle} value={name} onChange={(e) => setName(e.target.value)} placeholder="Malek" disabled={busy} />
          </div>
          <div>
            <label style={labelStyle}>Notes</label>
            <input style={inputStyle} value={notes} onChange={(e) => setNotes(e.target.value)} placeholder="Optional details for this upload" disabled={busy} />
          </div>
          <div>
            <label style={labelStyle}>Photos</label>
            <input
              style={inputStyle}
              type="file"
              accept=".jpg,.jpeg,.png,.webp,image/jpeg,image/png,image/webp"
              multiple
              onChange={(e) => setPhotos(Array.from(e.target.files || []))}
              disabled={busy}
            />
            <div style={{ ...mutedText, marginTop: 6 }}>
              Use clear face photos. The backend will detect the largest face and embed it with the existing FaceNet model.
            </div>
          </div>
          <button className="btn btn-primary" type="submit" disabled={busy} style={{ alignSelf: 'flex-start' }}>
            {busy ? 'Registering...' : 'Register Photos'}
          </button>
        </form>
      </div>

      <div className="card">
        <div className="card-title">Persons</div>
        <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, alignItems: 'center', marginBottom: 12 }}>
          <span style={mutedText}>{persons.length} active identities in Global Memory</span>
          <button className="btn btn-ghost" onClick={refreshAll} disabled={busy}>Refresh</button>
        </div>
        {persons.length === 0 ? (
          <div style={panelStyle}>No Global Memory persons yet.</div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ color: '#94a3b8', textAlign: 'left' }}>
                  <th style={{ padding: 8, borderBottom: '1px solid #1e2330' }}>Name</th>
                  <th style={{ padding: 8, borderBottom: '1px solid #1e2330' }}>Person ID</th>
                  <th style={{ padding: 8, borderBottom: '1px solid #1e2330' }}>Source</th>
                  <th style={{ padding: 8, borderBottom: '1px solid #1e2330' }}>Appearance</th>
                  <th style={{ padding: 8, borderBottom: '1px solid #1e2330' }}>Runs</th>
                  <th style={{ padding: 8, borderBottom: '1px solid #1e2330' }}>Photos</th>
                  <th style={{ padding: 8, borderBottom: '1px solid #1e2330' }}>Video faces</th>
                  <th style={{ padding: 8, borderBottom: '1px solid #1e2330' }}>Suggestions</th>
                  <th style={{ padding: 8, borderBottom: '1px solid #1e2330' }}>Updated</th>
                  <th style={{ padding: 8, borderBottom: '1px solid #1e2330' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {persons.map((p) => (
                  <tr
                    key={p.person_id}
                    style={{ background: selectedId === p.person_id ? '#1e2330' : 'transparent' }}
                  >
                    <td style={{ padding: 8 }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <Thumbnail path={p.profile_image_path} size={42} />
                        <div>
                          <div>{p.name}</div>
                          {p.notes && <div style={{ ...mutedText, fontSize: 12 }}>{p.notes}</div>}
                        </div>
                      </div>
                    </td>
                    <td style={{ padding: 8, fontFamily: 'monospace', color: '#cbd5e1' }}>{p.person_id}</td>
                    <td style={{ padding: 8 }}>{sourceLabel(p.identity_source)}</td>
                    <td style={{ padding: 8 }}>{p.appearance_count}</td>
                    <td style={{ padding: 8 }}>{p.profile_run_count}</td>
                    <td style={{ padding: 8 }}>{p.face_photo_count}</td>
                    <td style={{ padding: 8 }}>{p.video_face_crop_count || 0}</td>
                    <td style={{ padding: 8 }}>{p.pending_suggestion_count || 0}</td>
                    <td style={{ padding: 8, color: '#94a3b8' }}>{p.updated_at}</td>
                    <td style={{ padding: 8 }}>
                      <button className="btn btn-ghost" style={{ padding: '6px 10px' }} onClick={() => setSelectedId(p.person_id)}>
                        View / Edit
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {selectedPerson && (
        <div className="card">
          <div className="card-title">Person detail</div>
          <div style={{ display: 'grid', gridTemplateColumns: '140px repeat(auto-fit, minmax(180px, 1fr))', gap: 12, alignItems: 'stretch' }}>
            <div style={panelStyle}>
              <div style={labelStyle}>Profile picture</div>
              <Thumbnail path={detail.profile_image_path || selectedPerson.profile_image_path} size={104} />
            </div>
            <div style={panelStyle}><div style={labelStyle}>Name</div>{selectedPerson.name}</div>
            <div style={panelStyle}><div style={labelStyle}>Person ID</div><code>{selectedPerson.person_id}</code></div>
            <div style={panelStyle}><div style={labelStyle}>Source</div>{sourceLabel(selectedPerson.identity_source)}</div>
            <div style={panelStyle}><div style={labelStyle}>Face embedding</div>{selectedPerson.face_embedding_dim || 0} dimensions</div>
          </div>

          <div style={{ ...panelStyle, marginTop: 12 }}>
            <div style={labelStyle}>Edit details</div>
            <div style={{ display: 'grid', gridTemplateColumns: 'minmax(180px, 1fr) minmax(180px, 2fr) auto', gap: 8, alignItems: 'end' }}>
              <div>
                <label style={labelStyle}>Name</label>
                <input style={inputStyle} value={editName} onChange={(e) => setEditName(e.target.value)} disabled={busy} />
              </div>
              <div>
                <label style={labelStyle}>Notes</label>
                <input style={inputStyle} value={editNotes} onChange={(e) => setEditNotes(e.target.value)} disabled={busy} />
              </div>
              <button className="btn btn-primary" disabled={busy} onClick={savePerson}>Save</button>
            </div>
          </div>

          {mergeTargets.length > 0 && (
            <div style={{ ...panelStyle, marginTop: 12 }}>
              <div style={labelStyle}>Merge / link duplicate</div>
              <p style={{ ...mutedText, marginBottom: 10 }}>
                This will move video appearances, runs, photo sources, and crop references from the current source identity into the selected target identity.
              </p>
              <div style={{ display: 'grid', gridTemplateColumns: 'minmax(220px, 1fr) minmax(180px, 1fr) auto', gap: 8, alignItems: 'end' }}>
                <div>
                  <label style={labelStyle}>Target identity</label>
                  <select style={inputStyle} value={mergeTargetId} onChange={(e) => setMergeTargetId(e.target.value)} disabled={busy}>
                    <option value="">Choose target...</option>
                    {mergeTargets.map((p) => (
                      <option key={p.person_id} value={p.person_id}>{p.name} - {p.person_id}</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label style={labelStyle}>Final name</label>
                  <input style={inputStyle} value={mergeName} onChange={(e) => setMergeName(e.target.value)} placeholder="optional" disabled={busy} />
                </div>
                <button className="btn btn-danger" disabled={busy || !mergeTargetId} onClick={mergeSelected}>
                  Merge duplicate into selected person
                </button>
              </div>
            </div>
          )}

          <div style={{ ...panelStyle, marginTop: 12 }}>
            <div style={labelStyle}>Add photos to this person</div>
            <p style={{ ...mutedText, marginBottom: 10 }}>
              This is the explicit update path for adding more phone photos to the selected identity.
            </p>
            <form onSubmit={addPhotosToSelected} style={{ display: 'flex', gap: 8, alignItems: 'end', flexWrap: 'wrap' }}>
              <div style={{ flex: '1 1 320px' }}>
                <label style={labelStyle}>Photos</label>
                <input
                  style={inputStyle}
                  type="file"
                  accept=".jpg,.jpeg,.png,.webp,image/jpeg,image/png,image/webp"
                  multiple
                  onChange={(e) => setDetailPhotos(Array.from(e.target.files || []))}
                  disabled={busy}
                />
              </div>
              <button className="btn btn-primary" type="submit" disabled={busy || !detailPhotos.length}>
                Add photos
              </button>
            </form>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 12, marginTop: 12 }}>
            <div style={panelStyle}>
              <div style={labelStyle}>Appearances</div>
              {(detail.appearances || []).length === 0 ? 'none' : (
                <ul style={{ paddingLeft: 18, lineHeight: 1.6 }}>
                  {detail.appearances.map((a) => (
                    <li key={a.id}>{a.date || 'no date'}: {a.top || 'unknown'} / {a.bottom || 'unknown'} / {a.shoes || 'unknown'}</li>
                  ))}
                </ul>
              )}
            </div>
            <div style={panelStyle}>
              <div style={labelStyle}>Profile runs</div>
              {(detail.profile_runs || []).length === 0 ? 'none' : (
                <ul style={{ paddingLeft: 18, lineHeight: 1.6 }}>
                  {detail.profile_runs.map((r) => (
                    <li key={r.id}>cluster {r.cluster_id ?? '-'}: {r.profile_path || r.output_dir || r.created_at}</li>
                  ))}
                </ul>
              )}
            </div>
            <div style={panelStyle}>
              <div style={labelStyle}>Sources and crops</div>
              <div>phone photos: {(detail.face_photo_sources || []).length}</div>
              <div>video face crops: {detail.valid_video_face_crop_count ?? (detail.video_face_crops || []).length} / {detail.total_video_face_crop_count ?? (detail.video_face_crops || []).length}</div>
              <div>body crops: {detail.valid_body_crop_count ?? (detail.body_crops || []).length} / {detail.total_body_crop_count ?? (detail.body_crops || []).length}</div>
              <div>best body crops: {detail.valid_best_body_crop_count ?? (detail.best_body_crops || []).length} / {detail.total_best_body_crop_count ?? (detail.best_body_crops || []).length}</div>
            </div>
          </div>

          {(detail.pending_suggestions || []).length > 0 && (
            <div style={{ ...panelStyle, marginTop: 12 }}>
              <div style={labelStyle}>Suggestions involving this person</div>
              <ul style={{ paddingLeft: 18, lineHeight: 1.6 }}>
                {detail.pending_suggestions.map((s) => (
                  <li key={s.id}>{s.new_person_name} may be {s.candidate_person_name} - similarity {Number(s.similarity).toFixed(4)}</li>
                ))}
              </ul>
            </div>
          )}

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 12, marginTop: 12 }}>
            <ImageGallery title="Phone/profile photos" items={detail.face_photo_sources || []} />
            <ImageGallery
              title="Camera/video face crops"
              items={detail.video_face_crops || []}
              totalCount={detail.total_video_face_crop_count}
              missingCount={detail.missing_video_face_crop_count || 0}
            />
            <ImageGallery
              title="Body crops"
              items={detail.body_crops || []}
              totalCount={detail.total_body_crop_count}
              missingCount={detail.missing_body_crop_count || 0}
            />
            <ImageGallery
              title="Best body crops"
              items={detail.best_body_crops || []}
              totalCount={detail.total_best_body_crop_count}
              missingCount={detail.missing_best_body_crop_count || 0}
            />
          </div>
        </div>
      )}

      <div className="card">
        <div className="card-title">Compare profile.json to memory</div>
        <form onSubmit={compareProfile} style={{ display: 'flex', gap: 8, alignItems: 'flex-end', flexWrap: 'wrap' }}>
          <div style={{ flex: '1 1 360px' }}>
            <label style={labelStyle}>Profile path</label>
            <input
              style={{ ...inputStyle, fontFamily: 'monospace' }}
              value={profilePath}
              onChange={(e) => setProfilePath(e.target.value)}
              placeholder="forensics/person_db/session/cluster_0/profile.json"
              disabled={busy}
            />
          </div>
          <button className="btn btn-ghost" type="submit" disabled={busy}>Compare</button>
        </form>
        {compareResult && (
          <div style={{ ...panelStyle, marginTop: 12 }}>
            <div style={{ ...mutedText, marginBottom: 8 }}>
              Review: {compareResult.review_threshold} | Auto: {compareResult.auto_match_threshold} | Best match auto-passes: {compareResult.best_match_passes_threshold ? 'yes' : 'no'}
            </div>
            {(compareResult.matches || []).length === 0 ? 'No matches.' : (
              <ol style={{ paddingLeft: 20, lineHeight: 1.7 }}>
                {compareResult.matches.map((m) => (
                  <li key={m.person_id}>
                    {m.name} | <code>{m.person_id}</code> | {m.identity_source} | {Number(m.similarity).toFixed(4)} | {m.passes_threshold ? 'PASS' : 'FAIL'}
                  </li>
                ))}
              </ol>
            )}
          </div>
        )}
      </div>
    </>
  )
}
