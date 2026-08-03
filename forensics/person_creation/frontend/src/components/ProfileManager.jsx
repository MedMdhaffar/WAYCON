import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import SafeImage from './SafeImage.jsx'

const MANAGEMENT_TABS = ['Batch Import', 'Create Manually', 'Manage Profiles']

async function api(path, options = {}) {
  const response = await fetch(path, options)
  const data = await response.json().catch(() => ({}))
  if (!response.ok) throw new Error(data.error || `Request failed (HTTP ${response.status})`)
  return data
}

export function proposedNameFromFilename(filename) {
  const stem = String(filename || '').replaceAll('\\', '/').split('/').pop().replace(/\.[^.]+$/, '')
  return stem.split(/[\s_-]+/).filter(Boolean)
    .map(word => word.slice(0, 1).toUpperCase() + word.slice(1).toLowerCase()).join(' ') || 'Unnamed'
}

function Metric({ value, label }) {
  return <div className="manage-metric"><strong>{value}</strong><span>{label}</span></div>
}

function ErrorBanner({ children }) {
  return children ? <div className="manage-banner is-error">{children}</div> : null
}

function BatchImport({ profiles }) {
  const [localRows, setLocalRows] = useState([])
  const [batch, setBatch] = useState(null)
  const [edits, setEdits] = useState({})
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [commitResult, setCommitResult] = useState(null)
  const pollRef = useRef(null)

  const objectUrlsRef = useRef([])

  const stopPolling = () => {
    if (pollRef.current) window.clearInterval(pollRef.current)
    pollRef.current = null
  }
  const releaseObjectUrls = () => {
    objectUrlsRef.current.forEach(url => URL.revokeObjectURL(url))
    objectUrlsRef.current = []
  }
  useEffect(() => () => { stopPolling(); releaseObjectUrls() }, [])

  const mergeBatch = useCallback((next) => {
    setBatch(next)
    if (next.identities?.length) {
      setEdits(current => {
        const copy = { ...current }
        next.identities.forEach(identity => {
          copy[identity.identity_id] ??= {
            name: identity.proposed_name,
            // Renaming an existing profile needs two explicit signals: the
            // supervisor edited the field AND ticked the rename box.
            name_edited: false,
            update_existing_name: false,
            action: identity.proposed_action,
            existing_person_id: identity.existing_candidate?.person_id || '',
            primary_source_id: identity.primary_source_id,
            skip: identity.proposed_action === 'skip',
          }
        })
        return copy
      })
    }
    if (next.state !== 'processing') stopPolling()
  }, [])

  const poll = useCallback(async (batchId) => {
    try {
      mergeBatch(await api(`/api/profiles/import/${encodeURIComponent(batchId)}/status`))
    } catch (reason) {
      setError(reason.message)
      stopPolling()
    }
  }, [mergeBatch])

  const selectFolder = async (event) => {
    const files = Array.from(event.target.files || [])
    stopPolling()
    releaseObjectUrls()
    setBatch(null)
    setCommitResult(null)
    setError('')
    setEdits({})
    setLocalRows(files.map((file, index) => {
      const preview = URL.createObjectURL(file)
      objectUrlsRef.current.push(preview)
      return {
        key: `${file.name}-${file.lastModified}-${index}`,
        filename: file.name,
        name: proposedNameFromFilename(file.name),
        preview,
      }
    }))
    if (!files.length) return
    setBusy(true)
    try {
      const form = new FormData()
      files.forEach(file => form.append('images', file, file.name))
      const created = await api('/api/profiles/import/preview', { method: 'POST', body: form })
      mergeBatch(created)
      pollRef.current = window.setInterval(() => poll(created.batch_id), 900)
    } catch (reason) {
      setError(reason.message)
    } finally {
      setBusy(false)
    }
  }

  const editIdentity = (identityId, field, value) => {
    setEdits(current => ({
      ...current,
      [identityId]: { ...current[identityId], [field]: value },
    }))
  }

  const confirm = async () => {
    if (!batch?.can_commit || busy) return
    setBusy(true); setError('')
    try {
      const identities = batch.identities.map(identity => {
        const edit = edits[identity.identity_id] || {}
        const renameExisting = Boolean(edit.name_edited && edit.update_existing_name)
        return {
          identity_id: identity.identity_id,
          name: edit.name,
          action: edit.action,
          skip: edit.skip,
          existing_person_id: edit.existing_person_id,
          primary_source_id: edit.primary_source_id,
          update_existing_name: renameExisting,
        }
      })
      setCommitResult(await api(`/api/profiles/import/${batch.batch_id}/commit`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ identities }),
      }))
    } catch (reason) {
      setError(reason.message)
    } finally {
      setBusy(false)
    }
  }

  const cancel = async () => {
    if (!batch || busy) return
    setBusy(true); setError('')
    try {
      await api(`/api/profiles/import/${batch.batch_id}/cancel`, { method: 'POST' })
      stopPolling()
      setBatch(null); setLocalRows([]); setEdits({}); setCommitResult(null)
    } catch (reason) {
      setError(reason.message)
    } finally {
      setBusy(false)
    }
  }

  const progress = batch?.progress
  return (
    <section aria-label="Batch Import">
      <div className="card manage-intro">
        <div>
          <div className="card-title">Import phone photos from a folder</div>
          <p>Each image must contain exactly one clear face. Nothing is added to Global Memory until you confirm the review.</p>
        </div>
        <label className="btn btn-primary manage-file-button">
          Select folder
          <input
            data-testid="batch-folder-input"
            type="file"
            webkitdirectory=""
            directory=""
            multiple
            accept="image/*"
            onChange={selectFolder}
            disabled={busy}
          />
        </label>
      </div>
      <ErrorBanner>{error}</ErrorBanner>
      {progress && (
        <div className="manage-progress card" aria-label="Batch progress">
          <Metric value={`${progress.processed} / ${progress.total}`} label="Processed" />
          <Metric value={progress.valid} label="Valid" />
          <Metric value={progress.existing_matches} label="Existing matches" />
          <Metric value={progress.new_profiles} label="New profiles" />
          <Metric value={progress.review_required} label="Review required" />
          <Metric value={progress.failed} label="Failed" />
        </div>
      )}
      {!batch && localRows.length > 0 && (
        <div className="manage-grid">
          {localRows.map(row => (
            <article className="manage-review-card" key={row.key}>
              <img src={row.preview} alt={row.filename} className="manage-photo" />
              <small>{row.filename}</small>
              <input aria-label={`Proposed name for ${row.filename}`} value={row.name}
                onChange={event => setLocalRows(rows => rows.map(item => item.key === row.key ? { ...item, name: event.target.value } : item))} />
              <span className="manage-state">waiting</span>
            </article>
          ))}
        </div>
      )}
      {batch?.identities?.length > 0 && (
        <div className="manage-grid" data-testid="batch-review-rows">
          {batch.identities.map(identity => {
            const edit = edits[identity.identity_id] || {}
            const primary = identity.photos.find(photo => photo.source_id === edit.primary_source_id) || identity.photos[0]
            return (
              <article className={`manage-review-card ${identity.validation_error ? 'is-invalid' : ''}`} key={identity.identity_id}>
                <div className="manage-comparison">
                  <div><span>Phone photo</span><SafeImage path={primary?.original_photo} alt="Original phone upload" placeholder="No image" /></div>
                  <div><span>Face crop</span><SafeImage path={primary?.face_crop} alt="Detected face crop" placeholder="No crop" /></div>
                  {identity.existing_candidate && (
                    <div><span>Existing profile</span><SafeImage path={identity.existing_candidate.profile_image} alt="Existing profile" placeholder="No profile image" /></div>
                  )}
                </div>
                <label>Name<input value={edit.name || ''} disabled={Boolean(identity.validation_error)}
                  onChange={event => setEdits(current => ({
                    ...current,
                    [identity.identity_id]: {
                      ...current[identity.identity_id],
                      name: event.target.value,
                      name_edited: event.target.value !== identity.proposed_name,
                    },
                  }))} /></label>
                <div className="manage-facts">
                  <span>Quality <strong>{identity.quality ? Math.round(identity.quality.sharpness) : '—'}</strong></span>
                  <span>Grouped photos <strong>{identity.grouped_photo_count}</strong></span>
                  <span>Similarity <strong>{identity.similarity == null ? '—' : `${(identity.similarity * 100).toFixed(1)}%`}</strong></span>
                </div>
                {identity.photos.length > 1 && (
                  <label>Primary phone crop
                    <select value={edit.primary_source_id || ''} onChange={event => editIdentity(identity.identity_id, 'primary_source_id', event.target.value)}>
                      {identity.photos.map(photo => <option key={photo.source_id} value={photo.source_id}>{photo.source_filename}</option>)}
                    </select>
                  </label>
                )}
                <label>Action
                  <select value={edit.skip ? 'skip' : (edit.action || 'skip')} disabled={Boolean(identity.validation_error)}
                    onChange={event => setEdits(current => ({
                      ...current,
                      [identity.identity_id]: {
                        ...current[identity.identity_id],
                        action: event.target.value,
                        skip: event.target.value === 'skip',
                      },
                    }))}>
                    <option value="create_new">Create new</option>
                    <option value="attach_existing">Attach existing</option>
                    <option value="review_required">Send to review</option>
                    <option value="skip">Skip</option>
                  </select>
                </label>
                {!edit.skip && edit.action === 'attach_existing' && (
                  <>
                    <label>Existing person
                      <select value={edit.existing_person_id || ''} onChange={event => editIdentity(identity.identity_id, 'existing_person_id', event.target.value)}>
                        <option value="">Select a profile</option>
                        {profiles.filter(profile => profile.is_active).map(profile => <option key={profile.person_id} value={profile.person_id}>{profile.name} · {profile.person_id}</option>)}
                      </select>
                    </label>
                    <label className="manage-check">
                      <input
                        type="checkbox"
                        data-testid={`rename-consent-${identity.identity_id}`}
                        disabled={!edit.name_edited}
                        checked={Boolean(edit.name_edited && edit.update_existing_name)}
                        onChange={event => editIdentity(identity.identity_id, 'update_existing_name', event.target.checked)}
                      />
                      Also rename the existing profile to “{edit.name || ''}”
                    </label>
                  </>
                )}
                <small className="manage-memory-result">
                  {identity.memory_match?.reason?.replaceAll('_', ' ')}
                  {identity.existing_candidate ? ` · ${identity.existing_candidate.name}` : ''}
                </small>
                {identity.validation_error && <div className="manage-validation">{identity.validation_error}</div>}
                {identity.photos.length > 1 && <div className="manage-thumbnails">{identity.photos.map(photo => <SafeImage key={photo.source_id} path={photo.face_crop} alt={photo.source_filename} placeholder="×" />)}</div>}
              </article>
            )
          })}
        </div>
      )}
      {batch && (
        <div className="manage-actions">
          <button className="btn btn-primary" onClick={confirm} disabled={!batch.can_commit || busy}>Confirm Batch</button>
          <button className="btn btn-ghost" onClick={cancel} disabled={busy}>Cancel Batch</button>
          {batch.state === 'processing' && <span>Processing continues in the background…</span>}
        </div>
      )}
      {commitResult && (
        <div className="card manage-commit-summary">
          <div className="card-title">Batch committed</div>
          <Metric value={commitResult.summary.profiles_created} label="Profiles created" />
          <Metric value={commitResult.summary.profiles_updated} label="Profiles updated" />
          <Metric value={commitResult.summary.items_sent_to_review} label="Sent to review" />
          <Metric value={commitResult.summary.skipped_items} label="Skipped" />
          <Metric value={commitResult.summary.failed_items} label="Failed" />
        </div>
      )}
    </section>
  )
}

function ManualCreate({ profiles, onChanged }) {
  const [name, setName] = useState('')
  const [notes, setNotes] = useState('')
  const [photo, setPhoto] = useState(null)
  const [preview, setPreview] = useState(null)
  const [edit, setEdit] = useState({})
  const [result, setResult] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  const runPreview = async (event) => {
    event.preventDefault()
    if (!photo) return
    setBusy(true); setError(''); setResult(null)
    try {
      const form = new FormData()
      form.append('name', name); form.append('notes', notes); form.append('photo', photo, photo.name)
      const data = await api('/api/profiles/manual/preview', { method: 'POST', body: form })
      const identity = data.identities[0]
      setPreview(data)
      setEdit({
        name,
        notes,
        action: identity.proposed_action,
        existing_person_id: identity.existing_candidate?.person_id || '',
        primary_source_id: identity.primary_source_id,
        update_existing_name: false,
      })
    } catch (reason) {
      setError(reason.message)
    } finally {
      setBusy(false)
    }
  }

  const confirm = async () => {
    setBusy(true); setError('')
    try {
      const data = await api('/api/profiles/manual/commit', {
        method: 'POST', headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ preview_id: preview.preview_id, identity: edit }),
      })
      setResult(data); onChanged()
    } catch (reason) {
      setError(reason.message)
    } finally {
      setBusy(false)
    }
  }

  const identity = preview?.identities?.[0]
  return (
    <section aria-label="Create Manually">
      <form className="card manage-form" onSubmit={runPreview}>
        <div className="card-title">Create or update one profile</div>
        <label>Name<input required maxLength={100} value={name} onChange={event => setName(event.target.value)} /></label>
        <label>Phone photo<input required type="file" accept="image/*" onChange={event => setPhoto(event.target.files?.[0] || null)} /></label>
        <label>Optional notes<textarea maxLength={4000} value={notes} onChange={event => setNotes(event.target.value)} /></label>
        <button className="btn btn-primary" disabled={busy || !name.trim() || !photo}>Preview match</button>
      </form>
      <ErrorBanner>{error}</ErrorBanner>
      {identity && (
        <div className="card manual-preview">
          <div className="manage-comparison">
            <div><span>Face crop</span><SafeImage path={identity.photos[0]?.face_crop} alt="Manual face crop" placeholder="No crop" /></div>
            {identity.existing_candidate && <div><span>Existing profile</span><SafeImage path={identity.existing_candidate.profile_image} alt="Existing candidate" placeholder="No image" /></div>}
          </div>
          {identity.validation_error ? <div className="manage-validation">{identity.validation_error}</div> : (
            <>
              <label>Decision
                <select value={edit.action || ''} onChange={event => setEdit(current => ({ ...current, action: event.target.value }))}>
                  <option value="create_new">Create new profile</option>
                  <option value="attach_existing">Update existing profile</option>
                  <option value="review_required">Require review</option>
                  <option value="skip">Skip</option>
                </select>
              </label>
              {edit.action === 'attach_existing' && <>
                <label>Existing person
                  <select value={edit.existing_person_id || ''} onChange={event => setEdit(current => ({ ...current, existing_person_id: event.target.value }))}>
                    <option value="">Select a profile</option>
                    {profiles.filter(profile => profile.is_active).map(profile => <option value={profile.person_id} key={profile.person_id}>{profile.name} · {profile.person_id}</option>)}
                  </select>
                </label>
                <label className="manage-check">
                  <input
                    type="checkbox"
                    data-testid="manual-rename-consent"
                    checked={Boolean(edit.update_existing_name)}
                    onChange={event => setEdit(current => ({ ...current, update_existing_name: event.target.checked }))}
                  />
                  Also rename the existing profile to “{edit.name || ''}”
                </label>
              </>}
              <p>{identity.memory_match.reason.replaceAll('_', ' ')}{identity.similarity != null ? ` · ${(identity.similarity * 100).toFixed(1)}%` : ''}</p>
              <button className="btn btn-primary" onClick={confirm} disabled={busy || (edit.action === 'attach_existing' && !edit.existing_person_id)}>Confirm profile</button>
            </>
          )}
        </div>
      )}
      {result && <div className="manage-banner is-success">Saved: {result.summary.profiles_created} created, {result.summary.profiles_updated} updated.</div>}
    </section>
  )
}

function ImportReviewResolution({ item, profiles, onResolved }) {
  const [target, setTarget] = useState(item.candidate_person_id || '')
  const [name, setName] = useState(item.proposed_name || '')
  const [notes, setNotes] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  const resolve = async (action) => {
    setBusy(true); setError('')
    try {
      const result = await api(`/api/profiles/reviews/${encodeURIComponent(item.review_key)}/resolve`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          action,
          target_person_id: action === 'attach_existing' ? target : undefined,
          name: action === 'create_new' ? name : undefined,
          notes: action === 'skip' ? undefined : notes,
        }),
      })
      await onResolved(result)
    } catch (reason) {
      setError(reason.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="manage-review-resolution" data-testid="review-resolution-controls">
      <ErrorBanner>{error}</ErrorBanner>
      <label>Existing profile
        <select value={target} onChange={event => setTarget(event.target.value)}>
          <option value="">Select a profile</option>
          {profiles.filter(profile => profile.is_active).map(profile => (
            <option value={profile.person_id} key={profile.person_id}>{profile.name} · {profile.person_id}</option>
          ))}
        </select>
      </label>
      <label>Approved new name
        <input value={name} maxLength={100} onChange={event => setName(event.target.value)} />
      </label>
      <label>Optional notes
        <input value={notes} maxLength={4000} onChange={event => setNotes(event.target.value)} />
      </label>
      <div className="manage-inline">
        <button className="btn btn-primary" disabled={busy || !target} onClick={() => resolve('attach_existing')}>Attach to existing</button>
        <button className="btn btn-ghost" disabled={busy || !name.trim()} onClick={() => resolve('create_new')}>Create new</button>
        <button className="btn btn-ghost" disabled={busy} onClick={() => resolve('skip')}>Skip</button>
      </div>
    </div>
  )
}


function ProfileDetail({ detail, profiles, refresh, selectProfile }) {
  const [name, setName] = useState(detail.name)
  const [notes, setNotes] = useState(detail.notes || '')
  const [phone, setPhone] = useState(null)
  const [photoPreview, setPhotoPreview] = useState(null)
  const [mergeTarget, setMergeTarget] = useState('')
  const [mergeConfirmed, setMergeConfirmed] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [reviewNotice, setReviewNotice] = useState('')
  useEffect(() => { setName(detail.name); setNotes(detail.notes || ''); setPhotoPreview(null) }, [detail.person_id])

  const act = async (work) => {
    setBusy(true); setError('')
    try { await work(); await refresh() } catch (reason) { setError(reason.message) } finally { setBusy(false) }
  }
  const save = () => act(() => api(`/api/profiles/${detail.person_id}`, {
    method: 'PATCH', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ name, notes }),
  }))
  const previewPhone = () => act(async () => {
    const form = new FormData(); form.append('photo', phone, phone.name)
    setPhotoPreview(await api(`/api/profiles/${detail.person_id}/photos/preview`, { method: 'POST', body: form }))
  })
  const commitPhone = () => act(async () => {
    const identity = photoPreview.identities[0]
    await api(`/api/profiles/${detail.person_id}/photos/commit`, {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        preview_id: photoPreview.preview_id,
        identity: { identity_id: identity.identity_id, primary_source_id: identity.primary_source_id },
      }),
    })
    setPhotoPreview(null); setPhone(null)
  })
  const merge = () => act(async () => {
    await api('/api/profiles/merge', {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        source_person_id: detail.person_id, target_person_id: mergeTarget,
        confirm: mergeConfirmed, reason: 'Supervisor confirmed duplicate in profile workspace',
      }),
    })
    selectProfile(mergeTarget)
  })

  return (
    <div className="profile-detail">
      <ErrorBanner>{error}</ErrorBanner>
      {reviewNotice && <div className="manage-banner is-success">{reviewNotice}</div>}
      <div className="card profile-heading">
        <SafeImage
          path={detail.effective_profile_image}
          paths={detail.image_candidates || []}
          alt={`${detail.name} primary face`}
          placeholder="No face crop"
        />
        <div>
          <h2>{detail.name}</h2><code>{detail.person_id}</code>
          <span className={`manage-state is-${detail.state}`}>{detail.state}</span>
          <p>Identity source: {detail.identity_source} · Created {detail.enrolled_at} · Updated {detail.updated_at}</p>
          <small data-testid="profile-image-origin">Primary image: {String(detail.profile_image_origin || 'placeholder').replaceAll('_', ' ')}</small>
        </div>
      </div>
      <div className="card manage-form">
        <div className="card-title">Profile fields</div>
        <label>Name<input value={name} maxLength={100} onChange={event => setName(event.target.value)} /></label>
        <label>Notes<textarea value={notes} maxLength={4000} onChange={event => setNotes(event.target.value)} /></label>
        <button className="btn btn-primary" onClick={save} disabled={busy}>Save changes</button>
      </div>
      <div className="card">
        <div className="card-title">Add phone photo</div>
        <div className="manage-inline">
          <input type="file" accept="image/*" onChange={event => setPhone(event.target.files?.[0] || null)} />
          <button className="btn btn-ghost" onClick={previewPhone} disabled={busy || !phone}>Preview</button>
        </div>
        {photoPreview?.identities?.[0] && <div className="phone-preview">
          <SafeImage path={photoPreview.identities[0].photos[0]?.face_crop} alt="New phone crop" placeholder="No crop" />
          <div><strong>Proposed result: {photoPreview.identities[0].proposed_action.replaceAll('_', ' ')}</strong>
            <p>A supervisor confirmation will append this evidence and make its face crop primary.</p>
            <button className="btn btn-primary" onClick={commitPhone} disabled={busy || Boolean(photoPreview.identities[0].validation_error)}>Attach phone photo</button>
          </div>
        </div>}
      </div>
      <div className="card">
        <div className="card-title">Phone-photo history</div>
        <div className="evidence-grid">
          {detail.phone_photos.length === 0 && <p>No phone photos yet.</p>}
          {detail.phone_photos.map(item => <article key={item.source_id}>
            <SafeImage path={item.face_crop_path} alt={item.source_filename} placeholder="No crop" />
            <small>{item.source_filename} · {item.created_at}</small>
            {item.is_supervisor_selected ? <span className="manage-state">Primary (supervisor)</span> :
              item.is_primary ? <span className="manage-state">Primary</span> :
              <button className="btn btn-ghost" onClick={() => act(() => api(`/api/profiles/${detail.person_id}/primary-photo`, {
                method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ source_id: item.source_id }),
              }))}>Make primary</button>}
          </article>)}
        </div>
      </div>
      <div className="card">
        <div className="card-title">Video-crop history</div>
        <div className="evidence-grid">
          {detail.video_evidence.length === 0 && <p>No video evidence.</p>}
          {detail.video_evidence.map((item, index) => <article key={`${item.path}-${index}`}><SafeImage path={item.path} alt="Video face evidence" placeholder="No crop" /><small>{item.session_date}</small></article>)}
        </div>
      </div>
      <div className="card">
        <div className="card-title">Recent appearances</div>
        {detail.recent_appearances.length === 0 ? <p>No appearances.</p> : detail.recent_appearances.map(event => <div className="manage-log-row" key={event.id}><span>{event.event_type.replaceAll('_', ' ')}</span><time>{event.ts}</time></div>)}
      </div>
      <div className="card">
        <div className="card-title">Pending review suggestions</div>
        {detail.pending_review_suggestions.length === 0 ? <p>No uncertain suggestions.</p> :
          detail.pending_review_suggestions.map(item => item.kind === 'phone_import' ? (
            <div className="is-import-review" data-testid="phone-import-review" key={item.review_key}>
              <div className="manage-log-row">
                <SafeImage path={item.evidence?.[0]?.face_crop_path} alt="Uncertain phone crop" placeholder="No crop" />
                <span>Phone import “{item.proposed_name}” ↔ {detail.name}</span>
                <strong>{item.similarity == null ? '—' : `${(item.similarity * 100).toFixed(1)}%`}</strong>
              </div>
              <ImportReviewResolution
                item={item}
                profiles={profiles}
                onResolved={async result => {
                  setReviewNotice(
                    result.status === 'skipped'
                      ? 'Review skipped.'
                      : `Review resolved: ${String(result.status).replaceAll('_', ' ')}.`
                  )
                  await refresh()
                }}
              />
            </div>
          ) : (
            <div className="manage-log-row" key={item.suggestion_id}>
              <span>{item.source_name} ↔ {item.candidate_name}</span>
              <strong>{(item.similarity * 100).toFixed(1)}%</strong>
            </div>
          ))}
      </div>
      <div className="card danger-zone">
        <div className="card-title">Profile state</div>
        {detail.state === 'active' ?
          <button className="btn btn-danger" disabled={busy} onClick={() => act(() => api(`/api/profiles/${detail.person_id}/archive`, { method: 'POST' }))}>Archive profile</button> :
          detail.state === 'archived' && <button className="btn btn-success" disabled={busy} onClick={() => act(() => api(`/api/profiles/${detail.person_id}/restore`, { method: 'POST' }))}>Restore profile</button>}
      </div>
      {detail.state === 'active' && <div className="card danger-zone">
        <div className="card-title">Merge duplicate profile</div>
        <p>Move this profile’s identity evidence into a target. This profile becomes inactive and an audit record is retained.</p>
        <select value={mergeTarget} onChange={event => { setMergeTarget(event.target.value); setMergeConfirmed(false) }}>
          <option value="">Select target</option>
          {profiles.filter(profile => profile.is_active && profile.person_id !== detail.person_id).map(profile =>
            <option key={profile.person_id} value={profile.person_id}>{profile.name} · {profile.person_id} · {profile.phone_photo_count} phone / {profile.video_evidence_count} video</option>)}
        </select>
        <label className="manage-check"><input type="checkbox" checked={mergeConfirmed} onChange={event => setMergeConfirmed(event.target.checked)} /> I confirm these profiles represent the same person.</label>
        <button className="btn btn-danger" onClick={merge} disabled={busy || !mergeTarget || !mergeConfirmed}>Merge into target</button>
      </div>}
    </div>
  )
}

function ManageProfiles({ profiles, reloadProfiles }) {
  const [query, setQuery] = useState('')
  const [state, setState] = useState('active')
  const [selected, setSelected] = useState('')
  const [detail, setDetail] = useState(null)
  const [reviews, setReviews] = useState([])
  const [reviewNotice, setReviewNotice] = useState('')
  const [error, setError] = useState('')

  const loadDetail = useCallback(async (personId = selected) => {
    if (!personId) { setDetail(null); return }
    try { setDetail(await api(`/api/profiles/${encodeURIComponent(personId)}`)); setError('') }
    catch (reason) { setError(reason.message); setDetail(null) }
  }, [selected])

  const loadReviews = useCallback(async () => {
    try {
      const data = await api('/api/profiles/reviews')
      setReviews(data.reviews || [])
    } catch (reason) {
      setError(reason.message)
    }
  }, [])

  useEffect(() => { loadDetail(selected) }, [selected])
  useEffect(() => { loadReviews() }, [loadReviews])
  const refresh = async () => {
    await reloadProfiles()
    await loadDetail(selected)
    await loadReviews()
  }
  const visible = useMemo(() => profiles.filter(profile => {
    if (state === 'active' && !profile.is_active) return false
    if (state === 'archived' && (profile.is_active || profile.merged_into_person_id)) return false
    const needle = query.toLowerCase()
    return !needle || profile.name.toLowerCase().includes(needle) || profile.person_id.toLowerCase().includes(needle)
  }), [profiles, query, state])

  return (
    <section className="profiles-layout" aria-label="Manage Profiles">
      <aside className="card profiles-sidebar">
        <div className="card-title">Profiles</div>
        <input type="search" placeholder="Search name or person ID" value={query} onChange={event => setQuery(event.target.value)} />
        <select value={state} onChange={event => setState(event.target.value)}>
          <option value="active">Active</option><option value="archived">Archived</option><option value="all">All</option>
        </select>
        <div className="profile-list">
          {visible.map(profile => <button key={profile.person_id} className={selected === profile.person_id ? 'is-selected' : ''} onClick={() => setSelected(profile.person_id)}>
            <SafeImage path={profile.effective_profile_image} paths={profile.image_candidates || []} alt={profile.name} placeholder="?" />
            <span><strong>{profile.name}</strong><small>{profile.person_id} · {profile.phone_photo_count} phone / {profile.video_evidence_count} video</small></span>
          </button>)}
          {visible.length === 0 && <p>No profiles match.</p>}
        </div>
      </aside>
      <main>
        <ErrorBanner>{error}</ErrorBanner>
        {reviewNotice && <div className="manage-banner is-success">{reviewNotice}</div>}
        <div className="card">
          <div className="card-title">Pending phone-import reviews</div>
          {reviews.length === 0 ? <p>No pending phone-import reviews.</p> : reviews.map(item => (
            <div className="is-import-review" data-testid="global-phone-import-review" key={item.review_key}>
              <div className="manage-log-row">
                <SafeImage path={item.evidence?.[0]?.face_crop_path} alt="Uncertain phone crop" placeholder="No crop" />
                <span>{item.proposed_name}</span>
                <strong>{item.similarity == null ? '—' : `${(item.similarity * 100).toFixed(1)}%`}</strong>
              </div>
              <ImportReviewResolution
                item={item}
                profiles={profiles}
                onResolved={async result => {
                  setReviewNotice(
                    result.status === 'skipped'
                      ? 'Review skipped.'
                      : `Review resolved: ${String(result.status).replaceAll('_', ' ')}.`
                  )
                  await refresh()
                }}
              />
            </div>
          ))}
        </div>
        {detail ? <ProfileDetail detail={detail} profiles={profiles} refresh={refresh} selectProfile={setSelected} /> :
          <div className="card manage-empty">Select a profile to view its evidence and management controls.</div>}
      </main>
    </section>
  )
}

export default function ProfileManager() {
  const [tab, setTab] = useState(0)
  const [profiles, setProfiles] = useState([])
  const [error, setError] = useState('')

  const reloadProfiles = useCallback(async () => {
    try {
      const data = await api('/api/profiles?state=all')
      setProfiles(data.profiles || []); setError('')
    } catch (reason) {
      setError(reason.message)
    }
  }, [])
  useEffect(() => { reloadProfiles() }, [reloadProfiles])

  return (
    <div className="manage-workspace">
      <div className="manage-heading">
        <div><h2>Profile Management</h2><p>Review phone-photo evidence before changing Global Memory.</p></div>
      </div>
      <div className="tabs manage-tabs" role="tablist">
        {MANAGEMENT_TABS.map((label, index) => <button key={label} role="tab" aria-selected={tab === index}
          className={`tab-btn${tab === index ? ' active' : ''}`} onClick={() => setTab(index)}>{label}</button>)}
      </div>
      <ErrorBanner>{error}</ErrorBanner>
      <div className="manage-tab-panel" role="tabpanel">
        {tab === 0 && <BatchImport profiles={profiles} />}
        {tab === 1 && <ManualCreate profiles={profiles} onChanged={reloadProfiles} />}
        {tab === 2 && <ManageProfiles profiles={profiles} reloadProfiles={reloadProfiles} />}
      </div>
    </div>
  )
}
