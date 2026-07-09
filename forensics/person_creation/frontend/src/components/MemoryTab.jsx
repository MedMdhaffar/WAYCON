import { useCallback, useEffect, useState } from 'react'

function ImageThumb({ path, alt, fallback, size = 44, rounded = '50%' }) {
  return (
    <div style={{ width: size, height: size, borderRadius: rounded, overflow: 'hidden', background: '#1e2330', flexShrink: 0, display: 'grid', placeItems: 'center' }}>
      {path ? (
        <img
          src={`/api/images?path=${encodeURIComponent(path)}`}
          alt={alt}
          style={{ width: '100%', height: '100%', objectFit: 'cover' }}
          onError={event => { event.currentTarget.style.display = 'none' }}
        />
      ) : (
        <span style={{ color: '#64748b', fontWeight: 700, fontSize: 12 }}>{fallback}</span>
      )}
    </div>
  )
}

function PersonCard({ person, selected, onSelect }) {
  const app = person.latest_appearance
  return (
    <button
      type="button"
      onClick={() => onSelect(person)}
      className="memory-person-card"
      style={{
        width: '100%',
        textAlign: 'left',
        cursor: 'pointer',
        border: `1px solid ${selected ? '#8b5cf6' : '#1e2330'}`,
        background: selected ? 'rgba(139, 92, 246, 0.16)' : '#0f1117',
        borderRadius: 6,
        padding: 12,
        marginBottom: 8,
        color: '#e2e8f0',
        display: 'flex',
        gap: 12,
        alignItems: 'center',
      }}
    >
      <ImageThumb path={person.profile_image} alt={person.name} fallback={person.name?.charAt(0) || '?'} />
      <div style={{ minWidth: 0, flex: 1 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', gap: 10, alignItems: 'flex-start' }}>
          <div style={{ minWidth: 0 }}>
            <div style={{ fontWeight: 700, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{person.name}</div>
            <div style={{ color: '#64748b', fontSize: 12 }}>{person.person_id}</div>
          </div>
          <div style={{ color: '#64748b', fontSize: 11, whiteSpace: 'nowrap' }}>{person.enrolled_at}</div>
        </div>
        {app && (
          <div style={{ color: '#94a3b8', fontSize: 12, marginTop: 6, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {[app.top, app.bottom].filter(Boolean).join(' · ')}
          </div>
        )}
      </div>
    </button>
  )
}

function LogRow({ entry }) {
  const isNew = entry.event_type === 'new_enrollment'
  const delta = entry.embedding_count_before !== null
    ? `${entry.embedding_count_before} -> ${entry.embedding_count_after}`
    : `${entry.embedding_count_after} crops`

  return (
    <div style={{ borderBottom: '1px solid #1e2330', padding: '10px 0', fontSize: 13, display: 'flex', gap: 12 }}>
      <ImageThumb path={entry.best_face_crop} alt="face crop" fallback="?" size={42} rounded={6} />
      <div style={{ minWidth: 0, flex: 1 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', gap: 10 }}>
          <span style={{ color: isNew ? '#60a5fa' : '#22c55e', fontWeight: 600 }}>
            {isNew ? 'First enrollment' : 'Recognized'}
          </span>
          <span style={{ color: '#64748b', fontSize: 11 }}>{entry.ts}</span>
        </div>
        <div style={{ color: '#94a3b8', fontSize: 12, marginTop: 3 }}>
          {!isNew && entry.similarity !== null && (
            <span>Similarity: {Number(entry.similarity).toFixed(4)} · </span>
          )}
          <span>Crops: {delta}</span>
        </div>
        {(entry.video_sources ?? []).map((src, i) => (
          <div key={`${entry.id}-${i}`} style={{ color: '#64748b', fontSize: 11, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {src}
          </div>
        ))}
      </div>
    </div>
  )
}

function RenameField({ personId, initialName, onRenamed }) {
  const [editing, setEditing] = useState(false)
  const [value, setValue] = useState(initialName || '')
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    setValue(initialName || '')
  }, [initialName])

  const save = async () => {
    const next = value.trim()
    if (!next || next === initialName) {
      setValue(initialName || '')
      setEditing(false)
      return
    }
    setSaving(true)
    setError('')
    try {
      const res = await fetch(`/api/memory/persons/${personId}/rename`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: next }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.error || 'Rename failed')
      setEditing(false)
      onRenamed?.(next)
    } catch (err) {
      setError(err.message || 'Rename failed')
    } finally {
      setSaving(false)
    }
  }

  const onKeyDown = (event) => {
    if (event.key === 'Enter') save()
    if (event.key === 'Escape') {
      setValue(initialName || '')
      setEditing(false)
      setError('')
    }
  }

  if (editing) {
    return (
      <div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
          <input
            autoFocus
            value={value}
            onChange={e => setValue(e.target.value)}
            onKeyDown={onKeyDown}
            disabled={saving}
            style={{
              width: 220,
              padding: '7px 10px',
              borderRadius: 6,
              border: '1px solid #8b5cf6',
              background: '#0f1117',
              color: '#e2e8f0',
              fontSize: 20,
              fontWeight: 700,
            }}
          />
          <button className="btn btn-primary" type="button" onClick={save} disabled={saving} style={{ padding: '7px 12px' }}>
            {saving ? 'Saving...' : 'Save'}
          </button>
          <button
            className="btn btn-ghost"
            type="button"
            onClick={() => {
              setValue(initialName || '')
              setEditing(false)
              setError('')
            }}
            disabled={saving}
            style={{ padding: '7px 12px' }}
          >
            Cancel
          </button>
        </div>
        {error && <div style={{ color: '#ef4444', fontSize: 12, marginTop: 6 }}>{error}</div>}
      </div>
    )
  }

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
      <h2 style={{ margin: 0, fontSize: 22 }}>{value}</h2>
      <button
        type="button"
        onClick={() => setEditing(true)}
        title="Rename person"
        style={{
          border: '1px solid #1e2330',
          background: '#0f1117',
          color: '#94a3b8',
          borderRadius: 6,
          cursor: 'pointer',
          padding: '4px 7px',
          fontSize: 12,
        }}
      >
        Rename
      </button>
    </div>
  )
}

function ProfileImageEditor({ personId, currentImage, source, onUpdated }) {
  const [mode, setMode] = useState(null)
  const [pathInput, setPathInput] = useState('')
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')

  const handleFileUpload = async (event) => {
    const file = event.target.files?.[0]
    if (!file) return
    setSaving(true)
    setError('')
    const form = new FormData()
    form.append('image', file)
    try {
      const res = await fetch(`/api/memory/persons/${personId}/profile-image`, { method: 'POST', body: form })
      const data = await res.json()
      if (!res.ok) throw new Error(data.error || 'Upload failed')
      onUpdated?.(data.profile_image, data.source)
      setMode(null)
    } catch (err) {
      setError(err.message || 'Upload failed')
    } finally {
      setSaving(false)
    }
  }

  const handlePathSubmit = async () => {
    const nextPath = pathInput.trim()
    if (!nextPath) return
    setSaving(true)
    setError('')
    try {
      const res = await fetch(`/api/memory/persons/${personId}/profile-image`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: nextPath }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.error || 'Failed to update profile image')
      onUpdated?.(data.profile_image, data.source)
      setPathInput('')
      setMode(null)
    } catch (err) {
      setError(err.message || 'Failed to update profile image')
    } finally {
      setSaving(false)
    }
  }

  const handleAutoSelect = async (force = false) => {
    setSaving(true)
    setError('')
    try {
      const res = await fetch(`/api/memory/persons/${personId}/profile-image/auto${force ? '?force=true' : ''}`, { method: 'POST' })
      const data = await res.json()
      if (res.status === 409) {
        if (window.confirm('This person has a manually set profile image. Override it with the best auto crop?')) {
          await handleAutoSelect(true)
        }
        return
      }
      if (!res.ok) throw new Error(data.error || 'Auto-select failed')
      onUpdated?.(data.profile_image, data.source)
      setMode(null)
    } catch (err) {
      setError(err.message || 'Auto-select failed')
    } finally {
      setSaving(false)
    }
  }

  return (
    <div style={{ display: 'grid', gap: 8, width: 120, flexShrink: 0 }}>
      <div style={{ position: 'relative', width: 96, height: 96, borderRadius: 8, overflow: 'hidden', background: '#1e2330', border: '1px solid #1e2330' }}>
        {currentImage ? (
          <img
            src={`/api/images?path=${encodeURIComponent(currentImage)}`}
            alt="Profile"
            style={{ width: '100%', height: '100%', objectFit: 'cover' }}
            onError={event => { event.currentTarget.style.display = 'none' }}
          />
        ) : (
          <div style={{ width: '100%', height: '100%', display: 'grid', placeItems: 'center', color: '#64748b', fontWeight: 700 }}>?</div>
        )}
        {source === 'manual' && (
          <span style={{ position: 'absolute', top: 4, right: 4, background: '#7c3aed', color: 'white', fontSize: 10, padding: '2px 5px', borderRadius: 999 }}>
            manual
          </span>
        )}
      </div>

      <button type="button" className="btn btn-ghost" onClick={() => setMode(current => current ? null : 'menu')} disabled={saving} style={{ padding: '6px 8px', fontSize: 12, width: 96 }}>
        Change
      </button>

      {mode === 'menu' && (
        <div style={{ display: 'grid', gap: 6 }}>
          <button type="button" className="btn btn-ghost" onClick={() => setMode('upload')} style={{ padding: '6px 8px', fontSize: 12, textAlign: 'left' }}>Upload file</button>
          <button type="button" className="btn btn-ghost" onClick={() => setMode('path')} style={{ padding: '6px 8px', fontSize: 12, textAlign: 'left' }}>Set path</button>
          <button type="button" className="btn btn-ghost" onClick={() => handleAutoSelect(false)} disabled={saving} style={{ padding: '6px 8px', fontSize: 12, textAlign: 'left' }}>Best auto crop</button>
        </div>
      )}

      {mode === 'upload' && (
        <div style={{ display: 'grid', gap: 6 }}>
          <input type="file" accept="image/jpeg,image/png" onChange={handleFileUpload} disabled={saving} style={{ color: '#94a3b8', fontSize: 11, maxWidth: 160 }} />
          <button type="button" className="btn btn-ghost" onClick={() => setMode('menu')} disabled={saving} style={{ padding: '5px 7px', fontSize: 12 }}>Cancel</button>
        </div>
      )}

      {mode === 'path' && (
        <div style={{ display: 'grid', gap: 6, width: 240 }}>
          <input
            type="text"
            value={pathInput}
            onChange={event => setPathInput(event.target.value)}
            placeholder="/mnt/c/path/image.jpg"
            disabled={saving}
            style={{ padding: '7px 9px', borderRadius: 6, border: '1px solid #1e2330', background: '#0f1117', color: '#e2e8f0', fontSize: 12 }}
          />
          <div style={{ display: 'flex', gap: 6 }}>
            <button type="button" className="btn btn-primary" onClick={handlePathSubmit} disabled={saving || !pathInput.trim()} style={{ padding: '5px 9px', fontSize: 12 }}>{saving ? 'Saving...' : 'Set'}</button>
            <button type="button" className="btn btn-ghost" onClick={() => setMode('menu')} disabled={saving} style={{ padding: '5px 9px', fontSize: 12 }}>Cancel</button>
          </div>
        </div>
      )}

      {error && <div style={{ color: '#ef4444', fontSize: 12 }}>{error}</div>}
    </div>
  )
}

function GallerySection({ personId }) {
  const [faces, setFaces] = useState([])
  const [bodies, setBodies] = useState([])
  const [tab, setTab] = useState('face')

  useEffect(() => {
    if (!personId) return
    let cancelled = false
    fetch(`/api/memory/persons/${personId}/gallery?type=face`)
      .then(res => res.json())
      .then(data => { if (!cancelled && Array.isArray(data)) setFaces(data) })
      .catch(() => {})
    fetch(`/api/memory/persons/${personId}/gallery?type=body`)
      .then(res => res.json())
      .then(data => { if (!cancelled && Array.isArray(data)) setBodies(data) })
      .catch(() => {})
    return () => { cancelled = true }
  }, [personId])

  const items = tab === 'face' ? faces : bodies

  return (
    <div className="card" style={{ margin: 0 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 }}>
        <div className="card-title" style={{ margin: 0 }}>Best Crops Gallery</div>
        <div style={{ marginLeft: 'auto', display: 'flex', gap: 6 }}>
          <button type="button" onClick={() => setTab('face')} className={`btn ${tab === 'face' ? 'btn-primary' : 'btn-ghost'}`} style={{ padding: '5px 10px', fontSize: 12 }}>Faces ({faces.length})</button>
          <button type="button" onClick={() => setTab('body')} className={`btn ${tab === 'body' ? 'btn-primary' : 'btn-ghost'}`} style={{ padding: '5px 10px', fontSize: 12 }}>Bodies ({bodies.length})</button>
        </div>
      </div>
      {items.length ? (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(72px, 1fr))', gap: 8 }}>
          {items.map(entry => (
            <div key={entry.id} title={entry.session_date} style={{ position: 'relative', borderRadius: 6, overflow: 'hidden', background: '#0f1117', border: '1px solid #1e2330' }}>
              <img
                src={`/api/images?path=${encodeURIComponent(entry.path)}`}
                alt={entry.crop_type}
                style={{ width: '100%', aspectRatio: entry.crop_type === 'face' ? '1 / 1' : '1 / 2', objectFit: 'cover', display: 'block' }}
                onError={event => { event.currentTarget.style.display = 'none' }}
              />
              <div style={{ position: 'absolute', left: 0, right: 0, bottom: 0, background: 'rgba(0,0,0,0.72)', color: '#cbd5e1', fontSize: 10, textAlign: 'center', padding: '2px 3px' }}>
                {Number(entry.sharpness || 0).toFixed(0)}
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div style={{ color: '#64748b', fontSize: 13 }}>No {tab} crops in gallery yet.</div>
      )}
    </div>
  )
}

function PersonDetail({ personId, onRosterRefresh }) {
  const [detail, setDetail] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    if (!personId) return
    let cancelled = false
    setLoading(true)
    setError('')
    fetch(`/api/memory/persons/${personId}`)
      .then(res => res.json().then(data => ({ ok: res.ok, data })))
      .then(({ ok, data }) => {
        if (cancelled) return
        if (!ok) throw new Error(data.error || 'Failed to load person')
        setDetail(data)
      })
      .catch(err => {
        if (!cancelled) setError(err.message)
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => { cancelled = true }
  }, [personId])

  if (loading) return <div style={{ color: '#64748b', padding: 24 }}>Loading...</div>
  if (error) return <div style={{ color: '#ef4444', padding: 24 }}>{error}</div>
  if (!detail) return null

  const app = detail.latest_appearance
  const history = detail.recognition_history ?? []

  return (
    <div style={{ display: 'grid', gap: 16 }}>
      <div style={{ display: 'flex', gap: 16, alignItems: 'flex-start' }}>
        <ProfileImageEditor
          personId={detail.person_id}
          currentImage={detail.profile_image}
          source={detail.profile_image_source}
          onUpdated={(newPath, source) => {
            setDetail(current => ({
              ...current,
              profile_image: newPath,
              profile_image_source: source || current.profile_image_source,
            }))
            setTimeout(() => onRosterRefresh?.(), 0)
          }}
        />
        <div>
          <RenameField
            personId={detail.person_id}
            initialName={detail.name}
            onRenamed={(newName) => {
              setDetail(current => ({ ...current, name: newName }))
              setTimeout(() => onRosterRefresh?.(), 0)
            }}
          />
          <div style={{ color: '#64748b', fontSize: 13 }}>{detail.person_id}</div>
          <div style={{ color: '#64748b', fontSize: 12, marginTop: 4 }}>
            Enrolled: {detail.enrolled_at} · Last seen: {detail.updated_at} · Total crops: {detail.embedding_count}
          </div>
        </div>
      </div>

      {app && (
        <div className="card" style={{ margin: 0 }}>
          <div className="card-title">
            Latest Appearance · {app.date}
            {app.is_stale && <span style={{ color: '#eab308', marginLeft: 8 }}>(previous day)</span>}
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, minmax(0, 1fr))', gap: 12 }}>
            <div>
              <div style={{ color: '#64748b', fontSize: 12 }}>Top</div>
              <div>{app.top || '-'}</div>
            </div>
            <div>
              <div style={{ color: '#64748b', fontSize: 12 }}>Bottom</div>
              <div>{app.bottom || '-'}</div>
            </div>
            <div>
              <div style={{ color: '#64748b', fontSize: 12 }}>Shoes</div>
              <div>{app.shoes || '-'}</div>
            </div>
          </div>
          {(app.top_color || app.bottom_color) && (
            <div style={{ color: '#94a3b8', fontSize: 12, marginTop: 10 }}>
              Colors: {app.top_color || '-'} / {app.bottom_color || '-'}
            </div>
          )}
        </div>
      )}

      <div className="card" style={{ margin: 0 }}>
        <div className="card-title">Recognition History ({history.length})</div>
        <div style={{ maxHeight: 360, overflowY: 'auto' }}>
          {history.length ? history.map(entry => (
            <LogRow key={entry.id} entry={entry} />
          )) : (
            <div style={{ color: '#64748b', fontSize: 13 }}>No events recorded yet.</div>
          )}
        </div>
      </div>

      <GallerySection personId={detail.person_id} />
    </div>
  )
}

export default function MemoryTab() {
  const [persons, setPersons] = useState([])
  const [selected, setSelected] = useState(null)
  const [query, setQuery] = useState('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  const loadPersons = useCallback(() => {
    const q = query.trim()
    const url = q ? `/api/memory/search?q=${encodeURIComponent(q)}` : '/api/memory/persons'
    setError('')
    return fetch(url)
      .then(res => res.json().then(data => ({ ok: res.ok, data })))
      .then(({ ok, data }) => {
        if (!ok) throw new Error(data.error || 'Failed to load memory')
        setPersons(data)
        setLoading(false)
        setSelected(current => {
          if (!current) return current
          return data.find(person => person.person_id === current.person_id) || null
        })
      })
  }, [query])

  useEffect(() => {
    let cancelled = false
    const load = () => {
      loadPersons().catch(err => {
        if (!cancelled) {
          setError(err.message)
          setLoading(false)
        }
      })
    }

    load()
    const interval = setInterval(load, 10000)
    return () => {
      cancelled = true
      clearInterval(interval)
    }
  }, [loadPersons])

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '320px minmax(0, 1fr)', gap: 16, minHeight: 620 }}>
      <div className="card" style={{ margin: 0, display: 'flex', flexDirection: 'column', minHeight: 620 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 }}>
          <div className="card-title" style={{ margin: 0 }}>Enrolled Persons</div>
          <div style={{ marginLeft: 'auto', color: '#64748b', fontSize: 12 }}>{persons.length} total</div>
        </div>
        <input
          type="text"
          value={query}
          onChange={e => setQuery(e.target.value)}
          placeholder="Search by name or ID..."
          style={{
            width: '100%',
            padding: '8px 10px',
            borderRadius: 6,
            border: '1px solid #1e2330',
            background: '#0f1117',
            color: '#e2e8f0',
            marginBottom: 12,
          }}
        />
        {error && <div style={{ color: '#ef4444', fontSize: 13, marginBottom: 8 }}>{error}</div>}
        <div style={{ overflowY: 'auto', flex: 1 }}>
          {loading ? (
            <div style={{ color: '#64748b', fontSize: 13 }}>Loading...</div>
          ) : persons.length ? (
            persons.map(person => (
              <PersonCard
                key={person.person_id}
                person={person}
                selected={selected?.person_id === person.person_id}
                onSelect={setSelected}
              />
            ))
          ) : (
            <div style={{ color: '#64748b', fontSize: 13 }}>{query ? 'No matches found.' : 'No persons enrolled yet.'}</div>
          )}
        </div>
      </div>

      <div className="card" style={{ margin: 0, minHeight: 620, overflowY: 'auto' }}>
        {selected ? (
          <PersonDetail personId={selected.person_id} onRosterRefresh={loadPersons} />
        ) : (
          <div style={{ color: '#64748b', minHeight: 540, display: 'grid', placeItems: 'center' }}>
            Select a person from the roster.
          </div>
        )}
      </div>
    </div>
  )
}
