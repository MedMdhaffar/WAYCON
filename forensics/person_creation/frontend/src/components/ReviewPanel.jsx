import { useState } from 'react'

export default function ReviewPanel({ jobId, snapshot, clothingOverride, onDone, finalStatus }) {
  const [loading, setLoading] = useState(false)
  const [corrections, setCorrections] = useState('')
  const [error, setError] = useState('')

  const profile = snapshot?.profile ?? {}
  const appearance = profile.appearance ?? snapshot?.clothing_structured ?? {}
  const previewProfiles = snapshot?.profile_preview?.profiles ?? []
  const cleanName = (item) => {
    const isInternal = value => /cluster[_\s-]?\d+/i.test(value || '')
    if (item?.name && !isInternal(item.name)) return item.name
    if (item?.id && !isInternal(item.id)) return item.id
    return 'Pending identity'
  }

  const handleApprove = async () => {
    setError('')
    setLoading(true)
    try {
      const res = await fetch(`/api/person/approve/${jobId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          corrections: corrections.trim() || null,
          clothing_override: clothingOverride || null,
        }),
      })
      const data = await res.json()
      if (data.error) throw new Error(data.error)
      onDone()
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  if (finalStatus === 'done') {
    const savedProfiles = Object.values(snapshot?.per_cluster_profiles ?? {})
    return (
      <div className="card" style={{ borderColor: '#22c55e' }}>
        <div style={{ textAlign: 'center', padding: '24px' }}>
          <div style={{ fontSize: '42px', marginBottom: '12px', color: '#22c55e' }}>Saved</div>
          <div style={{ fontSize: '18px', fontWeight: 600, color: '#22c55e', marginBottom: '8px' }}>Profiles Saved</div>
          <div style={{ fontSize: '14px', color: '#64748b' }}>
            {(snapshot?.per_cluster_profiles && Object.keys(snapshot.per_cluster_profiles).length) || 1} profile(s) written
          </div>
          {!!savedProfiles.length && (
            <div style={{ marginTop: 18, display: 'grid', gap: 8, textAlign: 'left' }}>
              {savedProfiles.map(profile => (
                <div key={profile.id} style={{ border: '1px solid #1e2330', borderRadius: 6, padding: 10, background: '#0f1117' }}>
                  <h2 style={{ margin: 0, color: '#e2e8f0', fontSize: 18 }}>{profile.name || profile.id}</h2>
                  <p style={{ margin: '3px 0 0', color: '#64748b', fontSize: 13 }}>{profile.id}</p>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    )
  }

  const rowStyle = { display: 'flex', justifyContent: 'space-between', padding: '8px 0', borderBottom: '1px solid #1e2330', fontSize: '14px' }
  const keyStyle = { color: '#64748b' }
  const valStyle = { color: '#e2e8f0', fontWeight: 500 }

  return (
    <div className="card">
      <div className="card-title">Profile Review</div>

      {!!previewProfiles.length && (
        <div style={{ marginBottom: '20px', display: 'grid', gap: '10px' }}>
          {previewProfiles.map(p => {
            const pAppearance = p.appearance ?? {}
            return (
              <div key={p.cluster_id} style={{ border: '1px solid #1e2330', borderRadius: 6, padding: 10, background: '#0f1117' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, marginBottom: 6 }}>
                  <div>
                    <h2 style={{ color: '#e2e8f0', fontWeight: 600, fontSize: 16, margin: 0 }}>{cleanName(p)}</h2>
                    {p.id && cleanName(p) !== 'Pending identity' && <p style={{ color: '#64748b', fontSize: 12, margin: '2px 0 0' }}>{p.id}</p>}
                  </div>
                </div>
                <div style={{ color: '#64748b', fontSize: 12, marginBottom: 4 }}>
                  Faces: {p.face_crop_count} | Bodies: {p.associations_count} | Confidence: {p.cluster_confidence}
                </div>
                <div style={{ color: '#e2e8f0', fontSize: 13 }}>{pAppearance.full || '-'}</div>
              </div>
            )
          })}
        </div>
      )}

      {!previewProfiles.length && (
        <div style={{ marginBottom: '20px' }}>
          <div style={{ borderBottom: '1px solid #1e2330', paddingBottom: 10, marginBottom: 8 }}>
            <h2 style={{ margin: 0, color: '#e2e8f0', fontSize: 20 }}>{cleanName(profile)}</h2>
            {profile.id && cleanName(profile) !== 'Pending identity' && <p style={{ margin: '3px 0 0', color: '#64748b', fontSize: 13 }}>{profile.id}</p>}
          </div>
          <div style={rowStyle}><span style={keyStyle}>Face crops</span><span style={valStyle}>{profile.face_crop_count ?? (snapshot?.quality_face_crops?.length ?? '-')}</span></div>
          <div style={rowStyle}><span style={keyStyle}>Associations</span><span style={valStyle}>{(snapshot?.associations ?? []).length}</span></div>
          <div style={rowStyle}><span style={keyStyle}>Top</span><span style={valStyle}>{appearance.top || '-'}</span></div>
          <div style={rowStyle}><span style={keyStyle}>Bottom</span><span style={valStyle}>{appearance.bottom || '-'}</span></div>
          <div style={rowStyle}><span style={keyStyle}>Shoes</span><span style={valStyle}>{appearance.shoes || '-'}</span></div>
          <div style={{ ...rowStyle, borderBottom: 'none' }}>
            <span style={keyStyle}>Full</span>
            <span style={{ ...valStyle, maxWidth: '60%', textAlign: 'right' }}>{appearance.full || '-'}</span>
          </div>
        </div>
      )}

      <div style={{ marginBottom: '16px' }}>
        <label style={{ fontSize: '13px', color: '#94a3b8', display: 'block', marginBottom: '6px' }}>Corrections / notes (optional)</label>
        <textarea
          value={corrections}
          onChange={e => setCorrections(e.target.value)}
          rows={3}
          placeholder="e.g. Clothing description is wrong - top should be black t-shirt..."
          style={{
            width: '100%', padding: '8px 12px', background: '#0f1117',
            border: '1px solid #1e2330', borderRadius: '6px',
            color: '#e2e8f0', fontSize: '14px', resize: 'vertical',
          }}
        />
      </div>

      {error && <div style={{ color: '#ef4444', fontSize: '13px', marginBottom: '12px' }}>{error}</div>}

      <button className="btn btn-success" onClick={handleApprove} disabled={loading} style={{ padding: '10px 28px' }}>
        {loading ? 'Saving...' : 'Approve & Save'}
      </button>
    </div>
  )
}
