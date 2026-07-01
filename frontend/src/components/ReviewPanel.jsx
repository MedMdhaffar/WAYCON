import { useState } from 'react'

export default function ReviewPanel({ jobId, snapshot, clothingOverride, onDone, finalStatus }) {
  const [loading, setLoading] = useState(false)
  const [corrections, setCorrections] = useState('')
  const [error, setError] = useState('')

  const profile = snapshot?.profile ?? {}
  const appearance = profile.appearance ?? snapshot?.clothing_structured ?? {}

  const handleApprove = async () => {
    setError('')
    setLoading(true)
    try {
      const res = await fetch(`/api/person/approve/${jobId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          corrections:       corrections.trim() || null,
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
    return (
      <div className="card" style={{ borderColor: '#22c55e' }}>
        <div style={{ textAlign: 'center', padding: '24px' }}>
          <div style={{ fontSize: '48px', marginBottom: '12px' }}>✓</div>
          <div style={{ fontSize: '18px', fontWeight: 600, color: '#22c55e', marginBottom: '8px' }}>Profile Saved</div>
          <div style={{ fontSize: '14px', color: '#64748b' }}>
            {profile.name} — {profile.face_crop_count} face crops · profile.json written
          </div>
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

      <div style={{ marginBottom: '20px' }}>
        <div style={rowStyle}><span style={keyStyle}>Name</span><span style={valStyle}>{profile.name || '—'}</span></div>
        <div style={rowStyle}><span style={keyStyle}>Face crops</span><span style={valStyle}>{profile.face_crop_count ?? (snapshot?.quality_face_crops?.length ?? '—')}</span></div>
        <div style={rowStyle}><span style={keyStyle}>Associations</span><span style={valStyle}>{(snapshot?.associations ?? []).length}</span></div>
        <div style={rowStyle}><span style={keyStyle}>Top</span><span style={valStyle}>{appearance.top || '—'}</span></div>
        <div style={rowStyle}><span style={keyStyle}>Bottom</span><span style={valStyle}>{appearance.bottom || '—'}</span></div>
        <div style={rowStyle}><span style={keyStyle}>Shoes</span><span style={valStyle}>{appearance.shoes || '—'}</span></div>
        <div style={{ ...rowStyle, borderBottom: 'none' }}>
          <span style={keyStyle}>Full</span>
          <span style={{ ...valStyle, maxWidth: '60%', textAlign: 'right' }}>{appearance.full || '—'}</span>
        </div>
      </div>

      <div style={{ marginBottom: '16px' }}>
        <label style={{ fontSize: '13px', color: '#94a3b8', display: 'block', marginBottom: '6px' }}>Corrections / notes (optional)</label>
        <textarea
          value={corrections}
          onChange={e => setCorrections(e.target.value)}
          rows={3}
          placeholder="e.g. Clothing description is wrong — top should be black t-shirt…"
          style={{
            width: '100%', padding: '8px 12px', background: '#0f1117',
            border: '1px solid #1e2330', borderRadius: '6px',
            color: '#e2e8f0', fontSize: '14px', resize: 'vertical',
          }}
        />
      </div>

      {error && <div style={{ color: '#ef4444', fontSize: '13px', marginBottom: '12px' }}>{error}</div>}

      <button className="btn btn-success" onClick={handleApprove} disabled={loading} style={{ padding: '10px 28px' }}>
        {loading ? 'Saving…' : '✓ Approve & Save'}
      </button>
    </div>
  )
}
