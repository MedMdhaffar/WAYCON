import { useState } from 'react'

function MetaRow({ label, value, highlight }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', padding: '1px 0' }}>
      <span style={{ color: '#475569' }}>{label}</span>
      <span style={{ color: highlight ?? '#94a3b8', fontWeight: 500 }}>{value}</span>
    </div>
  )
}

function AssocPair({ assoc, jobId, onDeleted }) {
  const [deleting, setDeleting] = useState(false)

  const handleDelete = async () => {
    if (!jobId) return
    setDeleting(true)
    try {
      await fetch(`/api/person/crop/${jobId}`, {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: assoc.body_path, crop_type: 'body' }),
      })
      onDeleted()
    } catch (_) {
      setDeleting(false)
    }
  }

  const iouColor = assoc.iou_score >= 0.10 ? '#22c55e' : assoc.iou_score >= 0.05 ? '#f59e0b' : '#ef4444'
  const ratioColor = assoc.face_height_ratio >= 0.15 ? '#22c55e' : assoc.face_height_ratio >= 0.10 ? '#f59e0b' : '#ef4444'
  const sharpColor = assoc.body_sharpness >= 80 ? '#22c55e' : assoc.body_sharpness >= 50 ? '#f59e0b' : '#ef4444'

  return (
    <div style={{
      position: 'relative',
      background: '#0f1117',
      border: '1px solid #1e2330',
      borderRadius: '10px',
      padding: '10px',
      width: '220px',
      flexShrink: 0,
      opacity: deleting ? 0.3 : 1,
      transition: 'opacity 0.15s',
    }}>
      {/* Delete button */}
      <button
        onClick={handleDelete}
        disabled={deleting}
        title="Delete this pair"
        style={{
          position: 'absolute', top: '6px', right: '6px',
          width: '20px', height: '20px', borderRadius: '50%',
          background: 'rgba(239,68,68,0.9)', border: 'none',
          color: '#fff', fontSize: '13px', fontWeight: 700,
          cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center',
          zIndex: 1,
        }}
      >×</button>

      {/* Thumbnails row */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '10px' }}>
        <div style={{ textAlign: 'center', flex: '0 0 auto' }}>
          <img
            src={`/api/images?path=${encodeURIComponent(assoc.face_path)}`}
            alt="face"
            style={{ width: '64px', height: '64px', objectFit: 'cover', borderRadius: '6px', border: '1px solid #1e2330', display: 'block' }}
          />
          <div style={{ fontSize: '10px', color: '#475569', marginTop: '2px' }}>face</div>
        </div>
        <span style={{ color: '#334155', fontSize: '18px', flexShrink: 0 }}>→</span>
        <div style={{ textAlign: 'center', flex: '0 0 auto' }}>
          <img
            src={`/api/images?path=${encodeURIComponent(assoc.body_path)}`}
            alt="body"
            style={{ width: '64px', height: '84px', objectFit: 'cover', borderRadius: '6px', border: '1px solid #1e2330', display: 'block' }}
          />
          <div style={{ fontSize: '10px', color: '#475569', marginTop: '2px' }}>body</div>
        </div>
      </div>

      {/* Metadata */}
      <div style={{ borderTop: '1px solid #1e2330', paddingTop: '8px', display: 'flex', flexDirection: 'column', gap: '2px' }}>
        <MetaRow label="Frame" value={String(assoc.frame_idx).padStart(6, '0')} />
        <MetaRow label="Video" value={assoc.video_name ?? '—'} />
        <MetaRow label="IoU" value={assoc.iou_score?.toFixed(4) ?? '—'} highlight={iouColor} />
        <MetaRow label="Face/Body h" value={`${((assoc.face_height_ratio ?? 0) * 100).toFixed(1)}%`} highlight={ratioColor} />
        <MetaRow label="Face size" value={`${assoc.face_w}×${assoc.face_h}px`} />
        <MetaRow label="Body size" value={`${assoc.body_w}×${assoc.body_h}px`} />
        <MetaRow label="Body area" value={`${assoc.body_area?.toLocaleString() ?? '—'}px²`} />
        <MetaRow label="Sharpness" value={assoc.body_sharpness?.toFixed(1) ?? '—'} highlight={sharpColor} />
      </div>
    </div>
  )
}

export default function AssociationsView({ jobId, associations, onDeleted }) {
  if (!associations.length) return null

  return (
    <div className="card">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
        <div className="card-title" style={{ marginBottom: 0 }}>Face ↔ Body Associations ({associations.length})</div>
      </div>
      <div style={{ fontSize: '12px', color: '#475569', marginBottom: '14px' }}>
        Click <strong style={{ color: '#ef4444' }}>×</strong> to permanently delete a wrong pair.
        Colors: <span style={{ color: '#22c55e' }}>■</span> good &nbsp;
        <span style={{ color: '#f59e0b' }}>■</span> marginal &nbsp;
        <span style={{ color: '#ef4444' }}>■</span> weak
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '12px' }}>
        {associations.map((a, i) => (
          <AssocPair key={a.body_path + i} assoc={a} jobId={jobId} onDeleted={onDeleted} />
        ))}
      </div>
    </div>
  )
}
