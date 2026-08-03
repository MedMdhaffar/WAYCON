import { useState } from 'react'
import SafeImage from './SafeImage.jsx'

function CropThumb({ crop, jobId, cropType, onDeleted, mediaVersion }) {
  const [deleting, setDeleting] = useState(false)

  const sharp = crop.sharpness?.toFixed(1) ?? '?'
  const frame = crop.frame_idx ?? '?'

  const handleDelete = async (e) => {
    e.stopPropagation()
    if (!jobId) return
    setDeleting(true)
    try {
      await fetch(`/api/person/crop/${jobId}`, {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: crop.path, crop_type: cropType }),
      })
      onDeleted()
    } catch (_) {
      setDeleting(false)
    }
  }

  return (
    <div style={{
      position: 'relative',
      border: '1px solid #1e2330',
      borderRadius: '8px',
      overflow: 'hidden',
      background: '#0f1117',
      opacity: deleting ? 0.4 : 1,
      transition: 'opacity 0.15s',
    }}>
      <SafeImage
        path={crop.path}
        version={mediaVersion}
        alt={`${cropType} crop`}
        placeholder="Unavailable"
        style={{ width: '100%', aspectRatio: '1', objectFit: 'cover', display: 'grid', placeItems: 'center', color: '#64748b', fontSize: 11 }}
      />

      {/* Delete button — visible on hover via CSS group */}
      <button
        onClick={handleDelete}
        disabled={deleting}
        title="Delete crop"
        style={{
          position: 'absolute', top: '4px', right: '4px',
          width: '22px', height: '22px', borderRadius: '50%',
          background: 'rgba(239,68,68,0.9)', border: 'none',
          color: '#fff', fontSize: '12px', fontWeight: 700,
          cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center',
          lineHeight: 1,
        }}
      >
        ×
      </button>

      <div style={{ padding: '4px 6px', display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: '#64748b' }}>
        <span>f{frame}</span>
        <span style={{ color: Number(sharp) >= 80 ? '#22c55e' : Number(sharp) >= 50 ? '#f59e0b' : '#ef4444' }}>⬡{sharp}</span>
      </div>
    </div>
  )
}

export default function CropsGrid({ jobId, bodyCrops, faceCrops, onDeleted, mediaVersion }) {
  const [subTab, setSubTab] = useState('body')

  const crops = subTab === 'body' ? bodyCrops : faceCrops
  const cropType = subTab === 'body' ? 'body' : 'face'

  const subtabStyle = (t) => ({
    padding: '6px 14px', borderRadius: '6px', border: 'none', cursor: 'pointer', fontSize: '13px',
    background: subTab === t ? '#7c9ef820' : 'transparent',
    color: subTab === t ? '#7c9ef8' : '#64748b',
  })

  return (
    <div className="card">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
        <div className="card-title" style={{ marginBottom: 0 }}>Crops</div>
        <div style={{ display: 'flex', gap: '4px' }}>
          <button style={subtabStyle('body')} onClick={() => setSubTab('body')}>
            Body ({bodyCrops.length})
          </button>
          <button style={subtabStyle('face')} onClick={() => setSubTab('face')}>
            Face ({faceCrops.length})
          </button>
        </div>
      </div>

      {crops.length === 0 ? (
        <div style={{ textAlign: 'center', color: '#475569', padding: '40px', fontSize: '14px' }}>
          {subTab === 'body' ? 'Body' : 'Face'} crops will appear here as the pipeline runs…
        </div>
      ) : (
        <>
          <div style={{ fontSize: '12px', color: '#475569', marginBottom: '12px' }}>
            Click <strong style={{ color: '#ef4444' }}>×</strong> on any crop to permanently delete it.
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(130px, 1fr))', gap: '10px' }}>
            {crops.map((c, i) => (
              <CropThumb key={`${mediaVersion}-${c.path}`} crop={c} jobId={jobId} cropType={cropType} onDeleted={onDeleted} mediaVersion={mediaVersion} />
            ))}
          </div>
        </>
      )}
    </div>
  )
}
