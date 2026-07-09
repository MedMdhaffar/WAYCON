import { useState, useEffect } from 'react'

const FIELDS = ['top', 'bottom', 'shoes', 'full']

<<<<<<< HEAD
function PersonDescription({ personId, index, paths, description }) {
  return (
    <div style={{ background: '#0f1117', border: '1px solid #1e2330', borderRadius: 8, padding: 12 }}>
      <div style={{ fontSize: 14, fontWeight: 600, color: '#e2e8f0', marginBottom: 10 }}>
        Person {index + 1}
      </div>
      {paths.length > 0 && (
        <div style={{ display: 'flex', gap: 8, marginBottom: 12, overflowX: 'auto', paddingBottom: 4 }}>
          {paths.map((p, i) => (
            <img
              key={p + i}
              src={`/api/images?path=${encodeURIComponent(p)}`}
              alt={`${personId} best ${i + 1}`}
              style={{ height: 100, width: 'auto', borderRadius: 6, border: '1px solid #1e2330', flexShrink: 0 }}
=======
function ClusterClothingCard({ clusterId, crops, clothing }) {
  const structured = clothing?.structured ?? clothing ?? {}
  const labelStyle = { fontSize: '12px', color: '#94a3b8', marginBottom: '5px', display: 'block', textTransform: 'uppercase', letterSpacing: '0.05em' }
  const valueStyle = {
    width: '100%', padding: '7px 11px', background: '#0f1117',
    border: '1px solid #1e2330', borderRadius: '6px',
    color: '#e2e8f0', fontSize: '14px', minHeight: '34px',
  }

  return (
    <div style={{ border: '1px solid #1e2330', borderRadius: '6px', padding: '12px', background: '#11151f' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
        <div style={{ fontSize: '14px', fontWeight: 600, color: '#e2e8f0' }}>Cluster {clusterId}</div>
        <div style={{ fontSize: '12px', color: '#64748b' }}>{crops.length} crops</div>
      </div>

      {crops.length > 0 && (
        <div style={{ display: 'flex', gap: '8px', marginBottom: '14px', overflowX: 'auto', paddingBottom: '4px' }}>
          {crops.map((p, i) => (
            <img
              key={p + i}
              src={`/api/images?path=${encodeURIComponent(p)}`}
              alt={`cluster ${clusterId} crop ${i + 1}`}
              style={{ height: '110px', width: 'auto', borderRadius: '6px', border: '1px solid #1e2330', flexShrink: 0 }}
>>>>>>> Khalifa_branch
            />
          ))}
        </div>
      )}
<<<<<<< HEAD
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, fontSize: 13 }}>
        {FIELDS.map(f => (
          <div key={f} style={f === 'full' ? { gridColumn: '1 / -1' } : {}}>
            <div style={{ color: '#64748b', textTransform: 'uppercase', fontSize: 11, marginBottom: 3 }}>{f}</div>
            <div style={{ color: '#cbd5e1' }}>{description?.[f] || 'unknown'}</div>
=======

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
        {FIELDS.map(f => (
          <div key={f} style={f === 'full' ? { gridColumn: '1 / -1' } : {}}>
            <label style={labelStyle}>{f}</label>
            <div style={valueStyle}>{structured[f] || '-'}</div>
>>>>>>> Khalifa_branch
          </div>
        ))}
      </div>
    </div>
  )
}

<<<<<<< HEAD
export default function ClothingPanel({ bestBodyCrops, bestBodyCropsByPerson = {}, clothingStructured, clothingByPerson = {}, onChange }) {
  const [values, setValues] = useState({ top: '', bottom: '', shoes: '', full: '' })
  const personIds = Object.keys(clothingByPerson).length
    ? Object.keys(clothingByPerson)
    : Object.keys(bestBodyCropsByPerson)
=======
export default function ClothingPanel({ bestBodyCrops, clothingStructured, perClusterBestBodyCrops = {}, perClusterClothing = {}, onChange }) {
  const [values, setValues] = useState({ top: '', bottom: '', shoes: '', full: '' })
  const clusterIds = Object.keys(perClusterBestBodyCrops).sort((a, b) => Number(a) - Number(b))
>>>>>>> Khalifa_branch

  useEffect(() => {
    if (clothingStructured && Object.keys(clothingStructured).length) {
      setValues({
        top: clothingStructured.top ?? '',
        bottom: clothingStructured.bottom ?? '',
        shoes: clothingStructured.shoes ?? '',
        full: clothingStructured.full ?? '',
      })
    }
  }, [clothingStructured])

  const update = (field, val) => {
    const next = { ...values, [field]: val }
    setValues(next)
    onChange?.(next)
  }

  const inputStyle = {
    width: '100%', padding: '7px 11px', background: '#0f1117',
    border: '1px solid #1e2330', borderRadius: '6px',
    color: '#e2e8f0', fontSize: '14px',
  }
  const labelStyle = { fontSize: '12px', color: '#94a3b8', marginBottom: '5px', display: 'block', textTransform: 'uppercase', letterSpacing: '0.05em' }

  if (clusterIds.length > 1 || (clusterIds.length === 1 && Object.keys(perClusterClothing).length)) {
    return (
      <div className="card">
        <div className="card-title">Clothing Description</div>
        <div style={{ display: 'grid', gap: '12px' }}>
          {clusterIds.map(cid => (
            <ClusterClothingCard
              key={cid}
              clusterId={cid}
              crops={perClusterBestBodyCrops[cid] ?? []}
              clothing={perClusterClothing[cid] ?? {}}
            />
          ))}
        </div>
      </div>
    )
  }

  return (
    <div className="card">
      <div className="card-title">Review generated descriptions</div>

      {personIds.length > 0 && (
        <div style={{ display: 'grid', gap: 12 }}>
          {personIds.map((personId, i) => (
            <PersonDescription
              key={personId}
              personId={personId}
              index={i}
              paths={bestBodyCropsByPerson[personId] ?? []}
              description={clothingByPerson[personId] ?? {}}
            />
          ))}
        </div>
      )}

      {personIds.length === 0 && bestBodyCrops.length > 0 && (
        <div style={{ display: 'flex', gap: '8px', marginBottom: '20px', overflowX: 'auto', paddingBottom: '4px' }}>
          {bestBodyCrops.map((p, i) => (
            <img
              key={i}
              src={`/api/images?path=${encodeURIComponent(p)}`}
              alt={`best ${i + 1}`}
              style={{ height: '120px', width: 'auto', borderRadius: '6px', border: '1px solid #1e2330', flexShrink: 0 }}
            />
          ))}
        </div>
      )}

      {personIds.length === 0 && <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
        {FIELDS.map(f => (
          <div key={f} style={f === 'full' ? { gridColumn: '1 / -1' } : {}}>
            <label style={labelStyle}>{f}</label>
            <input style={inputStyle} value={values[f]} onChange={e => update(f, e.target.value)} placeholder={`Describe ${f}…`} />
          </div>
        ))}
      </div>}
    </div>
  )
}
