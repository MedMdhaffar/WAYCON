import SafeImage from './SafeImage.jsx'

const FIELDS = ['top', 'bottom', 'shoes', 'full']

function isInternalLabel(value) {
  return /cluster[_\s-]?\d+/i.test(value || '')
}

function CropStrip({ crops, prefix, height }) {
  if (!crops.length) return null
  return (
    <div style={{ display: 'flex', gap: 8, marginBottom: 14, overflowX: 'auto', paddingBottom: 4 }}>
      {crops.map((path, index) => (
        <SafeImage
          key={`${path}-${index}`}
          path={path}
          alt={`${prefix} crop ${index + 1}`}
          placeholder="Unavailable"
          style={{ height, width: 'auto', minWidth: 72, borderRadius: 6, border: '1px solid #1e2330', flexShrink: 0, display: 'grid', placeItems: 'center', color: '#64748b', fontSize: 11 }}
        />
      ))}
    </div>
  )
}

function ClothingFields({ result }) {
  if (result?.status === 'failed') {
    return (
      <div className="clothing-status-failed">
        Clothing description unavailable for this identity.
      </div>
    )
  }
  const labelStyle = { fontSize: 12, color: '#94a3b8', marginBottom: 5, display: 'block', textTransform: 'uppercase' }
  const valueStyle = { width: '100%', padding: '7px 11px', background: '#0f1117', border: '1px solid #1e2330', borderRadius: 6, color: '#e2e8f0', fontSize: 14, minHeight: 34 }
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
      {FIELDS.map(field => (
        <div key={field} style={field === 'full' ? { gridColumn: '1 / -1' } : {}}>
          <label style={labelStyle}>{field}</label>
          <div style={valueStyle}>{result?.[field] || '-'}</div>
        </div>
      ))}
    </div>
  )
}

function ClusterClothingCard({ clusterId, crops, clothing, profile }) {
  const result = clothing?.structured ?? clothing ?? {}
  const title = !isInternalLabel(profile?.name)
    ? profile?.name
    : (!isInternalLabel(profile?.id) ? profile?.id : 'Pending identity')
  return (
    <div style={{ border: '1px solid #1e2330', borderRadius: 6, padding: 12, background: '#11151f' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
        <div>
          <div style={{ fontSize: 14, fontWeight: 600, color: '#e2e8f0' }}>{title}</div>
          {profile?.id && !isInternalLabel(profile.id) && <div style={{ fontSize: 12, color: '#64748b' }}>{profile.id}</div>}
        </div>
        <div style={{ fontSize: 12, color: '#64748b' }}>{crops.length} crops</div>
      </div>
      <CropStrip crops={crops} prefix={`cluster ${clusterId}`} height={110} />
      <ClothingFields result={result} />
    </div>
  )
}

export default function ClothingPanel({ bestBodyCrops, clothingStructured, perClusterBestBodyCrops = {}, perClusterClothing = {}, clusterProfiles = {} }) {
  const clusterIds = Object.keys(perClusterBestBodyCrops).sort((a, b) => Number(a) - Number(b))
  if (clusterIds.length > 1 || (clusterIds.length === 1 && Object.keys(perClusterClothing).length)) {
    return (
      <div className="card">
        <div className="card-title">Clothing Description</div>
        <div style={{ display: 'grid', gap: 12 }}>
          {clusterIds.map(clusterId => (
            <ClusterClothingCard
              key={clusterId}
              clusterId={clusterId}
              crops={perClusterBestBodyCrops[clusterId] ?? []}
              clothing={perClusterClothing[clusterId] ?? {}}
              profile={clusterProfiles[clusterId] ?? clusterProfiles[Number(clusterId)] ?? {}}
            />
          ))}
        </div>
      </div>
    )
  }
  return (
    <div className="card">
      <div className="card-title">Clothing Description</div>
      <CropStrip crops={bestBodyCrops} prefix="best" height={120} />
      <ClothingFields result={clothingStructured} />
    </div>
  )
}
