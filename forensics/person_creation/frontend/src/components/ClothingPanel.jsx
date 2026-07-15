const FIELDS = ['top', 'bottom', 'shoes', 'full']

function isInternalLabel(value) {
  return /cluster[_\s-]?\d+/i.test(value || '')
}

function ClusterClothingCard({ clusterId, crops, clothing, profile }) {
  const structured = clothing?.structured ?? clothing ?? {}
  const title = !isInternalLabel(profile?.name)
    ? profile?.name
    : (!isInternalLabel(profile?.id) ? profile?.id : 'Pending identity')
  const labelStyle = { fontSize: '12px', color: '#94a3b8', marginBottom: '5px', display: 'block', textTransform: 'uppercase', letterSpacing: '0.05em' }
  const valueStyle = {
    width: '100%', padding: '7px 11px', background: '#0f1117',
    border: '1px solid #1e2330', borderRadius: '6px',
    color: '#e2e8f0', fontSize: '14px', minHeight: '34px',
  }

  return (
    <div style={{ border: '1px solid #1e2330', borderRadius: '6px', padding: '12px', background: '#11151f' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
        <div>
          <div style={{ fontSize: '14px', fontWeight: 600, color: '#e2e8f0' }}>{title}</div>
          {profile?.id && !isInternalLabel(profile.id) && (
            <div style={{ fontSize: '12px', color: '#64748b' }}>{profile.id}</div>
          )}
        </div>
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
            />
          ))}
        </div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
        {FIELDS.map(f => (
          <div key={f} style={f === 'full' ? { gridColumn: '1 / -1' } : {}}>
            <label style={labelStyle}>{f}</label>
            <div style={valueStyle}>{structured[f] || '-'}</div>
          </div>
        ))}
      </div>
    </div>
  )
}

export default function ClothingPanel({ bestBodyCrops, clothingStructured, perClusterBestBodyCrops = {}, perClusterClothing = {}, clusterProfiles = {} }) {
  const clusterIds = Object.keys(perClusterBestBodyCrops).sort((a, b) => Number(a) - Number(b))

  const labelStyle = { fontSize: '12px', color: '#94a3b8', marginBottom: '5px', display: 'block', textTransform: 'uppercase', letterSpacing: '0.05em' }
  const valueStyle = {
    width: '100%', padding: '7px 11px', background: '#0f1117',
    border: '1px solid #1e2330', borderRadius: '6px',
    color: '#e2e8f0', fontSize: '14px', minHeight: '34px',
  }

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
              profile={clusterProfiles[cid] ?? clusterProfiles[Number(cid)] ?? {}}
            />
          ))}
        </div>
      </div>
    )
  }

  return (
    <div className="card">
      <div className="card-title">Clothing Description</div>

      {bestBodyCrops.length > 0 && (
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

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
        {FIELDS.map(f => (
          <div key={f} style={f === 'full' ? { gridColumn: '1 / -1' } : {}}>
            <label style={labelStyle}>{f}</label>
            <div style={valueStyle}>{clothingStructured[f] || '-'}</div>
          </div>
        ))}
      </div>
    </div>
  )
}
