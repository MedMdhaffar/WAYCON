function cleanName(item) {
  const isInternal = value => /cluster[_\s-]?\d+/i.test(value || '')
  if (item?.name && !isInternal(item.name)) return item.name
  if (item?.id && !isInternal(item.id)) return item.id
  return 'Pending identity'
}

export default function ReviewPanel({ snapshot, finalStatus }) {
  const perClusterProfiles = Object.values(snapshot?.per_cluster_profiles ?? {})
  const profile = snapshot?.profile ?? {}
  const profiles = perClusterProfiles.length ? perClusterProfiles : (profile.id ? [profile] : [])

  if (finalStatus !== 'done') {
    return (
      <div className="card">
        <div className="card-title">Result</div>
        <div style={{ textAlign: 'center', color: '#64748b', padding: '32px', fontSize: '14px' }}>
          {finalStatus === 'error'
            ? 'Pipeline failed — see the error above.'
            : 'The pipeline runs fully automatically. Saved profile(s) will appear here once it finishes.'}
        </div>
      </div>
    )
  }

  return (
    <div className="card" style={{ borderColor: '#22c55e' }}>
      <div style={{ textAlign: 'center', padding: '24px' }}>
        <div style={{ fontSize: '18px', fontWeight: 600, color: '#22c55e', marginBottom: '8px' }}>✓ Profile(s) Saved</div>
        <div style={{ fontSize: '14px', color: '#64748b' }}>{profiles.length} profile(s) written</div>

        {!!profiles.length && (
          <div style={{ marginTop: 18, display: 'grid', gap: 8, textAlign: 'left' }}>
            {profiles.map(p => (
              <div key={p.id} style={{ border: '1px solid #1e2330', borderRadius: 6, padding: 10, background: '#0f1117' }}>
                <h2 style={{ margin: 0, color: '#e2e8f0', fontSize: 18 }}>{cleanName(p)}</h2>
                <p style={{ margin: '3px 0 0', color: '#64748b', fontSize: 13 }}>{p.id}</p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
