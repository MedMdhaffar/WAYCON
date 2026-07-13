export default function LiveStreamStats({ snapshot }) {
  if (snapshot?.source_type !== 'live_camera') return null
  const stats = snapshot.stream_stats ?? {}
  const values = [
    ['Stream opened', stats.stream_opened == null ? 'Connecting…' : stats.stream_opened ? 'Yes' : 'No'],
    ['Frames read', stats.frames_read ?? '—'],
    ['Frames processed', stats.frames_processed ?? '—'],
    ['Dropped frames', stats.frames_dropped ?? '—'],
    ['Duration', `${stats.duration_seconds ?? snapshot.duration_seconds ?? '—'} s`],
  ]

  return (
    <div className="card">
      <div className="card-title">Live Stream</div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: 10 }}>
        {values.map(([label, value]) => (
          <div key={label} style={{ padding: 12, borderRadius: 6, background: '#0f1117', border: '1px solid #1e2330' }}>
            <div style={{ color: '#64748b', fontSize: 11, marginBottom: 4 }}>{label}</div>
            <div style={{ color: '#e2e8f0', fontSize: 16, fontWeight: 600 }}>{value}</div>
          </div>
        ))}
      </div>
      {snapshot.stream_report_path && (
        <div style={{ marginTop: 12, color: '#94a3b8', fontSize: 12 }}>
          Stream report: <code>{snapshot.stream_report_path}</code>
        </div>
      )}
      {(stats.warnings ?? []).map((warning, index) => (
        <div key={index} style={{ marginTop: 8, color: '#f59e0b', fontSize: 12 }}>{warning}</div>
      ))}
    </div>
  )
}
