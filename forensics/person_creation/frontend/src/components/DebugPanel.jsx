import { useState, useEffect, useCallback } from 'react'

const CARD_STYLE = { padding: 12, borderRadius: 6, background: '#0f1117', border: '1px solid #1e2330' }
const PRE_STYLE = {
  background: '#0f1117', border: '1px solid #1e2330', borderRadius: 6, padding: 12,
  fontSize: 12, color: '#94a3b8', overflowX: 'auto', maxHeight: 320, overflowY: 'auto',
}

function JsonBlock({ data }) {
  return <pre style={PRE_STYLE}>{JSON.stringify(data, null, 2)}</pre>
}

function Section({ title, children, onRefresh }) {
  return (
    <div className="card">
      <div className="card-title" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span>{title}</span>
        {onRefresh && (
          <button className="btn btn-ghost" onClick={onRefresh} style={{ padding: '4px 10px', fontSize: 12 }}>↻ Refresh</button>
        )}
      </div>
      {children}
    </div>
  )
}

export default function DebugPanel() {
  const [system, setSystem] = useState(null)
  const [cameraEvents, setCameraEvents] = useState([])
  const [segments, setSegments] = useState([])
  const [clothingJobs, setClothingJobs] = useState([])
  const [jobs, setJobs] = useState([])
  const [error, setError] = useState('')

  const fetchAll = useCallback(async () => {
    try {
      const [sysRes, evRes, segRes, cjRes, jobsRes] = await Promise.all([
        fetch('/api/debug/system'),
        fetch('/api/debug/camera-events?limit=50'),
        fetch('/api/segments?limit=25'),
        fetch('/api/clothing-jobs?limit=25'),
        fetch('/api/debug/jobs'),
      ])
      setSystem(await sysRes.json())
      setCameraEvents((await evRes.json()).camera_events ?? [])
      setSegments((await segRes.json()).segments ?? [])
      setClothingJobs((await cjRes.json()).clothing_jobs ?? [])
      setJobs((await jobsRes.json()).jobs ?? [])
      setError('')
    } catch (err) {
      setError(err.message)
    }
  }, [])

  useEffect(() => { fetchAll() }, [fetchAll])

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {error && <div style={{ color: '#ef4444', fontSize: 13 }}>{error}</div>}

      <Section title="System / Process" onRefresh={fetchAll}>
        {system ? (
          <>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 10, marginBottom: 10 }}>
              <div style={CARD_STYLE}>
                <div style={{ color: '#64748b', fontSize: 11 }}>Threads</div>
                <div style={{ color: '#e2e8f0', fontSize: 16, fontWeight: 600 }}>{system.thread_count}</div>
              </div>
              <div style={CARD_STYLE}>
                <div style={{ color: '#64748b', fontSize: 11 }}>CUDA</div>
                <div style={{ color: system.torch?.cuda_available ? '#22c55e' : '#ef4444', fontSize: 16, fontWeight: 600 }}>
                  {system.torch?.cuda_available ? 'Available' : 'Unavailable'}
                </div>
              </div>
              <div style={CARD_STYLE}>
                <div style={{ color: '#64748b', fontSize: 11 }}>GPU mem allocated</div>
                <div style={{ color: '#e2e8f0', fontSize: 16, fontWeight: 600 }}>{system.torch?.memory_allocated_mb ?? '—'} MB</div>
              </div>
              <div style={CARD_STYLE}>
                <div style={{ color: '#64748b', fontSize: 11 }}>Person detector loaded</div>
                <div style={{ color: system.person_detector_loaded ? '#22c55e' : '#94a3b8', fontSize: 16, fontWeight: 600 }}>
                  {String(system.person_detector_loaded)}
                </div>
              </div>
              <div style={CARD_STYLE}>
                <div style={{ color: '#64748b', fontSize: 11 }}>Realtime cameras running</div>
                <div style={{ color: '#e2e8f0', fontSize: 16, fontWeight: 600 }}>{system.realtime_cameras?.length ?? 0}</div>
              </div>
            </div>
            <JsonBlock data={system} />
          </>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>Loading…</div>}
      </Section>

      <Section title="Camera Events (connect/disconnect/reconnect)" onRefresh={fetchAll}>
        <JsonBlock data={cameraEvents} />
      </Section>

      <Section title="Segments (core-detection state machine)" onRefresh={fetchAll}>
        <JsonBlock data={segments} />
      </Section>

      <Section title="Clothing Jobs (async VLM enrichment)" onRefresh={fetchAll}>
        <JsonBlock data={clothingJobs} />
      </Section>

      <Section title="Offline Jobs (this process, since startup)" onRefresh={fetchAll}>
        <JsonBlock data={jobs} />
      </Section>
    </div>
  )
}
