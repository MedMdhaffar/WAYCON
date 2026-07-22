import { useState, useEffect, useCallback, useRef } from 'react'

const CARD_STYLE = { padding: 12, borderRadius: 6, background: '#0f1117', border: '1px solid #1e2330' }
const LABEL_STYLE = { color: '#64748b', fontSize: 11, marginBottom: 4 }
const VALUE_STYLE = { color: '#e2e8f0', fontSize: 16, fontWeight: 600 }

function connectionColor(state) {
  if (state === 'connected') return '#22c55e'
  if (state === 'reconnecting') return '#f59e0b'
  return '#ef4444'
}

export default function RealtimeMonitor({ cameraId, onStopped }) {
  const [stats, setStats] = useState(null)
  const [error, setError] = useState('')
  const [stopping, setStopping] = useState(false)
  const pollRef = useRef(null)

  const poll = useCallback(async () => {
    try {
      const res = await fetch(`/api/realtime/status/${cameraId}`)
      const data = await res.json()
      if (!res.ok || data.error) {
        setError(data.error || `HTTP ${res.status}`)
        clearInterval(pollRef.current)
        return
      }
      setStats(data)
      setError('')
    } catch (err) {
      setError(err.message)
    }
  }, [cameraId])

  useEffect(() => {
    poll()
    pollRef.current = setInterval(poll, 2000)
    return () => clearInterval(pollRef.current)
  }, [poll])

  const handleStop = async () => {
    setStopping(true)
    try {
      await fetch(`/api/realtime/stop/${cameraId}`, { method: 'POST' })
      clearInterval(pollRef.current)
      onStopped?.()
    } catch (err) {
      setError(err.message)
    } finally {
      setStopping(false)
    }
  }

  if (error) {
    return (
      <div className="card">
        <div className="card-title">Realtime Camera — {cameraId}</div>
        <div style={{ color: '#ef4444', fontSize: 13 }}>{error}</div>
      </div>
    )
  }

  if (!stats) {
    return (
      <div className="card">
        <div className="card-title">Realtime Camera — {cameraId}</div>
        <div style={{ color: '#94a3b8', fontSize: 13 }}>Loading…</div>
      </div>
    )
  }

  const values = [
    ['Connection', stats.connection_state ?? '—', connectionColor(stats.connection_state)],
    ['Presence', stats.presence_state ?? '—', stats.presence_state === 'present' ? '#22c55e' : '#94a3b8'],
    ['Segments processed', stats.segments_processed ?? 0],
    ['Segments failed', stats.segments_failed ?? 0],
    ['Segment queue depth', stats.segment_queue_depth ?? 0],
    ['Reconnects', stats.reconnect_count ?? 0],
    ['Frames seen', stats.frames_seen ?? 0],
    ['Samples taken', stats.samples_taken ?? 0],
  ]

  return (
    <div className="card">
      <div className="card-title" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span>Realtime Camera — {stats.camera_id} ({stats.camera_uri_masked})</span>
        <button className="btn btn-danger" onClick={handleStop} disabled={stopping} style={{ padding: '6px 14px', fontSize: 13 }}>
          {stopping ? 'Stopping…' : '■ Stop Capture'}
        </button>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 10 }}>
        {values.map(([label, value, color]) => (
          <div key={label} style={CARD_STYLE}>
            <div style={LABEL_STYLE}>{label}</div>
            <div style={{ ...VALUE_STYLE, color: color ?? VALUE_STYLE.color }}>{value}</div>
          </div>
        ))}
      </div>
      {stats.last_disconnect_reason && (
        <div style={{ marginTop: 12, color: '#f59e0b', fontSize: 12 }}>
          Last disconnect: {stats.last_disconnect_reason}
        </div>
      )}
    </div>
  )
}
