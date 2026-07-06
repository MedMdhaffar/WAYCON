import { useState } from 'react'

const DEFAULT_VIDEOS = ['']

export default function StartForm({ onStart }) {
  const [name, setName] = useState('session_1')
  const [videos, setVideos] = useState(DEFAULT_VIDEOS)
  const [outputDir, setOutputDir] = useState('forensics/person_db/session_1')
  const [everyN, setEveryN] = useState(5)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [pathErrors, setPathErrors] = useState([])

  const addVideo = () => setVideos(v => [...v, ''])
  const removeVideo = (i) => setVideos(v => v.filter((_, idx) => idx !== i))
  const updateVideo = (i, val) => setVideos(v => v.map((x, idx) => idx === i ? val : x))

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    setPathErrors([])
    setLoading(true)
    try {
      const res = await fetch('/api/person/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name,
          video_paths: videos.filter(v => v.trim()),
          output_dir: outputDir,
          every_n: everyN,
        }),
      })
      const data = await res.json()
      if (!res.ok || data.error) {
        if (Array.isArray(data.details)) setPathErrors(data.details)
        throw new Error(data.error || `HTTP ${res.status}`)
      }
      onStart(data.job_id)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const inputStyle = {
    width: '100%', padding: '8px 12px', background: '#0f1117',
    border: '1px solid #1e2330', borderRadius: '6px',
    color: '#e2e8f0', fontSize: '14px',
  }
  const labelStyle = { fontSize: '13px', color: '#94a3b8', marginBottom: '6px', display: 'block' }

  return (
    <div className="card">
      <div className="card-title">Process video automatically</div>
      <div style={{ fontSize: 13, color: '#94a3b8', lineHeight: 1.5, marginBottom: 16 }}>
        Process a video and automatically detect people, match faces to bodies, and generate descriptions.
      </div>
      <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>

        <div>
          <label style={labelStyle}>Session/Profile name</label>
          <input style={inputStyle} value={name} onChange={e => setName(e.target.value)} required placeholder="session_1" />
        </div>

        <div>
          <label style={labelStyle}>Video path</label>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {videos.map((v, i) => (
              <div key={i} style={{ display: 'flex', gap: '8px' }}>
                <input
                  style={{ ...inputStyle, flex: 1 }}
                  value={v}
                  onChange={e => updateVideo(i, e.target.value)}
                  placeholder="/mnt/c/Users/.../clip.mp4"
                />
                {videos.length > 1 && (
                  <button type="button" className="btn btn-danger" onClick={() => removeVideo(i)} style={{ padding: '8px 12px' }}>✕</button>
                )}
              </div>
            ))}
            <button type="button" className="btn btn-ghost" onClick={addVideo} style={{ alignSelf: 'flex-start' }}>+ Add video</button>
          </div>
        </div>

        <div>
          <label style={labelStyle}>Output directory</label>
          <input style={inputStyle} value={outputDir} onChange={e => setOutputDir(e.target.value)} placeholder="forensics/person_db/session_1" />
        </div>

        <div>
          <label style={labelStyle}>Process every N frames: <strong style={{ color: '#7c9ef8' }}>{everyN}</strong></label>
          <input type="range" min={1} max={30} value={everyN} onChange={e => setEveryN(Number(e.target.value))}
            style={{ width: '100%', accentColor: '#7c9ef8' }} />
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', color: '#64748b', marginTop: '4px' }}>
            <span>1 (dense)</span><span>30 (sparse)</span>
          </div>
        </div>

        {error && <div style={{ color: '#ef4444', fontSize: '13px', padding: '8px 12px', background: '#1e1015', borderRadius: '6px' }}>{error}</div>}
        {pathErrors.length > 0 && (
          <ul style={{ color: '#ef4444', fontSize: 13, paddingLeft: 22, margin: 0, lineHeight: 1.5 }}>
            {pathErrors.map((d, i) => (
              <li key={i} style={{ marginBottom: 4 }}>
                <code style={{ fontFamily: 'monospace' }}>{d.input}</code>
                {d.normalized && d.normalized !== d.input && (
                  <> → <code style={{ fontFamily: 'monospace' }}>{d.normalized}</code></>
                )}
                : {d.reason}
              </li>
            ))}
          </ul>
        )}

        <button type="submit" className="btn btn-primary" disabled={loading} style={{ alignSelf: 'flex-start', padding: '10px 24px' }}>
          {loading ? 'Starting...' : 'Process video automatically'}
        </button>
      </form>
    </div>
  )
}
