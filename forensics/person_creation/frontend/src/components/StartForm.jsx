import { useState } from 'react'
import { safeErrorMessage } from '../liveJob.js'

const DEFAULT_VIDEOS = ['']

export default function StartForm({ onStart, activeJob = false }) {
  const [name, setName] = useState('Malek')
  const [inputType, setInputType] = useState('video_file')
  const [videos, setVideos] = useState(DEFAULT_VIDEOS)
  const [cameraUri, setCameraUri] = useState('')
  const [cameraId, setCameraId] = useState('')
  const [durationSeconds, setDurationSeconds] = useState(30)
  const [outputDir, setOutputDir] = useState('forensics/person_db/malek')
  const [everyN, setEveryN] = useState(5)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [pathErrors, setPathErrors] = useState([])

  const addVideo = () => setVideos(v => [...v, ''])
  const removeVideo = (i) => setVideos(v => v.filter((_, idx) => idx !== i))
  const updateVideo = (i, val) => setVideos(v => v.map((x, idx) => idx === i ? val : x))

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (activeJob) return
    setError('')
    setPathErrors([])
    setLoading(true)
    try {
      const sourcePayload = inputType === 'camera_uri'
        ? {
            input_type: 'camera_uri',
            camera_uri: cameraUri.trim(),
            camera_id: cameraId.trim() || undefined,
            duration_seconds: Number(durationSeconds),
          }
        : {
            input_type: 'video_file',
            video_paths: videos.filter(v => v.trim()),
          }
      const res = await fetch('/api/person/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name,
          ...sourcePayload,
          output_dir: outputDir,
          ...(inputType === 'video_file' ? { every_n: everyN } : {}),
        }),
      })
      const data = await res.json()
      if (!res.ok || data.error) {
        if (inputType === 'video_file' && Array.isArray(data.details)) setPathErrors(data.details)
        throw new Error(data.error || `HTTP ${res.status}`)
      }
      if (inputType === 'camera_uri') setCameraUri('')
      onStart(data.job_id, inputType === 'camera_uri' ? 'live_camera' : 'video_file')
    } catch (err) {
      setError(safeErrorMessage(err, 'Unable to start the job.'))
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
      <div className="card-title">New Person Profile</div>
      <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>

        {inputType === 'video_file' && <div>
          <label style={labelStyle}>Person Name</label>
          <input style={inputStyle} value={name} onChange={e => setName(e.target.value)} required placeholder="Malek" />
        </div>}

        <div>
          <label style={labelStyle}>Input Source</label>
          <select style={inputStyle} value={inputType} onChange={e => setInputType(e.target.value)}>
            <option value="video_file">Video file/path</option>
            <option value="camera_uri">Live camera URI</option>
          </select>
        </div>

        {inputType === 'video_file' ? (
          <div>
            <label style={labelStyle}>Video Clips</label>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              {videos.map((v, i) => (
                <div key={i} style={{ display: 'flex', gap: '8px' }}>
                  <input
                    style={{ ...inputStyle, flex: 1 }}
                    value={v}
                    onChange={e => updateVideo(i, e.target.value)}
                    placeholder="/home/user/video.mp4"
                    required
                  />
                  {videos.length > 1 && (
                    <button type="button" className="btn btn-danger" onClick={() => removeVideo(i)} style={{ padding: '8px 12px' }}>✕</button>
                  )}
                </div>
              ))}
              <button type="button" className="btn btn-ghost" onClick={addVideo} style={{ alignSelf: 'flex-start' }}>+ Add video</button>
            </div>
          </div>
        ) : (
          <>
            <div>
              <label style={labelStyle}>Camera URI</label>
              <input
                style={inputStyle}
                value={cameraUri}
                onChange={e => setCameraUri(e.target.value)}
                placeholder="rtsp://user:password@camera:554/Streaming/Channels/101"
                required
              />
              <div style={{ color: '#64748b', fontSize: 12, marginTop: 6 }}>
                Used only to connect. The URI is cleared from this form after submission.
              </div>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
              <div>
                <label style={labelStyle}>Camera ID (optional)</label>
                <input style={inputStyle} value={cameraId} onChange={e => setCameraId(e.target.value)} placeholder="103" />
              </div>
              <div>
                <label style={labelStyle}>Processing window duration (seconds)</label>
                <input
                  style={inputStyle}
                  type="number"
                  min={5}
                  max={300}
                  value={durationSeconds}
                  onChange={e => setDurationSeconds(Number(e.target.value))}
                  required
                />
                <div style={{ color: '#64748b', fontSize: 12, marginTop: 6 }}>
                  The camera continues running until Stop is pressed.
                </div>
              </div>
            </div>
          </>
        )}

        <div>
          <label style={labelStyle}>Output Directory</label>
          <input style={inputStyle} value={outputDir} onChange={e => setOutputDir(e.target.value)} placeholder="forensics/person_db/malek" />
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

        {activeJob && (
          <div style={{ color: '#f59e0b', fontSize: 13 }}>
            Finish the active job before starting another one.
          </div>
        )}
        <button type="submit" className="btn btn-primary" disabled={loading || activeJob} style={{ alignSelf: 'flex-start', padding: '10px 24px' }}>
          {loading ? 'Starting…' : '▶ Start Pipeline'}
        </button>
      </form>
    </div>
  )
}
