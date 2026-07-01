import { useState, useCallback } from 'react'

/* ── single crop thumbnail with delete button ─────────────────────────── */
function CropThumb({ crop, type, size, selected, paired, jobId, onClick, onDeleted }) {
  const [deleting, setDeleting] = useState(false)
  const [w, h] = size

  const handleDelete = async (e) => {
    e.stopPropagation()
    if (!jobId) return
    setDeleting(true)
    try {
      await fetch(`/api/person/crop/${jobId}`, {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: crop.path, crop_type: type }),
      })
      onDeleted(crop.path, type)
    } catch (_) { setDeleting(false) }
  }

  const borderColor = selected ? '#a78bfa' : paired ? '#a78bfa44' : '#1e2330'

  return (
    <div
      style={{
        position: 'relative', cursor: paired ? 'default' : 'pointer',
        opacity: deleting ? 0.2 : paired ? 0.55 : 1,
        transition: 'opacity 0.15s',
        flexShrink: 0,
      }}
      onClick={!paired ? onClick : undefined}
    >
      <img
        src={`/api/images?path=${encodeURIComponent(crop.path)}`}
        alt={type}
        style={{
          width: w, height: h, objectFit: 'cover', borderRadius: '6px', display: 'block',
          border: `3px solid ${borderColor}`,
          transition: 'border-color 0.1s',
          boxSizing: 'border-box',
        }}
      />
      {/* paired checkmark */}
      {paired && (
        <div style={{
          position: 'absolute', inset: 0, display: 'flex', alignItems: 'center',
          justifyContent: 'center', borderRadius: '6px',
          background: 'rgba(167,139,250,0.18)', pointerEvents: 'none',
        }}>
          <span style={{ fontSize: '24px', color: '#a78bfa' }}>✓</span>
        </div>
      )}
      {/* delete */}
      <button
        onClick={handleDelete}
        style={{
          position: 'absolute', top: 3, right: 3,
          width: 20, height: 20, borderRadius: '50%',
          background: 'rgba(239,68,68,0.92)', border: 'none',
          color: '#fff', fontSize: 12, fontWeight: 700,
          cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center',
          lineHeight: 1,
        }}
      >×</button>
      {/* sharpness */}
      <div style={{ fontSize: 10, color: '#475569', textAlign: 'center', marginTop: 2 }}>
        ⬡{crop.sharpness?.toFixed(0) ?? '?'}
      </div>
    </div>
  )
}

/* ── one frame group card ─────────────────────────────────────────────── */
function FrameGroup({ group, jobId, onGroupChange, onDeleted }) {
  const [pairs, setPairs]           = useState([])   // [{face_path, body_path}]
  const [selectedFace, setSelected] = useState(null)
  const [faces, setFaces]           = useState(group.faces)
  const [bodies, setBodies]         = useState(group.bodies)
  const [skipped, setSkipped]       = useState(false)

  const pairedFaces  = new Set(pairs.map(p => p.face_path))
  const pairedBodies = new Set(pairs.map(p => p.body_path))

  const handleFaceClick = (path) => {
    if (pairedFaces.has(path)) return
    setSelected(prev => prev === path ? null : path)
  }

  const handleBodyClick = (path) => {
    if (!selectedFace || pairedBodies.has(path)) return
    const next = [...pairs, { face_path: selectedFace, body_path: path }]
    setPairs(next)
    setSelected(null)
    onGroupChange(group.frame_idx, group.video, next)
  }

  const handleUnlink = (facePath, bodyPath) => {
    const next = pairs.filter(p => !(p.face_path === facePath && p.body_path === bodyPath))
    setPairs(next)
    onGroupChange(group.frame_idx, group.video, next)
  }

  const handleDeletedLocal = (path, type) => {
    if (type === 'face') {
      setFaces(f => f.filter(x => x.path !== path))
      const next = pairs.filter(p => p.face_path !== path)
      setPairs(next); onGroupChange(group.frame_idx, group.video, next)
      if (selectedFace === path) setSelected(null)
    } else {
      setBodies(b => b.filter(x => x.path !== path))
      const next = pairs.filter(p => p.body_path !== path)
      setPairs(next); onGroupChange(group.frame_idx, group.video, next)
    }
    onDeleted(path, type)
  }

  const sectionLabel = (txt, count) => (
    <div style={{ fontSize: 11, color: '#475569', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
      {txt} <span style={{ color: '#64748b' }}>({count})</span>
    </div>
  )

  return (
    <div style={{
      background: '#0f1117',
      border: `1px solid ${skipped ? '#2a2f3d' : pairs.length > 0 ? '#a78bfa44' : '#1e2330'}`,
      borderRadius: 10, padding: 14, marginBottom: 12,
      opacity: skipped ? 0.45 : 1, transition: 'opacity 0.15s',
    }}>
      {/* header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
        <span style={{ fontSize: 12, color: '#64748b' }}>
          Frame <strong style={{ color: '#94a3b8', fontFamily: 'monospace' }}>
            {String(group.frame_idx).padStart(6, '0')}
          </strong>&nbsp;·&nbsp;{group.video_name}
          {selectedFace && <span style={{ color: '#a78bfa', marginLeft: 10 }}>← now click a body</span>}
        </span>
        <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
          {pairs.length > 0 && (
            <span style={{ fontSize: 11, color: '#a78bfa', background: '#a78bfa18', padding: '2px 8px', borderRadius: 12 }}>
              {pairs.length} pair{pairs.length > 1 ? 's' : ''}
            </span>
          )}
          <button onClick={() => setSkipped(s => !s)} style={{
            fontSize: 11, padding: '2px 10px', borderRadius: 4,
            border: '1px solid #1e2330', background: 'transparent',
            color: skipped ? '#7c9ef8' : '#475569', cursor: 'pointer',
          }}>{skipped ? 'Unskip' : 'Skip'}</button>
        </div>
      </div>

      {!skipped && (
        <>
          {/* FACES row */}
          {faces.length > 0 && (
            <div style={{ marginBottom: 12 }}>
              {sectionLabel('Faces', faces.length)}
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                {faces.map((face, i) => (
                  <CropThumb
                    key={face.path + i} crop={face} type="face" size={[90, 90]}
                    jobId={jobId}
                    selected={selectedFace === face.path}
                    paired={pairedFaces.has(face.path)}
                    onClick={() => handleFaceClick(face.path)}
                    onDeleted={handleDeletedLocal}
                  />
                ))}
              </div>
            </div>
          )}

          {/* BODIES row */}
          {bodies.length > 0 && (
            <div style={{ marginBottom: pairs.length > 0 ? 12 : 0 }}>
              {sectionLabel('Bodies', bodies.length)}
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                {bodies.map((body, i) => (
                  <CropThumb
                    key={body.path + i} crop={body} type="body" size={[80, 120]}
                    jobId={jobId}
                    selected={false}
                    paired={pairedBodies.has(body.path)}
                    onClick={() => handleBodyClick(body.path)}
                    onDeleted={handleDeletedLocal}
                  />
                ))}
              </div>
            </div>
          )}

          {/* PAIRS row */}
          {pairs.length > 0 && (
            <div style={{ borderTop: '1px solid #1e2330', paddingTop: 10 }}>
              {sectionLabel('Confirmed Pairs', pairs.length)}
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                {pairs.map((pair, i) => {
                  const fImg = faces.find(f => f.path === pair.face_path)
                  const bImg = bodies.find(b => b.path === pair.body_path)
                  return (
                    <div key={i} style={{
                      display: 'flex', alignItems: 'center', gap: 6,
                      background: '#13161f', border: '1px solid #a78bfa44',
                      borderRadius: 8, padding: '6px 10px',
                    }}>
                      {fImg && <img src={`/api/images?path=${encodeURIComponent(fImg.path)}`}
                        style={{ width: 40, height: 40, objectFit: 'cover', borderRadius: 4 }} />}
                      <span style={{ color: '#a78bfa', fontSize: 16 }}>→</span>
                      {bImg && <img src={`/api/images?path=${encodeURIComponent(bImg.path)}`}
                        style={{ width: 36, height: 52, objectFit: 'cover', borderRadius: 4 }} />}
                      <button onClick={() => handleUnlink(pair.face_path, pair.body_path)} style={{
                        fontSize: 11, padding: '2px 7px', borderRadius: 4,
                        border: '1px solid #ef444444', background: 'transparent',
                        color: '#ef4444', cursor: 'pointer', marginLeft: 4,
                      }}>unlink</button>
                    </div>
                  )
                })}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}

/* ── main panel ───────────────────────────────────────────────────────── */
export default function FramePairingPanel({ jobId, frameGroups, onSubmitted }) {
  const [groupPairs,   setGroupPairs]   = useState({})
  const [deletedPaths, setDeletedPaths] = useState([])
  const [loading,      setLoading]      = useState(false)
  const [error,        setError]        = useState('')

  const handleGroupChange = useCallback((frameIdx, video, pairs) => {
    setGroupPairs(prev => ({ ...prev, [`${frameIdx}_${video}`]: pairs }))
  }, [])

  const handleDeleted = useCallback((path) => {
    setDeletedPaths(prev => [...prev, path])
  }, [])

  const allPairs   = Object.values(groupPairs).flat()
  const totalPairs = allPairs.length

  const handleSubmit = async () => {
    setError('')
    setLoading(true)
    try {
      const res = await fetch(`/api/person/confirm-pairs/${jobId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pairs: allPairs, deleted_paths: deletedPaths }),
      })
      const data = await res.json()
      if (data.error) throw new Error(data.error)
      onSubmitted()
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  if (!frameGroups.length) {
    return (
      <div className="card">
        <div className="card-title">Pair Faces & Bodies</div>
        <div style={{ textAlign: 'center', color: '#475569', padding: 40 }}>
          Waiting for pipeline to reach pairing stage…
        </div>
      </div>
    )
  }

  return (
    <div className="card">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
        <div className="card-title" style={{ marginBottom: 0 }}>Pair Faces & Bodies</div>
        <span style={{ fontSize: 13, color: '#64748b' }}>
          {frameGroups.length} frame groups · {totalPairs} pairs confirmed
        </span>
      </div>

      <div style={{ fontSize: 12, color: '#475569', marginBottom: 16, lineHeight: 1.6 }}>
        Click a <strong style={{ color: '#a78bfa' }}>face</strong> to select it (purple ring),
        then click a <strong style={{ color: '#7c9ef8' }}>body</strong> to pair them.&nbsp;
        <strong style={{ color: '#ef4444' }}>×</strong> deletes a crop permanently.&nbsp;
        Skip frames with no valid detections.
      </div>

      <div
        role="alert"
        style={{
          fontSize: 12,
          color: '#92400e',
          background: '#fef3c7',
          border: '1px solid #fcd34d',
          borderRadius: 6,
          padding: '8px 12px',
          marginBottom: 16,
          lineHeight: 1.5,
        }}
      >
        ⚠ Only the face↔body pairs you explicitly confirm here will be saved to
        the profile. Every unpaired detection is discarded on submit — review each
        frame and pair the correct person before continuing.
      </div>

      {frameGroups.map((group, i) => (
        <FrameGroup
          key={`${group.frame_idx}_${group.video}_${i}`}
          group={group}
          jobId={jobId}
          onGroupChange={handleGroupChange}
          onDeleted={handleDeleted}
        />
      ))}

      {error && <div style={{ color: '#ef4444', fontSize: 13, marginBottom: 12 }}>{error}</div>}

      <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginTop: 8 }}>
        <button
          className="btn btn-primary"
          onClick={handleSubmit}
          disabled={loading || totalPairs === 0}
          style={{ padding: '10px 28px' }}
        >
          {loading ? 'Submitting…' : `→ Confirm ${totalPairs} pair${totalPairs !== 1 ? 's' : ''} & Continue`}
        </button>
        {totalPairs === 0 && (
          <span style={{ fontSize: 13, color: '#ef4444' }}>
            ✕ Pair at least one face with a body before continuing
          </span>
        )}
      </div>
    </div>
  )
}
