import { useState, useEffect, useRef, useCallback } from 'react'
import StartForm from './components/StartForm.jsx'
import ProgressTracker from './components/ProgressTracker.jsx'
import LiveStreamStats from './components/LiveStreamStats.jsx'
import RealtimeMonitor from './components/RealtimeMonitor.jsx'
import CropsGrid from './components/CropsGrid.jsx'
import AssociationsView from './components/AssociationsView.jsx'
import ClothingPanel from './components/ClothingPanel.jsx'
import ReviewPanel from './components/ReviewPanel.jsx'
import ProfileManager from './components/ProfileManager.jsx'
import MemoryTab from './components/MemoryTab.jsx'
import DebugPanel from './components/DebugPanel.jsx'

const TABS = ['Setup', 'Progress & Crops', 'Results', 'Memory', 'Debug']

export default function App() {
  const [mode, setMode] = useState('enroll')   // 'enroll' | 'manage'
  const [tab, setTab] = useState(0)
  const [jobId, setJobId] = useState(null)
  const [jobStatus, setJobStatus] = useState(null)
  const [realtimeCameraId, setRealtimeCameraId] = useState(null)
  const pollRef = useRef(null)

  const fetchStatus = useCallback(async (id) => {
    try {
      const res = await fetch(`/api/person/status/${id}`)
      const data = await res.json()
      setJobStatus(data)
      return data
    } catch (_) {}
  }, [])

  useEffect(() => {
    if (!jobId) return
    fetchStatus(jobId)
    pollRef.current = setInterval(async () => {
      const data = await fetchStatus(jobId)
      if (!data) return
      if (data.status === 'done' || data.status === 'error') {
        setTab(2)
        clearInterval(pollRef.current)
      }
    }, 2000)
    return () => clearInterval(pollRef.current)
  }, [jobId, fetchStatus])

  const handleStart = (id) => {
    setRealtimeCameraId(null)
    setJobId(id)
    setJobStatus(null)
    setTab(1)
  }

  const handleRealtimeStart = (cameraId) => {
    setJobId(null)
    setJobStatus(null)
    setRealtimeCameraId(cameraId)
    setTab(1)
  }

  const refreshStatus = useCallback(() => {
    if (jobId) fetchStatus(jobId)
  }, [jobId, fetchStatus])

  const snapshot = jobStatus?.snapshot ?? {}
  const clusterProfiles = Object.fromEntries(
    Object.values(snapshot.per_cluster_profiles ?? {}).map(profile => [String(profile.cluster_id), profile])
  )

  return (
    <>
      <div className="header">
        <h1>Forensics — Person Creation</h1>
        {mode === 'enroll' && jobStatus && (
          <span>Job: {jobId?.slice(0, 8)} · {jobStatus.status}</span>
        )}
        {mode === 'enroll' && realtimeCameraId && (
          <span>Camera: {realtimeCameraId} · live</span>
        )}
        <div style={{ marginLeft: 'auto', display: 'flex', gap: 4 }}>
          <button
            className={`btn ${mode === 'enroll' ? 'btn-primary' : 'btn-ghost'}`}
            onClick={() => setMode('enroll')}
            style={{ padding: '6px 14px', fontSize: 13 }}
          >
            Enroll
          </button>
          <button
            className={`btn ${mode === 'manage' ? 'btn-primary' : 'btn-ghost'}`}
            onClick={() => setMode('manage')}
            style={{ padding: '6px 14px', fontSize: 13 }}
          >
            Manage
          </button>
        </div>
      </div>

      {mode === 'manage' ? (
        <div className="tab-content">
          <ProfileManager />
        </div>
      ) : (
        <>
          <div className="tabs">
            {TABS.map((label, i) => (
              <button
                key={label}
                className={`tab-btn${tab === i ? ' active' : ''}`}
                onClick={() => setTab(i)}
                disabled={i > 0 && i !== 3 && i !== 4 && !jobId && !realtimeCameraId}
              >
                {label}
              </button>
            ))}
          </div>

          <div className="tab-content">
            {tab === 0 && <StartForm onStart={handleStart} onRealtimeStart={handleRealtimeStart} />}

        {tab === 1 && realtimeCameraId && (
          <RealtimeMonitor
            cameraId={realtimeCameraId}
            onStopped={() => setRealtimeCameraId(null)}
          />
        )}

        {tab === 1 && jobId && (
          <>
            <ProgressTracker
              status={jobStatus?.status}
              node={jobStatus?.node}
              error={jobStatus?.error}
              sourceType={snapshot.source_type}
            />
            <LiveStreamStats snapshot={snapshot} />
            <CropsGrid
              jobId={jobId}
              bodyCrops={snapshot.quality_body_crops ?? []}
              faceCrops={snapshot.quality_face_crops ?? []}
              onDeleted={refreshStatus}
            />
          </>
        )}

        {tab === 2 && (
          <>
            <LiveStreamStats snapshot={snapshot} />
            <AssociationsView
              jobId={jobId}
              associations={snapshot.associations ?? []}
              onDeleted={refreshStatus}
            />
            <ClothingPanel
              bestBodyCrops={snapshot.best_body_crops ?? []}
              clothingStructured={snapshot.clothing_structured ?? {}}
              perClusterBestBodyCrops={snapshot.per_cluster_best_body_crops ?? {}}
              perClusterClothing={snapshot.per_cluster_clothing ?? {}}
              clusterProfiles={clusterProfiles}
            />
            <ReviewPanel
              snapshot={snapshot}
              finalStatus={jobStatus?.status}
            />
          </>
        )}

        {tab === 3 && <MemoryTab />}

        {tab === 4 && <DebugPanel />}
          </div>
        </>
      )}
    </>
  )
}
