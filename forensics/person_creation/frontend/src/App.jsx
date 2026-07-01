import { useState, useEffect, useRef, useCallback } from 'react'
import StartForm from './components/StartForm.jsx'
import ProgressTracker from './components/ProgressTracker.jsx'
import CropsGrid from './components/CropsGrid.jsx'
import FramePairingPanel from './components/FramePairingPanel.jsx'
import AssociationsView from './components/AssociationsView.jsx'
import ClothingPanel from './components/ClothingPanel.jsx'
import ReviewPanel from './components/ReviewPanel.jsx'
import ProfileManager from './components/ProfileManager.jsx'

const TABS = ['Setup', 'Progress & Crops', 'Pair Faces & Bodies', 'Review & Approve']
const RUNNING_STATUSES = new Set([
  'loading_models', 'processing_video', 'filtering', 'embedding',
  'selecting', 'describing',
])

export default function App() {
  const [mode, setMode] = useState('enroll')   // 'enroll' | 'manage'
  const [tab, setTab] = useState(0)
  const [jobId, setJobId] = useState(null)
  const [jobStatus, setJobStatus] = useState(null)
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
      if (data.status === 'awaiting_pairing') {
        setTab(2)
        clearInterval(pollRef.current)
      } else if (data.status === 'awaiting_review') {
        setTab(3)
        clearInterval(pollRef.current)
      } else if (data.status === 'done' || data.status === 'error') {
        clearInterval(pollRef.current)
      }
    }, 2000)
    return () => clearInterval(pollRef.current)
  }, [jobId, fetchStatus])

  const handleStart = (id) => {
    setJobId(id)
    setJobStatus(null)
    setTab(1)
  }

  const refreshStatus = useCallback(() => {
    if (jobId) fetchStatus(jobId)
  }, [jobId, fetchStatus])

  const handlePairsSubmitted = useCallback(() => {
    // Resume polling after pairs confirmed — pipeline continues
    if (!jobId) return
    pollRef.current = setInterval(async () => {
      const data = await fetchStatus(jobId)
      if (!data) return
      if (data.status === 'awaiting_review') {
        setTab(3)
        clearInterval(pollRef.current)
      } else if (data.status === 'done' || data.status === 'error') {
        clearInterval(pollRef.current)
      }
    }, 2000)
  }, [jobId, fetchStatus])

  const snapshot = jobStatus?.snapshot ?? {}
  const [clothingOverride, setClothingOverride] = useState(null)

  return (
    <>
      <div className="header">
        <h1>Forensics — Person Creation</h1>
        {mode === 'enroll' && jobStatus && (
          <span>Job: {jobId?.slice(0, 8)} · {jobStatus.status}</span>
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
                disabled={i > 0 && !jobId}
              >
                {label}
              </button>
            ))}
          </div>

          <div className="tab-content">
            {tab === 0 && <StartForm onStart={handleStart} />}

        {tab === 1 && (
          <>
            <ProgressTracker status={jobStatus?.status} node={jobStatus?.node} error={jobStatus?.error} />
            <CropsGrid
              jobId={jobId}
              bodyCrops={snapshot.quality_body_crops ?? []}
              faceCrops={snapshot.quality_face_crops ?? []}
              onDeleted={refreshStatus}
            />
          </>
        )}

        {tab === 2 && (
          <FramePairingPanel
            jobId={jobId}
            frameGroups={snapshot.frame_groups ?? []}
            onSubmitted={handlePairsSubmitted}
          />
        )}

        {tab === 3 && (
          <>
            <AssociationsView
              jobId={jobId}
              associations={snapshot.associations ?? []}
              onDeleted={refreshStatus}
            />
            <ClothingPanel
              bestBodyCrops={snapshot.best_body_crops ?? []}
              clothingStructured={snapshot.clothing_structured ?? {}}
              onChange={setClothingOverride}
            />
            <ReviewPanel
              jobId={jobId}
              snapshot={snapshot}
              clothingOverride={clothingOverride}
              onDone={refreshStatus}
              finalStatus={jobStatus?.status}
            />
          </>
        )}
          </div>
        </>
      )}
    </>
  )
}
