import { useState, useEffect, useRef, useCallback } from 'react'
import StartForm from './components/StartForm.jsx'
import ProgressTracker from './components/ProgressTracker.jsx'
import LiveStreamStats from './components/LiveStreamStats.jsx'
import LiveJobControls from './components/LiveJobControls.jsx'
import LiveIdentityPanel from './components/LiveIdentityPanel.jsx'
import CropsGrid from './components/CropsGrid.jsx'
import AssociationsView from './components/AssociationsView.jsx'
import ClothingPanel from './components/ClothingPanel.jsx'
import ReviewPanel from './components/ReviewPanel.jsx'
import ProfileManager from './components/ProfileManager.jsx'
import MemoryTab from './components/MemoryTab.jsx'
import IdentityReviewView from './components/IdentityReviewView.jsx'
import {
  isJobActive,
  isCanonicalFinalizing,
  isStopPending,
  createStatusRequestGuard,
  mergeJobStatus,
  normalizeSnapshot,
  postStopRequest,
  runSingleFlight,
  safeErrorMessage,
  shouldShowRollingIdentityPanel,
  shouldShowStop,
  startStatusPolling,
  statusLabel,
} from './liveJob.js'

const TABS = ['Setup', 'Progress & Crops', 'Results', 'Memory', 'Identity Reviews']

export default function App() {
  const [mode, setMode] = useState('enroll')   // 'enroll' | 'manage'
  const [tab, setTab] = useState(0)
  const [jobId, setJobId] = useState(null)
  const [jobStatus, setJobStatus] = useState(null)
  const [jobSourceType, setJobSourceType] = useState(null)
  const [pollingError, setPollingError] = useState('')
  const [stopError, setStopError] = useState('')
  const [stopRequestPending, setStopRequestPending] = useState(false)
  const stopRequestRef = useRef(null)
  const statusRequestGuardRef = useRef(null)
  if (statusRequestGuardRef.current === null) {
    statusRequestGuardRef.current = createStatusRequestGuard()
  }

  const fetchStatus = useCallback(async (id) => {
    const guard = statusRequestGuardRef.current
    const request = guard.start(id)
    try {
      const res = await fetch(`/api/person/status/${encodeURIComponent(id)}`, {
        signal: request.signal,
      })
      const data = await res.json().catch(() => ({}))
      if (!res.ok || data.error) {
        throw new Error(data.error || `Status request failed (HTTP ${res.status})`)
      }
      if (!guard.isCurrent(request)) return null
      setJobStatus(current => mergeJobStatus(current, data))
      setPollingError('')
      return data
    } catch (error) {
      if (error?.name === 'AbortError' || !guard.isCurrent(request)) return null
      setPollingError(safeErrorMessage(error, 'Unable to refresh job status. Retrying...'))
      return null
    }
  }, [])

  useEffect(() => {
    if (!jobId) return
    setPollingError('')
    const stopPolling = startStatusPolling({
      fetchStatus: () => fetchStatus(jobId),
      onTerminal: () => {
        setTab(2)
        setStopRequestPending(false)
      },
    })
    return () => {
      stopPolling()
      statusRequestGuardRef.current.cancel()
    }
  }, [jobId, fetchStatus])

  const handleStart = (id, sourceType) => {
    statusRequestGuardRef.current.cancel()
    setJobId(id)
    setJobSourceType(sourceType)
    setJobStatus(null)
    setStopError('')
    setStopRequestPending(false)
    stopRequestRef.current = null
    setTab(1)
  }

  const handleStop = useCallback(() => {
    if (!jobId) return
    setStopError('')
    setStopRequestPending(true)
    return runSingleFlight(stopRequestRef, async () => {
      try {
        const data = await postStopRequest(jobId)
        if (data.status === 'stop_requested') {
          setJobStatus(current => ({
            ...(current ?? {}),
            status: 'stop_requested',
            snapshot: {
              ...(current?.snapshot ?? {}),
              stop_requested: true,
            },
          }))
        }
        await fetchStatus(jobId)
      } catch (error) {
        setStopRequestPending(false)
        setStopError(safeErrorMessage(error, 'Unable to request camera stop.'))
      }
    })
  }, [fetchStatus, jobId])

  const refreshStatus = useCallback(() => {
    if (jobId) fetchStatus(jobId)
  }, [jobId, fetchStatus])

  const snapshot = normalizeSnapshot(jobStatus)
  const sourceType = snapshot.source_type ?? jobSourceType
  const activeJob = isJobActive(jobId, jobStatus?.status)
  const showStop = shouldShowStop(jobId, sourceType, jobStatus?.status, jobStatus?.node, snapshot)
  const stopping = isStopPending(jobStatus?.status, snapshot, stopRequestPending)
  const finalizingCanonical = isCanonicalFinalizing(
    sourceType,
    jobStatus?.status,
    jobStatus?.node,
    snapshot,
    stopRequestPending,
  )
  const clusterProfiles = Object.fromEntries(
    Object.values(snapshot.per_cluster_profiles ?? {}).map(profile => [String(profile.cluster_id), profile])
  )

  return (
    <>
      <div className="header">
        <h1>Forensics — Person Creation</h1>
        {mode === 'enroll' && jobStatus && (
          <span>Job: {jobId?.slice(0, 8)} · {statusLabel(jobStatus.status)}</span>
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
                disabled={i > 0 && i !== 3 && i !== 4 && !jobId}
              >
                {label}
              </button>
            ))}
          </div>

          <div className="tab-content">
            {showStop && (
              <LiveJobControls stopping={stopping} onStop={handleStop} error={stopError} />
            )}
            {pollingError && (
              <div className="card" style={{ borderColor: '#f59e0b', color: '#f59e0b', fontSize: 13 }}>
                {pollingError}
              </div>
            )}
            {tab === 0 && <StartForm onStart={handleStart} activeJob={activeJob} />}

        {tab === 1 && (
          <>
            <ProgressTracker
              status={jobStatus?.status}
              node={jobStatus?.node}
              error={jobStatus?.error ? safeErrorMessage(jobStatus.error, 'Pipeline failed.') : ''}
              sourceType={sourceType}
            />
            <LiveStreamStats snapshot={snapshot} status={jobStatus?.status} node={jobStatus?.node} sourceType={sourceType} />
            {shouldShowRollingIdentityPanel(jobStatus?.status) && (
              <div className="live-identities-section">
                <LiveIdentityPanel
                  rollingAnalysis={snapshot.rolling_analysis}
                  finalizing={finalizingCanonical}
                />
              </div>
            )}
            <CropsGrid
              jobId={jobId}
              bodyCrops={snapshot.quality_body_crops ?? []}
              faceCrops={snapshot.quality_face_crops ?? []}
              mediaVersion={snapshot.media_lifecycle_version ?? 0}
              onDeleted={refreshStatus}
            />
          </>
        )}

        {tab === 2 && (
          <>
            {jobStatus?.status === 'error' && jobStatus.error && (
              <div className="card job-error-banner">
                {safeErrorMessage(jobStatus.error, 'Pipeline failed.')}
              </div>
            )}
            {jobStatus?.status === 'error' && (
              <div className="live-identities-section">
                <LiveIdentityPanel rollingAnalysis={snapshot.rolling_analysis} />
              </div>
            )}
            <LiveStreamStats snapshot={snapshot} status={jobStatus?.status} node={jobStatus?.node} sourceType={sourceType} />
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
        {tab === 4 && <IdentityReviewView />}
          </div>
        </>
      )}
    </>
  )
}
