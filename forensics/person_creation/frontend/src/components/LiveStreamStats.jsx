import {
  cameraPhase,
  displayMetric,
  liveProgress,
  safeErrorMessage,
} from '../liveJob.js'

function Metric({ label, value }) {
  return (
    <div style={{ padding: 12, borderRadius: 6, background: '#0f1117', border: '1px solid #1e2330' }}>
      <div style={{ color: '#64748b', fontSize: 11, marginBottom: 4 }}>{label}</div>
      <div style={{ color: '#e2e8f0', fontSize: 16, fontWeight: 600 }}>{value}</div>
    </div>
  )
}

export default function LiveStreamStats({ snapshot, status, node, sourceType }) {
  if ((snapshot?.source_type ?? sourceType) !== 'live_camera') return null
  const progress = liveProgress(snapshot)
  const currentValues = [
    ['Camera', cameraPhase(status, node, snapshot)],
    ['Completed windows', displayMetric(progress.completedWindows)],
    ['Current / last window', displayMetric(progress.windowIndex)],
    ['Window duration', displayMetric(progress.windowDuration, ' s')],
    ['Frames read', displayMetric(progress.windowFramesRead)],
    ['Frames processed', displayMetric(progress.windowFramesProcessed)],
    ['Body detections', displayMetric(progress.windowBodyDetections)],
    ['Face detections', displayMetric(progress.windowFaceDetections)],
  ]
  const totalValues = [
    ['Total frames read', displayMetric(progress.totalFramesRead)],
    ['Total frames processed', displayMetric(progress.totalFramesProcessed)],
    ['Total body detections', displayMetric(progress.totalBodyDetections)],
    ['Total face detections', displayMetric(progress.totalFaceDetections)],
  ]

  return (
    <div className="card">
      <div className="card-title">Live Camera Configured</div>
      <div style={{ color: '#64748b', fontSize: 11, marginBottom: 8, textTransform: 'uppercase' }}>Latest window</div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: 10 }}>
        {currentValues.map(([label, value]) => <Metric key={label} label={label} value={value} />)}
      </div>

      <div style={{ color: '#64748b', fontSize: 11, margin: '16px 0 8px', textTransform: 'uppercase' }}>Session totals</div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 10 }}>
        {totalValues.map(([label, value]) => <Metric key={label} label={label} value={value} />)}
      </div>

      {snapshot.stream_report_path && (
        <div style={{ marginTop: 12, color: '#94a3b8', fontSize: 12 }}>
          Stream report: <code>{snapshot.stream_report_path}</code>
        </div>
      )}
      {progress.warnings.map((warning, index) => (
        <div key={index} style={{ marginTop: 8, color: '#f59e0b', fontSize: 12 }}>
          {safeErrorMessage(warning, 'Camera warning')}
        </div>
      ))}
    </div>
  )
}
