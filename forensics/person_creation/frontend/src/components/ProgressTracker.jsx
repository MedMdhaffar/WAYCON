import { statusLabel } from '../liveJob.js'

const pipelineNodes = (sourceType) => [
  { id: 'load_models', label: 'Load Models' },
  sourceType === 'live_camera'
    ? { id: 'process_live_stream', label: 'Capture Live Camera' }
    : { id: 'process_video', label: 'Extract Video Crops' },
  { id: 'filter_quality', label: 'Filter Quality' },
  { id: 'embed_all_faces', label: 'Embed Faces' },
  { id: 'cluster_identities', label: 'Cluster Identities' },
  { id: 'assign_bodies_to_clusters', label: 'Assign Bodies' },
  { id: 'promote_crops', label: 'Promote Crops' },
  { id: 'select_best', label: 'Select Best' },
  { id: 'compute_reid', label: 'Compute ReID' },
  { id: 'describe_clothing', label: 'Describe Clothing' },
  { id: 'build_profile', label: 'Build Profile' },
  { id: 'finalize', label: 'Finalize' },
]

const STATUS_COLORS = {
  done: '#22c55e',
  error: '#ef4444',
  stop_requested: '#f59e0b',
  stopping: '#f59e0b',
  default: '#7c9ef8',
}

export default function ProgressTracker({ status, node, error, sourceType }) {
  const nodes = pipelineNodes(sourceType)
  const currentIndex = nodes.findIndex(item => item.id === node)
  const statusColor = STATUS_COLORS[status] ?? STATUS_COLORS.default

  return (
    <div className="card">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
        <div className="card-title" style={{ marginBottom: 0 }}>Pipeline Progress</div>
        <span style={{ fontSize: 13, fontWeight: 600, color: statusColor, background: statusColor + '22', padding: '4px 12px', borderRadius: 20 }}>
          {statusLabel(status)}
        </span>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', overflowX: 'auto', paddingBottom: 4 }}>
        {nodes.map((item, index) => {
          const isDone = index < currentIndex || status === 'done'
          const isActive = item.id === node
          const color = isDone ? '#22c55e' : isActive ? statusColor : '#1e2330'
          const textColor = isDone ? '#22c55e' : isActive ? statusColor : '#475569'

          return (
            <div key={item.id} style={{ display: 'flex', alignItems: 'center', flex: '0 0 auto' }}>
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6, minWidth: 90 }}>
                <div style={{
                  width: 32, height: 32, borderRadius: '50%',
                  background: color + '22', border: `2px solid ${color}`,
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  fontSize: 13, color,
                }}>
                  {isDone ? 'OK' : index + 1}
                </div>
                <span style={{ fontSize: 11, color: textColor, textAlign: 'center', lineHeight: 1.3 }}>{item.label}</span>
              </div>
              {index < nodes.length - 1 && (
                <div style={{ width: 24, height: 2, background: isDone ? '#22c55e' : '#1e2330', flexShrink: 0, marginBottom: 20 }} />
              )}
            </div>
          )
        })}
      </div>

      {error && (
        <div style={{ marginTop: 12, padding: '10px 14px', background: '#1e1015', border: '1px solid #ef4444', borderRadius: 6, fontSize: 13, color: '#ef4444', whiteSpace: 'pre-wrap' }}>
          {error}
        </div>
      )}
    </div>
  )
}
