const NODES = [
  { id: 'load_models',       label: 'Load Models' },
  { id: 'process_video',     label: 'Extract Crops' },
  { id: 'filter_quality',    label: 'Filter Quality' },
  { id: 'embed_faces',       label: 'Embed Faces' },
  { id: 'human_in_the_loop', label: 'Pair Manually' },
  { id: 'select_best',       label: 'Select Best' },
  { id: 'describe_clothing', label: 'Describe Clothing' },
  { id: 'build_profile',     label: 'Build Profile' },
  { id: 'finalize',          label: 'Finalize' },
]

const STATUS_COLORS = {
  done:             '#22c55e',
  awaiting_review:  '#f59e0b',
  awaiting_pairing: '#a78bfa',
  error:            '#ef4444',
  default:          '#7c9ef8',
}

export default function ProgressTracker({ status, node, error }) {
  const currentIdx = NODES.findIndex(n => n.id === node)

  const statusColor = STATUS_COLORS[status] ?? STATUS_COLORS.default
  const statusLabel = status === 'awaiting_pairing'  ? '⏸ Awaiting Pairing'
    : status === 'awaiting_review' ? '⏸ Awaiting Review'
    : status === 'done'            ? '✓ Done'
    : status === 'error'           ? '✕ Error'
    : status ? `⟳ ${status.replace(/_/g, ' ')}` : '—'

  return (
    <div className="card">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
        <div className="card-title" style={{ marginBottom: 0 }}>Pipeline Progress</div>
        <span style={{ fontSize: '13px', fontWeight: 600, color: statusColor, background: statusColor + '22', padding: '4px 12px', borderRadius: '20px' }}>
          {statusLabel}
        </span>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', overflowX: 'auto', paddingBottom: '4px' }}>
        {NODES.map((n, i) => {
          const isDone   = i < currentIdx || status === 'done'
          const isActive = n.id === node
          const color    = isDone ? '#22c55e' : isActive ? statusColor : '#1e2330'
          const textColor = isDone ? '#22c55e' : isActive ? statusColor : '#475569'

          return (
            <div key={n.id} style={{ display: 'flex', alignItems: 'center', flex: '0 0 auto' }}>
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '6px', minWidth: '90px' }}>
                <div style={{
                  width: '32px', height: '32px', borderRadius: '50%',
                  background: color + '22', border: `2px solid ${color}`,
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  fontSize: '13px', color,
                }}>
                  {isDone ? '✓' : i + 1}
                </div>
                <span style={{ fontSize: '11px', color: textColor, textAlign: 'center', lineHeight: 1.3 }}>{n.label}</span>
              </div>
              {i < NODES.length - 1 && (
                <div style={{ width: '24px', height: '2px', background: isDone ? '#22c55e' : '#1e2330', flexShrink: 0, marginBottom: '20px' }} />
              )}
            </div>
          )
        })}
      </div>

      {error && (
        <div style={{ marginTop: '12px', padding: '10px 14px', background: '#1e1015', border: '1px solid #ef4444', borderRadius: '6px', fontSize: '13px', color: '#ef4444', fontFamily: 'monospace', whiteSpace: 'pre-wrap' }}>
          {error}
        </div>
      )}
    </div>
  )
}
