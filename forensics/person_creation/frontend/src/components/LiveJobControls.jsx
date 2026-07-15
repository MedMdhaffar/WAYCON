export default function LiveJobControls({ stopping, onStop, error }) {
  return (
    <div className="card" style={{ borderColor: stopping ? '#f59e0b' : '#ef4444' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 16, flexWrap: 'wrap' }}>
        <div>
          <div className="card-title" style={{ marginBottom: 4 }}>Live Camera Control</div>
          <div style={{ color: '#94a3b8', fontSize: 13 }}>
            {stopping
              ? 'The request was accepted. Captured evidence will continue through processing.'
              : 'Capture continues until you stop the camera.'}
          </div>
        </div>
        <button
          type="button"
          className="btn btn-danger"
          onClick={onStop}
          disabled={stopping}
          style={{ minWidth: 142, minHeight: 40, fontWeight: 700 }}
        >
          {stopping ? 'Stopping...' : 'Stop camera'}
        </button>
      </div>
      {error && <div style={{ marginTop: 12, color: '#ef4444', fontSize: 13 }}>{error}</div>}
    </div>
  )
}
