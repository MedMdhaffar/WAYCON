import { normalizeRollingAnalysis, safeErrorMessage } from '../liveJob.js'
import LiveIdentityCard from './LiveIdentityCard.jsx'
import LiveIdentitySummary from './LiveIdentitySummary.jsx'
import LiveRecognitionEvents from './LiveRecognitionEvents.jsx'

function stateMessage(analysis) {
  if (analysis.state === 'analyzing') return 'Analyzing accumulated identity evidence...'
  if (analysis.state === 'scheduled') return 'Identity analysis is waiting for the current pass.'
  if (analysis.state === 'waiting') return 'Waiting for sufficient face evidence.'
  if (analysis.state === 'ready' && !analysis.identities.length) {
    return 'No provisional identities are available yet.'
  }
  return ''
}

export default function LiveIdentityPanel({ rollingAnalysis, finalizing = false }) {
  const analysis = normalizeRollingAnalysis(rollingAnalysis)
  if (!analysis.enabled) return null
  const message = stateMessage(analysis)

  return (
    <section className="card live-identity-panel" aria-labelledby="live-identity-title">
      <div className="live-identity-heading">
        <div>
          <div className="card-title" id="live-identity-title">Live Identity Analysis</div>
          <p>Provisional session identities from accumulated live evidence</p>
        </div>
        {analysis.analysisInProgress && <span className="live-analysis-indicator">Analyzing</span>}
      </div>

      {finalizing && (
        <div className="live-analysis-notice">Finalizing canonical profiles...</div>
      )}
      {analysis.warning && (
        <div className={`live-analysis-notice ${analysis.state === 'error' ? 'is-error' : 'is-warning'}`}>
          {safeErrorMessage(analysis.warning, 'Rolling identity analysis warning.')}
        </div>
      )}

      <LiveIdentitySummary analysis={analysis} />
      {message && <div className="live-identity-empty">{message}</div>}

      {!!analysis.identities.length && (
        <div className="live-identity-grid">
          {analysis.identities.map(identity => (
            <LiveIdentityCard key={identity.sessionPersonId} identity={identity} />
          ))}
        </div>
      )}

      <LiveRecognitionEvents events={analysis.events} />
    </section>
  )
}
