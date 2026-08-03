import { displayMetric } from '../liveJob.js'

function SummaryMetric({ label, value }) {
  return (
    <div className="live-identity-summary-metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  )
}

export default function LiveIdentitySummary({ analysis }) {
  return (
    <div className="live-identity-summary" aria-label="Rolling identity analysis summary">
      <SummaryMetric label="State" value={analysis.stateLabel} />
      <SummaryMetric label="Identities" value={analysis.identities.length} />
      <SummaryMetric label="Face evidence" value={analysis.faceEvidenceCount} />
      <SummaryMetric label="Embeddings analyzed" value={analysis.analyzedEmbeddingCount} />
      <SummaryMetric label="Latest chunk" value={displayMetric(analysis.lastCompletedChunk)} />
      <SummaryMetric label="Requested version" value={analysis.requestedVersion} />
      <SummaryMetric label="Completed analysis" value={analysis.analysisVersion} />
    </div>
  )
}
