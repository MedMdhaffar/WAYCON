import { completeCardLatency, formatSimilarity } from '../liveJob.js'
import SafeImage from './SafeImage.jsx'

function observationLabel(count, singular, plural) {
  return `${count} ${count === 1 ? singular : plural}`
}

function seenRange(identity) {
  const first = identity.firstSeenChunk
  const last = identity.lastSeenChunk
  if (first === null && last === null) return 'Chunk range unavailable'
  if (first === null) return `Last seen in chunk ${last}`
  if (last === null || first === last) return `Seen in chunk ${first}`
  return `Seen from chunk ${first} to ${last}`
}

function readable(value, fallback = 'Pending') {
  return value ? String(value).replaceAll('_', ' ') : fallback
}

function pathValue(value) {
  return value || 'Unavailable'
}

export default function LiveIdentityCard({ identity }) {
  const known = identity.persisted && Boolean(identity.canonicalPersonId)
  const strongKnown = identity.reason === 'strong_clear_match' && Boolean(identity.candidatePersonId)
  const name = identity.memoryMatch?.name || identity.canonicalPersonId || 'Known person'
  const candidateSimilarity = formatSimilarity(identity.candidateSimilarity)
  const secondCandidateSimilarity = formatSimilarity(identity.secondCandidateSimilarity)
  const margin = formatSimilarity(identity.margin)
  const bodyImagePath = identity.selectedBodyCrop || identity.bestBodyPath
  const latency = completeCardLatency(identity.latencyMetrics)
  const imageAlt = known || strongKnown
    ? `Best face for ${name}`
    : `Best face for ${identity.liveIdentityId}`

  return (
    <article
      className={`live-identity-card ${known ? 'is-known' : 'is-unknown'}`}
      data-live-identity-id={identity.liveIdentityId}
      data-vlm-status={identity.vlmStatus}
      data-clustering-state={identity.clusteringState}
      data-status-to-card-ms={latency.statusToCardMs ?? ''}
      data-capture-to-card-ms={latency.captureToCardMs ?? ''}
    >
      <div className="live-identity-media">
        <SafeImage
          className="live-identity-image is-face"
          placeholderClassName="live-identity-image-placeholder is-face"
          path={identity.bestFacePath}
          alt={imageAlt}
          placeholder="No face image"
        />
        <SafeImage
          className="live-identity-image is-body"
          placeholderClassName="live-identity-image-placeholder is-body"
          path={bodyImagePath}
          alt={`Best body crop for ${identity.liveIdentityId}`}
          placeholder="No body image"
        />
      </div>
      <div className="live-identity-card-body">
        <div className="live-identity-kind">
          {strongKnown
            ? 'Strong known-person match'
            : identity.state === 'observing'
            ? 'Waiting for valid face'
            : identity.provisional
            ? 'Provisional comparison'
            : known ? 'Known person detected' : 'Unknown person detected'}
        </div>
        <h3>{known ? name : identity.candidatePersonId || 'Unknown person'}</h3>
        <div className="live-identity-id">{identity.liveIdentityId}</div>
        <dl className="live-identity-facts">
          <div><dt>Canonical person</dt><dd>{identity.canonicalPersonId || 'Pending'}</dd></div>
          <div><dt>State</dt><dd>{readable(identity.state)}</dd></div>
          <div><dt>Clustering</dt><dd>{readable(identity.clusteringState)}</dd></div>
          <div><dt>Decision</dt><dd>{readable(identity.decision)}</dd></div>
          <div><dt>Comparison</dt><dd>{identity.provisional ? 'Provisional' : 'Durable'}</dd></div>
          <div><dt>Persisted</dt><dd>{identity.persisted ? 'Yes' : 'No'}</dd></div>
          <div><dt>Reason</dt><dd>{readable(identity.reason)}</dd></div>
          <div><dt>Candidate</dt><dd>{identity.candidatePersonId || 'None'}</dd></div>
          <div><dt>Similarity</dt><dd>{candidateSimilarity || 'Unavailable'}</dd></div>
          <div><dt>Second candidate</dt><dd>{identity.secondCandidatePersonId || 'None'}</dd></div>
          <div><dt>Second similarity</dt><dd>{secondCandidateSimilarity || 'Unavailable'}</dd></div>
          <div><dt>Margin</dt><dd>{margin || 'Unavailable'}</dd></div>
          <div><dt>Capture to embedding</dt><dd>{latency.captureToEmbeddingMs === null ? 'Unavailable' : `${Math.round(latency.captureToEmbeddingMs)} ms`}</dd></div>
          <div><dt>Capture to card</dt><dd>{latency.captureToCardMs === null ? 'Unavailable' : `${Math.round(latency.captureToCardMs)} ms`}</dd></div>
        </dl>
        <div className="live-identity-evidence">
          <span>{observationLabel(identity.observationCount, 'face observation', 'face observations')}</span>
          <span>{observationLabel(identity.bodyCount, 'body observation', 'body observations')}</span>
          <span>{seenRange(identity)}</span>
        </div>
        <div className="live-identity-paths">
          <span>Best face</span><code title={identity.bestFacePath}>{pathValue(identity.bestFacePath)}</code>
          <span>Best body</span><code title={identity.bestBodyPath}>{pathValue(identity.bestBodyPath)}</code>
        </div>
        <div className={`live-vlm-state is-${identity.vlmStatus}`}>
          <div>
            <span>Clothing VLM</span>
            <strong>{readable(identity.vlmStatus, 'Not started')}</strong>
          </div>
          <div className="live-identity-selected-crop">
            <span>Selected body crop</span>
            <code title={identity.selectedBodyCrop}>{pathValue(identity.selectedBodyCrop)}</code>
          </div>
          <p>{identity.clothingDescription || 'Clothing description pending.'}</p>
          {identity.vlmError && (
            <div className="live-vlm-error">{readable(identity.vlmError, 'Inference error')}</div>
          )}
        </div>
        <span className="live-identity-status">{readable(identity.state, 'Observing')}</span>
      </div>
    </article>
  )
}
