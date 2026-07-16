import { formatSimilarity } from '../liveJob.js'
import SafeIdentityImage from './SafeIdentityImage.jsx'

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

export default function LiveIdentityCard({ identity }) {
  const known = identity.memoryMatch !== null
  const name = identity.memoryMatch?.name || 'Known person'
  const similarity = formatSimilarity(identity.memoryMatch?.similarity)
  const imageAlt = known ? `Representative face for ${name}` : `Representative face for ${identity.sessionPersonId}`

  return (
    <article className={`live-identity-card ${known ? 'is-known' : 'is-unknown'}`}>
      <SafeIdentityImage path={identity.representativeFacePath} alt={imageAlt} />
      <div className="live-identity-card-body">
        <div className="live-identity-kind">{known ? 'Known person detected' : 'Unknown person detected'}</div>
        <h3>{known ? name : 'Unknown person'}</h3>
        {known && similarity && <div className="live-identity-similarity">{similarity} similarity</div>}
        <div className="live-identity-id">{identity.sessionPersonId}</div>
        <div className="live-identity-evidence">
          <span>{observationLabel(identity.faceCount, 'face observation', 'face observations')}</span>
          <span>{observationLabel(identity.associatedBodyCount, 'body association', 'body associations')}</span>
          <span>{seenRange(identity)}</span>
        </div>
        <span className="live-identity-status">{identity.status || 'provisional'}</span>
      </div>
    </article>
  )
}
