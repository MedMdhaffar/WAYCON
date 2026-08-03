const EVENT_LABELS = {
  identity_created: 'New provisional identity',
  identity_split: 'Provisional identity split',
  identity_merged: 'Provisional identities merged',
  memory_no_match: 'Unknown person observed',
  memory_match_found: 'Known person recognized',
  memory_match_changed: 'Identity match updated',
  memory_match_lost: 'Previous memory match lost',
}

function eventDetail(event) {
  const name = event.memoryMatch?.name
  if (name) return `${name} (${event.sessionPersonId || 'provisional identity'})`
  return event.sessionPersonId || 'Provisional identity'
}

export default function LiveRecognitionEvents({ events }) {
  if (!events.length) return null
  return (
    <div className="live-events">
      <h3>Recent identity events</h3>
      <ol>
        {events.map(event => (
          <li key={event.eventId}>
            <span>{EVENT_LABELS[event.type] ?? 'Identity update'}</span>
            <small>{eventDetail(event)}</small>
          </li>
        ))}
      </ol>
    </div>
  )
}
