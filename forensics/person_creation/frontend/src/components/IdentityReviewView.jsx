import { useCallback, useEffect, useRef, useState } from 'react'

import {
  createIdentityReviewDetailRequestGuard,
  displayedIdentityReviewId,
  fetchIdentityReviewDetail,
  fetchIdentityReviewQueue,
  postIdentityReviewDecision,
} from '../identityReviewApi.js'
import SafeImage from './SafeImage.jsx'

function formatPercent(value) {
  const number = Number(value)
  return Number.isFinite(number) ? `${(number * 100).toFixed(1)}%` : 'Unavailable'
}

function formatTimestamp(value) {
  if (!value) return 'Unknown'
  const parsed = new Date(value)
  return Number.isNaN(parsed.getTime()) ? String(value) : parsed.toLocaleString()
}

function QueueCard({ review, selected, onSelect }) {
  return (
    <button
      type="button"
      className={`identity-review-queue-card${selected ? ' is-selected' : ''}`}
      onClick={() => onSelect(review.suggestion_id)}
    >
      <SafeImage
        path={review.source_preview}
        alt={`Source ${review.source_name}`}
        placeholder="No preview"
        className="identity-review-queue-image"
        placeholderClassName="identity-review-queue-image is-missing"
      />
      <div className="identity-review-queue-copy">
        <div className="identity-review-queue-status">{review.status}</div>
        <strong>{review.source_name || review.source_person_id}</strong>
        <span>Candidate: {review.candidate_name || review.candidate_person_id}</span>
        <span>Similarity {formatPercent(review.similarity)}</span>
        {review.margin != null && <span>Margin {formatPercent(review.margin)}</span>}
        <small>{formatTimestamp(review.created_at)}</small>
      </div>
    </button>
  )
}

function CropStrip({ gallery, kind, label }) {
  const crops = gallery.filter(item => item.crop_type === kind)
  return (
    <div className="identity-review-evidence-group">
      <h4>{label}</h4>
      {crops.length ? (
        <div className="identity-review-crops">
          {crops.map(crop => (
            <SafeImage
              key={`${crop.original_person_id}-${crop.id}`}
              path={crop.path}
              alt={`${label} evidence`}
              placeholder="Unavailable"
              className={`identity-review-crop is-${kind}`}
              placeholderClassName={`identity-review-crop is-${kind} is-missing`}
            />
          ))}
        </div>
      ) : <div className="identity-review-muted">No {label.toLowerCase()} available.</div>}
    </div>
  )
}

function AppearanceList({ appearances }) {
  if (!appearances.length) return <div className="identity-review-muted">No appearance descriptions available.</div>
  return (
    <div className="identity-review-appearance-list">
      {appearances.map(appearance => (
        <div key={`${appearance.original_person_id}-${appearance.id}`}>
          <strong>{appearance.date || 'Unknown date'}</strong>
          <span>{appearance.full_description || [appearance.top, appearance.bottom, appearance.shoes].filter(Boolean).join(' · ') || 'No description'}</span>
          {appearance.video_sources?.length > 0 && <small>Source: {appearance.video_sources.join(', ')}</small>}
        </div>
      ))}
    </div>
  )
}

function ProfileComparison({ title, profile, gallery, appearances, recognitionEvents }) {
  return (
    <section className="identity-review-profile">
      <div className="identity-review-profile-heading">
        <SafeImage
          path={profile.profile_image}
          alt={profile.name || profile.person_id}
          placeholder="No profile image"
          className="identity-review-profile-image"
          placeholderClassName="identity-review-profile-image is-missing"
        />
        <div>
          <div className="identity-review-side-label">{title}</div>
          <h3>{profile.name || profile.person_id}</h3>
          <code>{profile.person_id}</code>
          <span className={`identity-review-person-state${profile.is_active ? '' : ' is-inactive'}`}>
            {profile.is_active ? 'Active profile' : `Inactive · redirects to ${profile.merged_into_person_id || 'unknown'}`}
          </span>
        </div>
      </div>

      <dl className="identity-review-facts">
        <div><dt>Enrolled</dt><dd>{formatTimestamp(profile.enrolled_at)}</dd></div>
        <div><dt>Updated</dt><dd>{formatTimestamp(profile.updated_at)}</dd></div>
        <div><dt>Evidence count</dt><dd>{profile.embedding_count ?? 'Unknown'}</dd></div>
        <div><dt>Cameras</dt><dd>{profile.cameras?.length ? profile.cameras.join(', ') : 'None recorded'}</dd></div>
      </dl>

      <CropStrip gallery={gallery} kind="face" label="Face crops" />
      <CropStrip gallery={gallery} kind="body" label="Body crops" />

      <div className="identity-review-evidence-group">
        <h4>Recent appearances</h4>
        <AppearanceList appearances={appearances} />
      </div>

      <div className="identity-review-evidence-group">
        <h4>Recognition evidence</h4>
        {recognitionEvents.length ? (
          <div className="identity-review-recognition-list">
            {recognitionEvents.map(event => (
              <div key={`${event.original_person_id}-${event.id}`}>
                <span>{event.event_type}</span>
                <small>{formatTimestamp(event.ts)}{event.similarity != null ? ` · ${formatPercent(event.similarity)}` : ''}</small>
              </div>
            ))}
          </div>
        ) : <div className="identity-review-muted">No recent recognition events.</div>}
      </div>
    </section>
  )
}

export default function IdentityReviewView() {
  const [reviews, setReviews] = useState([])
  const [pendingCount, setPendingCount] = useState(0)
  const [selectedId, setSelectedId] = useState(null)
  const [detail, setDetail] = useState(null)
  const [queueLoading, setQueueLoading] = useState(true)
  const [detailLoading, setDetailLoading] = useState(false)
  const [queueError, setQueueError] = useState('')
  const [detailError, setDetailError] = useState('')
  const [actionError, setActionError] = useState('')
  const [resultBanner, setResultBanner] = useState('')
  const [actionPending, setActionPending] = useState(false)
  const [acceptArmed, setAcceptArmed] = useState(false)
  const actionInFlight = useRef(false)
  const mounted = useRef(false)
  const detailRequests = useRef(createIdentityReviewDetailRequestGuard())

  useEffect(() => {
    mounted.current = true
    detailRequests.current.mount()
    return () => {
      mounted.current = false
      detailRequests.current.unmount()
    }
  }, [])

  const loadQueue = useCallback(async ({ preserveContext = false } = {}) => {
    setQueueLoading(true)
    setQueueError('')
    try {
      const data = await fetchIdentityReviewQueue()
      if (!mounted.current) return null
      setReviews(data.reviews)
      setPendingCount(data.pending_count)
      setSelectedId(current => {
        if (preserveContext && current) return current
        if (current && data.reviews.some(review => review.suggestion_id === current)) return current
        return data.reviews[0]?.suggestion_id ?? null
      })
      return data.reviews
    } catch (error) {
      if (!mounted.current) return null
      setQueueError(error.message || 'Unable to load the review queue.')
      return null
    } finally {
      if (mounted.current) setQueueLoading(false)
    }
  }, [])

  const loadDetail = useCallback(async suggestionId => {
    const request = detailRequests.current.begin(suggestionId)
    setDetail(null)
    setDetailError('')
    if (!suggestionId) {
      setDetailLoading(false)
      return null
    }
    if (!request) return null
    setDetailLoading(true)
    try {
      const data = await fetchIdentityReviewDetail(suggestionId, {
        signal: request.signal,
      })
      if (!detailRequests.current.isCurrent(request, suggestionId)) return null
      setDetail(data)
      return data
    } catch (error) {
      if (!detailRequests.current.isCurrent(request, suggestionId)) return null
      setDetailError(error.message || 'Unable to load this identity review.')
      return null
    } finally {
      if (detailRequests.current.isCurrent(request, suggestionId)) {
        setDetailLoading(false)
      }
    }
  }, [])

  useEffect(() => {
    loadQueue()
  }, [loadQueue])

  useEffect(() => {
    setAcceptArmed(false)
    setActionError('')
    loadDetail(selectedId)
  }, [loadDetail, selectedId])

  useEffect(() => {
    if (!resultBanner) return undefined
    const timeout = window.setTimeout(() => setResultBanner(''), 10000)
    return () => window.clearTimeout(timeout)
  }, [resultBanner])

  const selectReviewManually = useCallback(suggestionId => {
    setResultBanner('')
    setSelectedId(suggestionId)
  }, [])

  const decide = useCallback(async decision => {
    const submissionId = displayedIdentityReviewId(detail, selectedId)
    if (!submissionId || actionInFlight.current) return
    actionInFlight.current = true
    setActionPending(true)
    setActionError('')
    setResultBanner('')
    try {
      const result = await postIdentityReviewDecision(submissionId, decision)
      if (!mounted.current) return
      setResultBanner(
        decision === 'accept'
          ? `Accepted review ${submissionId}: ${result.source_person_id} was merged into ${result.target_person_id}.${result.idempotent_replay ? ' This decision was already complete.' : ''}`
          : `Rejected review ${submissionId}: ${result.source_person_id} and ${result.target_person_id} remain separate.${result.idempotent_replay ? ' This decision was already complete.' : ''}`,
      )
      const nextReviews = await loadQueue()
      if (mounted.current && nextReviews) {
        const nextId = nextReviews.find(review => review.suggestion_id !== submissionId)?.suggestion_id ?? null
        setSelectedId(nextId)
        if (!nextId) setDetail(null)
      }
    } catch (error) {
      if (!mounted.current) return
      if (error.status === 409) {
        setActionError('Another supervisor decision or identity change occurred. The review has been refreshed; no decision was retried.')
        await Promise.all([
          loadQueue({ preserveContext: true }),
          loadDetail(submissionId),
        ])
      } else {
        setActionError(error.message || 'The decision could not be saved. No automatic retry was attempted.')
      }
    } finally {
      actionInFlight.current = false
      if (mounted.current) {
        setActionPending(false)
        setAcceptArmed(false)
      }
    }
  }, [detail, loadDetail, loadQueue, selectedId])

  const displayedId = displayedIdentityReviewId(detail, selectedId)
  const pendingReview = displayedId !== null && detail.suggestion.status === 'pending'

  return (
    <div className="identity-review-layout">
      <aside className="card identity-review-sidebar">
        <div className="identity-review-toolbar">
          <div>
            <div className="card-title">Review Queue</div>
            <span>{pendingCount} pending</span>
          </div>
          <button type="button" className="btn btn-ghost" onClick={() => loadQueue()} disabled={queueLoading || actionPending}>
            {queueLoading ? 'Refreshing…' : 'Refresh queue'}
          </button>
        </div>
        {queueError && <div className="identity-review-banner is-error">{queueError}</div>}
        <div className="identity-review-queue">
          {queueLoading && !reviews.length ? (
            <div className="identity-review-empty">Loading review queue…</div>
          ) : reviews.length ? reviews.map(review => (
            <QueueCard
              key={review.suggestion_id}
              review={review}
              selected={selectedId === review.suggestion_id}
              onSelect={selectReviewManually}
            />
          )) : (
            <div className="identity-review-empty">No identity matches are waiting for review.</div>
          )}
        </div>
      </aside>

      <main className="card identity-review-main">
        {resultBanner && <div className="identity-review-banner is-success">{resultBanner}</div>}
        {actionError && <div className="identity-review-banner is-error">{actionError}</div>}
        {detailError && <div className="identity-review-banner is-error">{detailError}</div>}

        {detailLoading ? (
          <div className="identity-review-empty">Loading source and candidate evidence…</div>
        ) : detail ? (
          <>
            <header className="identity-review-heading">
              <div>
                <div className="card-title">Identity Match Review</div>
                <h2>{formatPercent(detail.suggestion.similarity)} similarity</h2>
                <p>
                  Suggestion {detail.suggestion.suggestion_id}
                  {detail.suggestion.margin != null ? ` · ${formatPercent(detail.suggestion.margin)} margin` : ''}
                  {` · created ${formatTimestamp(detail.suggestion.created_at)}`}
                </p>
              </div>
              <span className={`identity-review-status is-${detail.suggestion.status}`}>{detail.suggestion.status}</span>
            </header>

            <div className="identity-review-comparison">
              <ProfileComparison
                title="Source profile · newly created"
                profile={detail.source_profile}
                gallery={detail.source_gallery}
                appearances={detail.source_appearances}
                recognitionEvents={detail.source_recognition_events}
              />
              <ProfileComparison
                title="Candidate profile · canonical target"
                profile={detail.candidate_profile}
                gallery={detail.candidate_gallery}
                appearances={detail.candidate_appearances}
                recognitionEvents={detail.candidate_recognition_events}
              />
            </div>

            <div className="identity-review-actions">
              <div>
                <strong>Accept: merge source into candidate</strong>
                <span>{detail.source_profile.person_id} → {detail.candidate_profile.person_id}</span>
              </div>
              {acceptArmed ? (
                <div className="identity-review-confirm">
                  <span>Confirm this atomic merge?</span>
                  <button type="button" className="btn btn-success" onClick={() => decide('accept')} disabled={actionPending || !pendingReview || displayedId === null}>
                    {actionPending ? 'Saving decision…' : 'Confirm accept merge'}
                  </button>
                  <button type="button" className="btn btn-ghost" onClick={() => setAcceptArmed(false)} disabled={actionPending}>Cancel</button>
                </div>
              ) : (
                <div className="identity-review-action-buttons">
                  <button type="button" className="btn btn-success" onClick={() => setAcceptArmed(true)} disabled={actionPending || !pendingReview || displayedId === null}>
                    Accept merge
                  </button>
                  <button type="button" className="btn btn-danger" onClick={() => decide('reject')} disabled={actionPending || !pendingReview || displayedId === null}>
                    {actionPending ? 'Saving decision…' : 'Reject match'}
                  </button>
                </div>
              )}
            </div>
          </>
        ) : (
          <div className="identity-review-empty">Select a pending suggestion to compare both profiles.</div>
        )}
      </main>
    </div>
  )
}
