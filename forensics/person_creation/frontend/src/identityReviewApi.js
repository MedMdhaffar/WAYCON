const REQUEST_TIMEOUT_MS = 12000

export class IdentityReviewApiError extends Error {
  constructor(message, status = 0, { aborted = false } = {}) {
    super(message)
    this.name = 'IdentityReviewApiError'
    this.status = status
    this.aborted = aborted
  }
}

async function requestJson(url, options = {}) {
  const { signal: callerSignal, ...fetchOptions } = options
  const controller = new AbortController()
  let timedOut = false
  const abortFromCaller = () => controller.abort(callerSignal?.reason)
  if (callerSignal?.aborted) abortFromCaller()
  else callerSignal?.addEventListener('abort', abortFromCaller, { once: true })
  const timeout = setTimeout(() => {
    timedOut = true
    controller.abort()
  }, REQUEST_TIMEOUT_MS)
  try {
    const response = await fetch(url, { ...fetchOptions, signal: controller.signal })
    const data = await response.json().catch(() => null)
    if (!data || typeof data !== 'object' || Array.isArray(data)) {
      throw new IdentityReviewApiError('The review service returned a malformed response.', response.status)
    }
    if (!response.ok) {
      throw new IdentityReviewApiError(
        typeof data.error === 'string' && data.error ? data.error : `Review request failed (HTTP ${response.status}).`,
        response.status,
      )
    }
    return data
  } catch (error) {
    if (error?.name === 'AbortError') {
      if (!timedOut && callerSignal?.aborted) {
        throw new IdentityReviewApiError('The review request was cancelled.', 0, { aborted: true })
      }
      throw new IdentityReviewApiError('The review request timed out. The decision was not retried.', 0)
    }
    if (error instanceof IdentityReviewApiError) throw error
    throw new IdentityReviewApiError('The identity review service is unavailable.', 0)
  } finally {
    clearTimeout(timeout)
    callerSignal?.removeEventListener('abort', abortFromCaller)
  }
}

export function createIdentityReviewDetailRequestGuard() {
  let generation = 0
  let selectedId = null
  let controller = null
  let mounted = true

  const abortCurrent = () => {
    generation += 1
    controller?.abort()
    controller = null
  }

  return {
    mount() {
      mounted = true
    },
    begin(suggestionId) {
      abortCurrent()
      selectedId = suggestionId ?? null
      if (!mounted || !selectedId) return null
      controller = new AbortController()
      return Object.freeze({
        generation,
        suggestionId: selectedId,
        signal: controller.signal,
      })
    },
    isCurrent(request, currentSelectedId) {
      return Boolean(
        mounted
        && request
        && !request.signal.aborted
        && request.generation === generation
        && request.suggestionId === selectedId
        && request.suggestionId === currentSelectedId,
      )
    },
    unmount() {
      mounted = false
      selectedId = null
      abortCurrent()
    },
  }
}

export function displayedIdentityReviewId(detail, selectedId) {
  const detailId = detail?.suggestion?.suggestion_id
  return typeof detailId === 'string' && detailId === selectedId ? detailId : null
}

function validSummary(review) {
  return review && typeof review === 'object'
    && typeof review.suggestion_id === 'string'
    && typeof review.source_person_id === 'string'
    && typeof review.candidate_person_id === 'string'
    && review.status === 'pending'
}

export async function fetchIdentityReviewQueue() {
  const data = await requestJson('/api/identity-reviews?limit=100&offset=0')
  if (!Array.isArray(data.reviews) || !Number.isInteger(data.pending_count) || data.pending_count < 0) {
    throw new IdentityReviewApiError('The review queue response is malformed.', 0)
  }
  if (!data.reviews.every(validSummary)) {
    throw new IdentityReviewApiError('The review queue contains malformed entries.', 0)
  }
  return data
}

export async function fetchIdentityReviewDetail(suggestionId, { signal } = {}) {
  const data = await requestJson(
    `/api/identity-reviews/${encodeURIComponent(suggestionId)}`,
    { signal },
  )
  if (
    !data.suggestion || typeof data.suggestion.suggestion_id !== 'string'
    || data.suggestion.suggestion_id !== suggestionId
    || !data.source_profile || typeof data.source_profile.person_id !== 'string'
    || !data.candidate_profile || typeof data.candidate_profile.person_id !== 'string'
    || !Array.isArray(data.source_gallery) || !Array.isArray(data.candidate_gallery)
    || !Array.isArray(data.source_appearances) || !Array.isArray(data.candidate_appearances)
    || !Array.isArray(data.source_recognition_events) || !Array.isArray(data.candidate_recognition_events)
  ) {
    throw new IdentityReviewApiError('The review detail response is malformed.', 0)
  }
  return data
}

export async function postIdentityReviewDecision(suggestionId, decision) {
  if (decision !== 'accept' && decision !== 'reject') {
    throw new IdentityReviewApiError('Unsupported identity review decision.', 0)
  }
  return requestJson(
    `/api/identity-reviews/${encodeURIComponent(suggestionId)}/${decision}`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: '{}',
    },
  )
}
