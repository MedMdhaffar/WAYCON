import { useEffect, useMemo, useState } from 'react'

import { mediaImageUrl } from '../liveJob.js'

const failedUrlsByScope = new Map()
const MAX_FAILURE_SCOPES = 64

function failureSet(scope) {
  if (!failedUrlsByScope.has(scope)) {
    failedUrlsByScope.set(scope, new Set())
    if (failedUrlsByScope.size > MAX_FAILURE_SCOPES) {
      failedUrlsByScope.delete(failedUrlsByScope.keys().next().value)
    }
  }
  return failedUrlsByScope.get(scope)
}

export function clearFailedMediaUrls() {
  failedUrlsByScope.clear()
}

export default function SafeImage({
  path,
  paths = [],
  alt,
  className = '',
  style,
  placeholder = '?',
  placeholderClassName = '',
  version = '',
}) {
  const canonicalCandidates = useMemo(() => {
    const values = [path, ...(Array.isArray(paths) ? paths : [])]
    return [...new Set(values.map(mediaImageUrl).filter(Boolean))]
  }, [path, paths])
  const candidateKey = canonicalCandidates.join('\n')
  const failureScope = `${version}\n${candidateKey}`
  const [index, setIndex] = useState(0)

  useEffect(() => {
    setIndex(0)
  }, [candidateKey, version])

  const failed = failureSet(failureScope)
  let candidateIndex = index
  while (
    candidateIndex < canonicalCandidates.length
    && failed.has(canonicalCandidates[candidateIndex])
  ) {
    candidateIndex += 1
  }
  const src = canonicalCandidates[candidateIndex]
  if (!src) {
    return (
      <div
        className={placeholderClassName}
        style={style}
        role="img"
        aria-label={`${alt || 'Image'} unavailable`}
      >
        {placeholder}
      </div>
    )
  }

  return (
    <img
      className={className}
      style={style}
      src={src}
      alt={alt}
      onError={() => {
        failed.add(src)
        setIndex(candidateIndex + 1)
      }}
    />
  )
}
