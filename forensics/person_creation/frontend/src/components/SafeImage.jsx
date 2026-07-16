import { useEffect, useMemo, useState } from 'react'

import { mediaImageUrl } from '../liveJob.js'

export default function SafeImage({
  path,
  paths = [],
  alt,
  className = '',
  style,
  placeholder = '?',
  placeholderClassName = '',
}) {
  const candidates = useMemo(() => {
    const values = [path, ...(Array.isArray(paths) ? paths : [])]
    return [...new Set(values.map(mediaImageUrl).filter(Boolean))]
  }, [path, paths])
  const candidateKey = candidates.join('\n')
  const [index, setIndex] = useState(0)

  useEffect(() => {
    setIndex(0)
  }, [candidateKey])

  const src = candidates[index]
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
      onError={() => setIndex(current => current + 1)}
    />
  )
}
