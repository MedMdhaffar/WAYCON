import { useEffect, useMemo, useState } from 'react'

import { identityImageUrl } from '../liveJob.js'

export default function SafeIdentityImage({ path, alt }) {
  const [failed, setFailed] = useState(false)
  const src = useMemo(() => identityImageUrl(path), [path])

  useEffect(() => {
    setFailed(false)
  }, [src])

  if (!src || failed) {
    return (
      <div className="live-identity-image-placeholder" role="img" aria-label="Face image unavailable">
        No face image
      </div>
    )
  }

  return (
    <img
      className="live-identity-image"
      src={src}
      alt={alt}
      onError={() => setFailed(true)}
    />
  )
}
