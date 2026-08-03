import SafeImage from './SafeImage.jsx'

export default function SafeIdentityImage({ path, alt }) {
  return (
    <SafeImage
      className="live-identity-image"
      placeholderClassName="live-identity-image-placeholder"
      path={path}
      alt={alt}
      placeholder="No face image"
    />
  )
}
