export function artifactUrl(value) {
  if (typeof value !== 'string' || !value.trim()) return null;
  try {
    const url = new URL(value, document.baseURI);
    if (url.protocol === 'https:' || (url.protocol === 'http:' && url.origin === location.origin)) {
      return url.href;
    }
  } catch {
    return null;
  }
  return null;
}

export function setArtifactLink(link, path) {
  const url = artifactUrl(path);
  link.hidden = !url;
  if (url) link.href = url;
  else link.removeAttribute('href');
}

export function previewImage(build) {
  const image = build?.preview_image;
  return artifactUrl(typeof image === 'string' ? image : image?.url);
}

export function validRecordingPath(path) {
  return typeof path === 'string' && (!path || /^[a-z0-9_-]+(?:\/[a-z0-9_-]+)*$/i.test(path));
}
