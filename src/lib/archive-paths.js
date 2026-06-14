/**
 * Archive asset path helpers.
 * Archives must only reference files under /archive/* — never shared live-site paths.
 */

export const ARCHIVE_MOSAIC_BASE = "/archive/mosaic";
export const ARCHIVE_MOSAIC_IMAGES = `${ARCHIVE_MOSAIC_BASE}/images`;
export const ARCHIVE_MOSAIC_THUMBS = `${ARCHIVE_MOSAIC_BASE}/images_small`;

export function archiveMosaicImage(file) {
  return `${ARCHIVE_MOSAIC_IMAGES}/${file}`;
}

export function archiveMosaicThumb(file) {
  const stem = file.replace(/\.[^.]+$/, "");
  return `${ARCHIVE_MOSAIC_THUMBS}/${stem}.webp`;
}

export const ARCHIVE_V1 = "/archive/v1";
export const ARCHIVE_V2 = "/archive/v2";
