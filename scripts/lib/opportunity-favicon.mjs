/**
 * Resolve site favicon and composite onto a landscape cover canvas.
 */

import sharp from "sharp";

const CANVAS_WIDTH = 1200;
const CANVAS_HEIGHT = 630;
const CANVAS_BACKGROUND = { r: 245, g: 245, b: 240 };
const MAX_FAVICON_HEIGHT = 220;

const FAVICON_LINK_PATTERNS = [
  /<link[^>]+rel=["'](?:shortcut )?icon["'][^>]+href=["']([^"']+)["']/gi,
  /<link[^>]+href=["']([^"']+)["'][^>]+rel=["'](?:shortcut )?icon["']/gi,
  /<link[^>]+rel=["']apple-touch-icon["'][^>]+href=["']([^"']+)["']/gi,
  /<link[^>]+href=["']([^"']+)["'][^>]+rel=["']apple-touch-icon["']/gi,
];

function firstMatch(pattern, html) {
  pattern.lastIndex = 0;
  return pattern.exec(html)?.[1] ?? null;
}

function hostnameVariants(hostname) {
  const variants = [hostname];
  const parts = hostname.split(".");

  if (parts.length > 2) {
    variants.push(parts.slice(-2).join("."));
  }

  return [...new Set(variants)];
}

export function resolveFaviconCandidates(pageUrl, html = "") {
  const page = new URL(pageUrl);
  const candidates = [];

  for (const pattern of FAVICON_LINK_PATTERNS) {
    const href = firstMatch(pattern, html);
    if (!href) {
      continue;
    }
    try {
      candidates.push(new URL(href, pageUrl).href);
    } catch {
      // ignore invalid href
    }
  }

  candidates.push(`${page.origin}/favicon.ico`);

  for (const hostname of hostnameVariants(page.hostname)) {
    candidates.push(
      `https://t1.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://${hostname}&size=128`,
    );
    candidates.push(
      `https://www.google.com/s2/favicons?domain=${hostname}&sz=128`,
    );
    candidates.push(`https://icons.duckduckgo.com/ip3/${hostname}.ico`);
  }

  return [...new Set(candidates)];
}

async function fetchImageBuffer(url, fetchWithTimeout) {
  const response = await fetchWithTimeout(url, {
    headers: { Accept: "image/*,*/*;q=0.8" },
  });
  if (!response.ok) {
    throw new Error(`favicon HTTP ${response.status}`);
  }

  const contentType = response.headers.get("content-type") ?? "";
  if (contentType.startsWith("text/")) {
    throw new Error("favicon response was HTML");
  }

  const buffer = Buffer.from(await response.arrayBuffer());
  if (buffer.byteLength < 32) {
    throw new Error("favicon too small");
  }

  return { buffer, contentType, url: response.url || url };
}

export async function fetchFaviconBuffer({
  pageUrl,
  html,
  fetchWithTimeout,
}) {
  const candidates = resolveFaviconCandidates(pageUrl, html);
  const errors = [];

  for (const candidate of candidates) {
    try {
      return await fetchImageBuffer(candidate, fetchWithTimeout);
    } catch (error) {
      errors.push(
        `${candidate}: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  }

  throw new Error(errors.join(" | ") || "no favicon candidates");
}

export async function composeFaviconLandscape(faviconBuffer) {
  const resized = await sharp(faviconBuffer)
    .resize({
      width: CANVAS_WIDTH - 320,
      height: MAX_FAVICON_HEIGHT,
      fit: "inside",
      withoutEnlargement: false,
    })
    .png()
    .toBuffer();

  return sharp({
    create: {
      width: CANVAS_WIDTH,
      height: CANVAS_HEIGHT,
      channels: 3,
      background: CANVAS_BACKGROUND,
    },
  })
    .composite([{ input: resized, gravity: "center" }])
    .jpeg({ quality: 85 })
    .toBuffer();
}
