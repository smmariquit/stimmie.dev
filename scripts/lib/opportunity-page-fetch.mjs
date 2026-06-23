/**
 * Fetch page HTML and detect bot-block / error interstitials.
 */

const BLOCKED_HTML_MARKERS = [
  "cf-browser-verification",
  "challenge-platform",
  "cdn-cgi/challenge",
  "__cf_chl",
  "checking your browser before accessing",
  "just a moment...",
  "enable javascript and cookies to continue",
  "you have been blocked",
  "blocked.network",
  "verify you are human",
  "bot detection",
];

const BLOCKED_TITLE_PATTERN =
  /just a moment|attention required|access denied|403 forbidden|please wait|error/i;

export function isBlockedPageHtml(html, statusCode) {
  if (statusCode === 403 || statusCode === 451) {
    return true;
  }

  const lower = html.toLowerCase();
  if (BLOCKED_HTML_MARKERS.some((marker) => lower.includes(marker))) {
    return true;
  }

  const title =
    html.match(/<title[^>]*>([^<]*)<\/title>/i)?.[1]?.trim().toLowerCase() ??
    "";
  return BLOCKED_TITLE_PATTERN.test(title);
}

export function extractOgImageFromHtml(html, pageUrl) {
  const patterns = [
    /<meta[^>]+property=["']og:image(?::url)?["'][^>]+content=["']([^"']+)["']/gi,
    /<meta[^>]+content=["']([^"']+)["'][^>]+property=["']og:image(?::url)?["']/gi,
    /<meta[^>]+name=["']twitter:image(?::src)?["'][^>]+content=["']([^"']+)["']/gi,
    /<meta[^>]+content=["']([^"']+)["'][^>]+name=["']twitter:image(?::src)?["']/gi,
  ];

  for (const pattern of patterns) {
    pattern.lastIndex = 0;
    const match = pattern.exec(html);
    if (match?.[1]) {
      try {
        return new URL(match[1].trim(), pageUrl).href;
      } catch {
        // try next pattern
      }
    }
  }

  return null;
}
