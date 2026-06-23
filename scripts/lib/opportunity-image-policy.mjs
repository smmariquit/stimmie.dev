/**
 * Reject images from this site or other personal assets when fetching covers.
 */

const DISALLOWED_HOSTS = new Set([
  "stimmie.dev",
  "www.stimmie.dev",
  "workshops.stimmie.dev",
  "guide.stimmie.dev",
  "hearthcraft.stimmie.dev",
  "atlas-of-my-skies.stimmie.dev",
  "room-tba.stimmie.dev",
  "gradesim.stimmie.dev",
  "localhost",
  "127.0.0.1",
]);

const DISALLOWED_PATH_PATTERNS = [
  /\/talks\//i,
  /\/workshops\//i,
  /\/r\/workshops\//i,
  /\/r\/talks\//i,
  /\/projects\//i,
  /\/archive\//i,
  /navigating-hackathons/i,
];

export function isDisallowedImageUrl(imageUrl, pageUrl = "") {
  let parsed;
  try {
    parsed = new URL(imageUrl);
  } catch {
    return true;
  }

  const host = parsed.hostname.toLowerCase().replace(/^www\./, "");
  const bareHost = parsed.hostname.toLowerCase();

  if (
    DISALLOWED_HOSTS.has(bareHost) ||
    DISALLOWED_HOSTS.has(host) ||
    host.endsWith(".stimmie.dev")
  ) {
    return true;
  }

  const target = `${parsed.pathname}${parsed.search}`;
  if (DISALLOWED_PATH_PATTERNS.some((pattern) => pattern.test(target))) {
    return true;
  }

  if (pageUrl) {
    try {
      const pageHost = new URL(pageUrl).hostname.toLowerCase();
      if (pageHost.endsWith(".stimmie.dev") || DISALLOWED_HOSTS.has(pageHost)) {
        return true;
      }
    } catch {
      // ignore invalid page URL
    }
  }

  return false;
}

export function assertAllowedImageUrl(imageUrl, pageUrl = "") {
  if (isDisallowedImageUrl(imageUrl, pageUrl)) {
    throw new Error("image URL points to disallowed personal/site asset");
  }
}
