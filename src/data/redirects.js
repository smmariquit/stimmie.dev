// src/data/redirects.js

// Single source of truth for short-link / subdomain redirects.
//
// Two surfaces consume this map:
//   1. Edge middleware (middleware.js)          — handles <category>.stimmie.dev/<slug>
//   2. App router page (src/app/r/[...slug])    — handles stimmie.dev/r/<slug> on apex
//
// A migration script (scripts/cf-migrate-redirects.mjs) syncs this map into a
// Cloudflare Bulk Redirect List so the redirect happens at the edge, before
// the request ever reaches the Next.js runtime.
//
// Conventions:
//   * Categories are subdomains. Adding a new category here also implies a
//     CNAME for <category>.stimmie.dev. DNS is configured outside this file.
//   * The `links` category is the catch-all for top-level apex short links —
//     /r/freshie, /r/linkedin, etc. — without a category prefix.
//   * Slugs in non-`links` categories are accessed at /r/<category>/<slug>
//     on the apex and at <category>.stimmie.dev/<slug> on the subdomain.
//   * URLs ending with the literal placeholder TODO_TARGET are intentionally
//     unset; they exist as scaffolding for upcoming events. They are skipped
//     by the CF migration script and the sitemap, but the in-app middleware
//     still respects them so a half-configured slug stays a no-op rather than
//     silently 404'ing somewhere unexpected.

export const TODO_TARGET = "https://docs.google.com/presentation/d/...";

// category -> { slug -> destination }
export const redirectMap = {
  workshops: {
    "dsg-codeit-day1": TODO_TARGET,
    "dsg-codeit-day2": TODO_TARGET,
    "nextstep-hacks": TODO_TARGET,
  },
  talks: {
    "qcu-ml-python": TODO_TARGET,
    "dep-ai-study": TODO_TARGET,
  },
  links: {
    freshie: "https://guide.stimmie.dev/freshie",
    hackathons: "https://guide.stimmie.dev/hackathons",
    tech: "https://guide.stimmie.dev/tech",
    linkedin: "https://www.linkedin.com/in/stimmie",
    github: "https://www.github.com/smmariquit",
    instagram: "https://www.instagram.com/friedicecrm",
    osu: "https://osu.ppy.sh/users/14900686",
    spotify: "https://open.spotify.com/user/opzo90f4votlfqmg9rl94qrra",
  },
};

// Apex host that owns the /r/<...> namespace.
export const APEX_HOST = "stimmie.dev";

// The catch-all category — its slugs are exposed at /r/<slug> on apex
// (no category prefix) AND at links.stimmie.dev/<slug>.
export const CATCH_ALL_CATEGORY = "links";

// Hostname for a given category. Centralised so a single rename here
// propagates to middleware, the migration script, and the sitemap.
export function subdomainHostFor(category) {
  return `${category}.${APEX_HOST}`;
}

// All hostnames the sitemap should treat as in-scope for crawlable URLs.
// Order matters only for cosmetics — the sitemap dedupes by URL.
export function getSitemapHosts() {
  return [APEX_HOST, ...Object.keys(redirectMap).map(subdomainHostFor)];
}

/**
 * Subdomain-style lookup. Used by edge middleware: given a request to
 *   <category>.stimmie.dev/<slug>
 * return the destination URL or null.
 */
export function lookupSubdomainRedirect(category, slug) {
  const cat = redirectMap[category];
  if (!cat) return null;
  return cat[slug] || null;
}

/**
 * Apex /r/ lookup. Accepts the path slug as one of:
 *   "linkedin"                 -> links/linkedin
 *   "talks/qcu-ml-python"      -> talks/qcu-ml-python
 *   ["talks","qcu-ml-python"]  -> talks/qcu-ml-python
 */
export function lookupApexRedirect(slugOrParts) {
  const parts = Array.isArray(slugOrParts)
    ? slugOrParts
    : String(slugOrParts || "")
        .split("/")
        .filter(Boolean);

  if (parts.length === 0) return null;

  if (parts.length === 1) {
    return redirectMap[CATCH_ALL_CATEGORY]?.[parts[0]] || null;
  }

  const [category, ...rest] = parts;
  return lookupSubdomainRedirect(category, rest.join("/"));
}

/**
 * Flatten the map into one entry per (host, slug, target) triple.
 * Each logical redirect produces TWO source URLs — the apex /r/ form and
 * the subdomain form — so the CF migration script and the sitemap both
 * see every public URL the redirect responds on.
 *
 * Entries with target === TODO_TARGET are included with `placeholder: true`
 * so callers can skip them without re-checking the sentinel.
 */
export function flattenRedirects() {
  const out = [];

  for (const [category, slugs] of Object.entries(redirectMap)) {
    for (const [slug, target] of Object.entries(slugs)) {
      const placeholder = target === TODO_TARGET;
      const subdomainHost = subdomainHostFor(category);

      const apexPath =
        category === CATCH_ALL_CATEGORY
          ? `/r/${slug}`
          : `/r/${category}/${slug}`;

      out.push({
        category,
        slug,
        target,
        placeholder,
        sources: [
          { host: APEX_HOST, path: apexPath },
          { host: subdomainHost, path: `/${slug}` },
        ],
      });
    }
  }

  return out;
}
