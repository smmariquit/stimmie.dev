// src/lib/cloudflare.js

/**
 * Cloudflare API helpers for the sitemap.
 *
 * Cloudflare doesn't fire webhooks when you create a Bulk Redirect or change
 * a DNS record, so there's no event-driven "hook" to consume. Instead we pull
 * from the API every time the sitemap regenerates. The Next.js fetch cache
 * (`next.revalidate`) keeps API traffic to roughly one request per origin per
 * hour, regardless of how often /sitemap.xml is requested.
 *
 * Failure is intentionally silent: a transient CF API error should never
 * break the sitemap or the build. Callers will receive an empty array and
 * a console warning.
 *
 * Required env vars (all optional — feature is no-op without them):
 *   CLOUDFLARE_API_TOKEN       - scoped token, needs Account.Filter Lists: Read
 *   CLOUDFLARE_ACCOUNT_ID      - the account that owns the Bulk Redirect lists
 *
 * Optional env vars:
 *   CLOUDFLARE_SITEMAP_HOSTS   - comma-separated host whitelist; sources whose
 *                                host matches any entry are kept. Defaults to
 *                                the value of CLOUDFLARE_SITEMAP_HOST or
 *                                "stimmie.dev" if neither is set.
 *   CLOUDFLARE_SITEMAP_HOST    - legacy single-host fallback (still honoured).
 *   CLOUDFLARE_REVALIDATE      - cache TTL in seconds for CF API calls
 *                                (default: 3600).
 */

const CF_API_BASE = "https://api.cloudflare.com/client/v4";
const DEFAULT_HOST = "stimmie.dev";
const DEFAULT_REVALIDATE = 3600;

/**
 * Fetch every Bulk Redirect rule across all redirect-kind Lists in the
 * configured account, then return the ones whose source host matches any
 * configured site host. Wildcard sources (`*` in the source URL) are
 * skipped — sitemaps should only contain concrete, crawlable URLs.
 *
 * @param {object} [opts]
 * @param {string[]} [opts.hosts] - optional override for the host whitelist.
 *   When provided, env vars are ignored. Useful for tests.
 * @returns {Promise<Array<{url: string, lastModified: Date, changeFrequency: string, priority: number}>>}
 */
export async function fetchCloudflareRedirectEntries(opts = {}) {
  const token = process.env.CLOUDFLARE_API_TOKEN;
  const accountId = process.env.CLOUDFLARE_ACCOUNT_ID;
  const hosts = opts.hosts || resolveHosts();

  if (!token || !accountId) {
    // Don't warn here — this is the expected path during local dev where
    // contributors won't have CF credentials. Warning only on actual errors.
    return [];
  }

  if (hosts.length === 0) {
    console.warn(
      "[sitemap] No CLOUDFLARE_SITEMAP_HOST(S) configured; CF redirects skipped."
    );
    return [];
  }

  try {
    const lists = await cfFetch(
      `${CF_API_BASE}/accounts/${accountId}/rules/lists`,
      token
    );

    const redirectLists = (lists || []).filter((l) => l.kind === "redirect");

    // Fan out one request per list. Lists are typically few (<10) so we
    // don't need a concurrency limiter; a sequential loop also keeps us
    // well under CF's per-token rate limits.
    const items = [];
    for (const list of redirectLists) {
      const listItems = await fetchAllListItems(accountId, list.id, token);
      items.push(...listItems);
    }

    return items.map((it) => toSitemapEntry(it, hosts)).filter(Boolean);
  } catch (err) {
    console.warn(
      "[sitemap] Cloudflare redirect sync failed; sitemap will exclude " +
        "CF-managed redirects this regeneration:",
      err.message
    );
    return [];
  }
}

/**
 * Resolve the host whitelist from env vars. Order of precedence:
 *   1. CLOUDFLARE_SITEMAP_HOSTS (comma-separated)
 *   2. CLOUDFLARE_SITEMAP_HOST  (single host, legacy)
 *   3. DEFAULT_HOST
 */
function resolveHosts() {
  const multi = process.env.CLOUDFLARE_SITEMAP_HOSTS;
  if (multi) {
    return multi
      .split(",")
      .map((h) => h.trim().toLowerCase())
      .filter(Boolean);
  }
  const single = process.env.CLOUDFLARE_SITEMAP_HOST || DEFAULT_HOST;
  return [single.trim().toLowerCase()];
}

/**
 * Walk every page of a redirect list. CF caps `per_page` at 100; lists with
 * more than that paginate via `cursor` in `result_info`.
 */
async function fetchAllListItems(accountId, listId, token) {
  const out = [];
  let cursor;

  // Bounded loop; protects against an unexpected paging bug returning the
  // same cursor forever.
  for (let page = 0; page < 50; page++) {
    const url = new URL(
      `${CF_API_BASE}/accounts/${accountId}/rules/lists/${listId}/items`
    );
    url.searchParams.set("per_page", "100");
    if (cursor) url.searchParams.set("cursor", cursor);

    const { result, result_info } = await cfFetchRaw(url.toString(), token);
    if (Array.isArray(result)) out.push(...result);

    cursor = result_info?.cursors?.after;
    if (!cursor) break;
  }

  return out;
}

/**
 * Issues a single CF API GET, retries once on 5xx, and unwraps the
 * `result` envelope. Throws on non-success.
 */
async function cfFetch(url, token, attempt = 0) {
  const { result } = await cfFetchRaw(url, token, attempt);
  return result;
}

async function cfFetchRaw(url, token, attempt = 0) {
  const revalidate = Number(
    process.env.CLOUDFLARE_REVALIDATE || DEFAULT_REVALIDATE
  );

  const res = await fetch(url, {
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
    },
    // The Next.js fetch cache does the heavy lifting here: same URL +
    // headers within `revalidate` seconds returns cached body without
    // hitting CF.
    next: { revalidate },
  });

  if (!res.ok) {
    if (attempt < 1 && res.status >= 500) {
      return cfFetchRaw(url, token, attempt + 1);
    }
    throw new Error(`Cloudflare API ${res.status} ${res.statusText} for ${url}`);
  }

  const json = await res.json();
  if (json && json.success === false) {
    const msg = (json.errors || [])
      .map((e) => `${e.code}: ${e.message}`)
      .join("; ");
    throw new Error(`Cloudflare API error: ${msg || "unknown"}`);
  }
  return json;
}

/**
 * Convert a single Bulk Redirect rule into a sitemap entry. Returns null
 * for rules that are wildcards, point at a host outside the whitelist,
 * or look like non-page resources.
 *
 * CF Bulk Redirect rules look like:
 *   {
 *     id, modified_on,
 *     redirect: {
 *       source_url: "stimmie.dev/old-talk",
 *       target_url: "https://docs.google.com/...",
 *       status_code: 301, include_subdomains, subpath_matching, ...
 *     }
 *   }
 */
function toSitemapEntry(item, hosts) {
  const r = item?.redirect;
  if (!r || typeof r.source_url !== "string") return null;

  let src = r.source_url.trim();
  // Normalise: strip protocol; we'll match host-then-path.
  src = src.replace(/^https?:\/\//i, "");

  // Sitemaps should only list concrete URLs. Wildcards / subpath patterns
  // can't be enumerated meaningfully.
  if (src.includes("*")) return null;

  const lower = src.toLowerCase();

  // Find the longest matching host so e.g. "links.stimmie.dev" wins over
  // "stimmie.dev" when both are in the whitelist.
  const matchedHost = hosts
    .slice()
    .sort((a, b) => b.length - a.length)
    .find(
      (h) =>
        lower === h ||
        lower.startsWith(`${h}/`) ||
        lower.startsWith(`${h}?`) ||
        lower.startsWith(`${h}#`)
    );
  if (!matchedHost) return null;

  const path = src.slice(matchedHost.length) || "/";

  // Filter out URLs that obviously aren't HTML pages.
  if (/\.(xml|json|txt|ico|png|jpe?g|gif|webp|svg|css|js)(\?|#|$)/i.test(path)) {
    return null;
  }

  const lastModified = item.modified_on
    ? new Date(item.modified_on)
    : new Date();

  return {
    url: `https://${matchedHost}${path}`,
    lastModified,
    changeFrequency: "monthly",
    priority: 0.5,
  };
}
