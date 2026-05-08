import { talks } from "@/data/talks";
import { projects } from "@/data/projects";
import { fetchCloudflareRedirectEntries } from "@/lib/cloudflare";

const SITE_URL = "https://stimmie.dev";

// Static (non-dynamic) routes the site exposes. Keep this list in sync with the
// app router; routes under /r/* are redirect handlers and intentionally excluded.
const staticRoutes = [
  { path: "/", changeFrequency: "weekly", priority: 1.0 },
  { path: "/talks", changeFrequency: "monthly", priority: 0.8 },
  { path: "/projects", changeFrequency: "monthly", priority: 0.8 },
  { path: "/blog", changeFrequency: "monthly", priority: 0.7 },
  { path: "/blog/books", changeFrequency: "yearly", priority: 0.5 },
  { path: "/archive", changeFrequency: "yearly", priority: 0.3 },
  { path: "/archive/v1", changeFrequency: "yearly", priority: 0.2 },
];

// Regenerate the sitemap once an hour. Combined with the fetch-level
// revalidate inside the Cloudflare helper, this means a Bulk Redirect added
// in the CF dashboard appears in /sitemap.xml within ~1h with no redeploy.
export const revalidate = 3600;

export default async function sitemap() {
  const now = new Date();

  const staticEntries = staticRoutes.map((r) => ({
    url: `${SITE_URL}${r.path}`,
    lastModified: now,
    changeFrequency: r.changeFrequency,
    priority: r.priority,
  }));

  const talkEntries = talks.map((t) => ({
    url: `${SITE_URL}/talks/${t.slug}`,
    lastModified: now,
    changeFrequency: "yearly",
    priority: 0.6,
  }));

  const projectEntries = projects.map((p) => ({
    url: `${SITE_URL}/projects/${p.slug}`,
    lastModified: now,
    changeFrequency: "yearly",
    priority: 0.6,
  }));

  // Pull anything we've configured as a Cloudflare Bulk Redirect (e.g. short
  // links like /r/old-talk). Fails safely to [] when env vars are missing or
  // the API is unreachable.
  const cfEntries = await fetchCloudflareRedirectEntries();

  // Dedupe by URL. Hand-curated entries win over CF-derived ones if the same
  // path appears in both — this preserves the priority/changeFrequency we
  // actually want for first-class routes.
  const merged = [
    ...staticEntries,
    ...talkEntries,
    ...projectEntries,
    ...cfEntries,
  ];
  const seen = new Set();
  return merged.filter((e) => {
    if (seen.has(e.url)) return false;
    seen.add(e.url);
    return true;
  });
}
