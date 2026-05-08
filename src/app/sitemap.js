import { talks } from "@/data/talks";
import { projects } from "@/data/projects";

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

export default function sitemap() {
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

  return [...staticEntries, ...talkEntries, ...projectEntries];
}
