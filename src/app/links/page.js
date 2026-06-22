// src/app/links/page.js

import Link from "next/link";
import PageShell from "@/components/neo/PageShell";
import { projects } from "@/data/projects";
import { talks } from "@/data/talks";
import { fetchCloudflareRedirects } from "@/lib/cloudflare";

export const metadata = {
  title: "Links Directory",
  description:
    "A directory of all pages and live Cloudflare redirects on stimmie.dev",
};

const staticRoutes = [
  { title: "Home", path: "/" },
  { title: "Talks", path: "/talks" },
  { title: "Projects", path: "/projects" },
  { title: "Blog", path: "/blog" },
  { title: "Keys", path: "/keys" },
  { title: "Changelog", path: "/changelog" },
  { title: "Archive", path: "/archive" },
];

function stripProtocol(url) {
  return url.replace(/^https?:\/\//i, "");
}

// Targets are often long Canva/Drive URLs with tracking params. Drop the
// protocol and query/hash so the directory stays readable; the row still
// links to the source, which performs the real redirect.
function prettyTarget(url) {
  return stripProtocol(url).split("?")[0].split("#")[0].replace(/\/$/, "");
}

// A single directory row. `sub` is optional — internal pages omit it (the
// title already says where it goes); redirects use it to show the target.
function LinkRow({ href, label, sub, external }) {
  return (
    <li>
      <Link
        href={href}
        target={external ? "_blank" : undefined}
        rel={external ? "noopener noreferrer" : undefined}
        className="neo-link-card group block"
      >
        <span className="font-bold block break-words group-hover:text-[#cc0066]">
          {label}
        </span>
        {sub && (
          <span className="block text-base neo-muted font-mono mt-0.5 break-all">
            {sub}
          </span>
        )}
      </Link>
    </li>
  );
}

export default async function LinksDirectoryPage() {
  const redirects = await fetchCloudflareRedirects();

  return (
    <PageShell
      title="~ directory ~"
      intro="A map of every page and edge-level redirect on this site."
      current="/links"
      maxWidth="64rem"
    >
      <p className="m-0 mb-6">
        <span className="neo-badge neo-badge-workshop">
          ● Redirects live from Cloudflare
        </span>
      </p>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-x-8 gap-y-8">
        <section aria-labelledby="nav-h">
          <h2 id="nav-h" className="neo-section-title">
            navigation
          </h2>
          <ul className="flex flex-col gap-2 list-none p-0 m-0">
            {staticRoutes.map((route) => (
              <LinkRow key={route.path} href={route.path} label={route.title} />
            ))}
          </ul>
        </section>

        <section aria-labelledby="proj-h">
          <h2 id="proj-h" className="neo-section-title">
            projects
          </h2>
          <ul className="flex flex-col gap-2 list-none p-0 m-0">
            {projects.map((project) => (
              <LinkRow
                key={project.slug}
                href={`/projects/${project.slug}`}
                label={project.title}
              />
            ))}
          </ul>
        </section>

        <section aria-labelledby="talks-h">
          <h2 id="talks-h" className="neo-section-title">
            talks
          </h2>
          <ul className="flex flex-col gap-2 list-none p-0 m-0">
            {talks.map((talk) => (
              <LinkRow
                key={talk.slug}
                href={`/talks/${talk.slug}`}
                label={talk.title}
              />
            ))}
          </ul>
        </section>

        <section aria-labelledby="edge-h">
          <h2 id="edge-h" className="neo-section-title">
            edge redirects
          </h2>
          {redirects.length === 0 ? (
            <div
              className="p-5 text-center"
              style={{ border: "2px dashed #999" }}
            >
              <p className="m-0">
                No live Cloudflare redirects found or the API was unreachable.
              </p>
              <p className="text-base neo-muted m-0 mt-1">
                These load from your Bulk Redirect lists at build time.
              </p>
            </div>
          ) : (
            <ul className="flex flex-col gap-2 list-none p-0 m-0">
              {redirects.map((r, idx) => {
                const source = stripProtocol(r.source);
                const href = r.source.match(/^https?:\/\//i)
                  ? r.source
                  : `https://${r.source}`;
                return (
                  <LinkRow
                    key={`${r.source}-${idx}`}
                    href={href}
                    external
                    label={source}
                    sub={`→ ${prettyTarget(r.target)}`}
                  />
                );
              })}
            </ul>
          )}
        </section>
      </div>

      <p className="mt-8 text-base neo-muted">
        Edge data refreshed hourly ·{" "}
        <Link href="/sitemap.xml">XML Sitemap</Link>
      </p>
    </PageShell>
  );
}
