// src/app/links/page.js

import Link from "next/link";
import PageShell from "@/components/neo/PageShell";
import { projects } from "@/data/projects";
import { flattenRedirects, getSitemapHosts } from "@/data/redirects";
import { talks } from "@/data/talks";
import { fetchCloudflareRedirectEntries } from "@/lib/cloudflare";

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
  { title: "Changelog", path: "/changelog" },
  { title: "Archive", path: "/archive" },
];

function LinkRow({ href, label, sub, external }) {
  return (
    <li>
      <Link
        href={href}
        target={external ? "_blank" : undefined}
        rel={external ? "noopener noreferrer" : undefined}
        className="neo-link-card group flex items-center justify-between gap-3"
      >
        <span className="font-bold truncate">{label}</span>
        <span className="text-base neo-muted shrink-0 font-mono">{sub}</span>
      </Link>
    </li>
  );
}

export default async function LinksDirectoryPage() {
  const localRedirects = flattenRedirects();
  const cfEntries = await fetchCloudflareRedirectEntries({
    hosts: getSitemapHosts(),
  });

  const pendingLocal = localRedirects.filter(
    (l) =>
      !cfEntries.some((e) =>
        l.sources.some((s) => `https://${s.host}${s.path}` === e.url),
      ),
  );

  return (
    <PageShell
      title="~ directory ~"
      intro="A map of every page and edge-level redirect on this site."
      current="/links"
      maxWidth="64rem"
    >
      <p className="m-0 mb-6">
        <span className="neo-badge neo-badge-workshop">
          ● Live from Cloudflare
        </span>
      </p>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-x-8 gap-y-8">
        <section aria-labelledby="nav-h">
          <h2 id="nav-h" className="neo-section-title">
            navigation
          </h2>
          <ul className="flex flex-col gap-2 list-none p-0 m-0">
            {staticRoutes.map((route) => (
              <LinkRow
                key={route.path}
                href={route.path}
                label={route.title}
                sub={route.path}
              />
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
                sub={`/projects/${project.slug}`}
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
                sub={`/talks/${talk.slug}`}
              />
            ))}
          </ul>
        </section>

        <section aria-labelledby="edge-h">
          <h2 id="edge-h" className="neo-section-title">
            live edge redirects
          </h2>
          {cfEntries.length === 0
            ? <div
                className="p-5 text-center"
                style={{ border: "2px dashed #999" }}
              >
                <p className="m-0">
                  No live Cloudflare redirects found or API unreachable.
                </p>
                <p className="text-base neo-muted m-0 mt-1">
                  Check your CLOUDFLARE_API_TOKEN in Vercel/local env.
                </p>
              </div>
            : <ul className="flex flex-col gap-2 list-none p-0 m-0">
                {cfEntries.map((entry, idx) => {
                  const localMatch = localRedirects.find((l) =>
                    l.sources.some(
                      (s) => `https://${s.host}${s.path}` === entry.url,
                    ),
                  );
                  return (
                    <LinkRow
                      key={idx}
                      href={entry.url}
                      external
                      label={entry.url.replace("https://", "")}
                      sub={localMatch?.category || "ACTIVE @ EDGE"}
                    />
                  );
                })}
              </ul>}

          {pendingLocal.length > 0 && (
            <div className="mt-6">
              <h3
                className="font-bold mb-2"
                style={{ fontFamily: "var(--neo-ui)" }}
              >
                Locally configured (pending sync)
              </h3>
              <ul className="flex flex-col gap-2 list-none p-0 m-0">
                {pendingLocal.map((item, idx) => (
                  <li
                    key={idx}
                    className="neo-link-card flex items-center justify-between gap-3"
                  >
                    <span className="font-bold">{item.slug}</span>
                    <span className="neo-tag shrink-0">
                      {item.placeholder ? "Draft" : "Pending Sync"}
                    </span>
                  </li>
                ))}
              </ul>
            </div>
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
