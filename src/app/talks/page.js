// src/app/talks/page.js

import Image from "next/image";
import Link from "next/link";
import PageShell from "@/components/neo/PageShell";
import { getTotalAudience, talks } from "@/data/talks";

export const metadata = {
  title: "Talks & Workshops",
  description:
    "Talks and workshops Stimmie has given on data science, software engineering, and tech careers.",
};

function formatDate(date) {
  return new Date(date).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

export default function TalksPage() {
  const now = new Date();
  const sorted = [...talks].sort(
    (a, b) => new Date(b.date).getTime() - new Date(a.date).getTime(),
  );
  const totalAudience = getTotalAudience();

  return (
    <PageShell
      title="~ talks & workshops ~"
      intro={`I love sharing knowledge! Across these talks and workshops, I've spoken to an estimated ${totalAudience.toLocaleString()}+ people. Here are the ones I've given on data science, tech, and more.`}
      current="/talks"
      maxWidth="64rem"
    >
      <ul
        className="grid grid-cols-1 sm:grid-cols-2 gap-5 list-none p-0 m-0"
        aria-label="List of talks and workshops"
      >
        {sorted.map((talk) => {
          const isUpcoming = new Date(talk.date) > now;
          const displayImage = talk.actionPhoto || talk.slidesThumbnail;
          const badgeClass =
            talk.type === "Workshop" ? "neo-badge-workshop" : "neo-badge-talk";

          return (
            <li key={talk.slug}>
              <Link
                href={`/talks/${talk.slug}`}
                className="neo-media-card group block h-full"
                aria-label={`Read more about ${talk.title}`}
              >
                <article>
                  {displayImage && (
                    <Image
                      src={displayImage}
                      alt={`Photo from ${talk.title}`}
                      width={800}
                      height={450}
                      quality={90}
                      sizes="(max-width: 640px) 100vw, 360px"
                      className="neo-thumb-lg w-full aspect-video object-cover"
                    />
                  )}
                  <p className="m-0 mt-2 flex flex-wrap items-center gap-2">
                    <span className={`neo-badge ${badgeClass}`}>
                      {talk.type}
                    </span>
                    {isUpcoming && (
                      <span className="neo-badge neo-badge-upcoming">
                        Upcoming
                      </span>
                    )}
                    <time
                      className="text-base neo-muted"
                      dateTime={talk.date}
                      style={{ fontFamily: "var(--neo-ui)" }}
                    >
                      {formatDate(talk.date)}
                    </time>
                  </p>
                  <h2 className="font-bold text-lg mt-1 group-hover:text-[var(--neo-hover-accent)]">
                    {talk.title}
                  </h2>
                  <p className="text-base neo-muted mt-0.5 mb-1">
                    {talk.event}
                  </p>
                  {talk.audienceSize ? (
                    <p
                      className="text-base neo-muted mt-0.5 mb-1"
                      style={{ fontFamily: "var(--neo-ui)" }}
                    >
                      👥 ~{talk.audienceSize.toLocaleString()} in the audience
                    </p>
                  ) : null}
                  <p className="m-0">{talk.description}</p>
                </article>
              </Link>
            </li>
          );
        })}
      </ul>

      <div
        className="mt-8 p-5 text-center"
        style={{ border: "3px double #d6008f", background: "#fff7fb" }}
      >
        <h2 className="neo-section-title">want me to speak at your event?</h2>
        <p className="mx-auto mb-3">
          I&apos;m always excited to share knowledge about data science,
          software engineering, and tech careers.
        </p>
        <div className="flex flex-wrap justify-center gap-3">
          <Link
            href="mailto:semariquit@gmail.com?subject=Speaking Invitation"
            className="neo-link-card inline-block font-bold"
          >
            ✉ get in touch
          </Link>
          <Link
            href="https://cal.stimmie.dev"
            target="_blank"
            rel="noopener noreferrer"
            className="neo-link-card inline-block font-bold"
          >
            ☎ book a call
          </Link>
        </div>
      </div>
    </PageShell>
  );
}
