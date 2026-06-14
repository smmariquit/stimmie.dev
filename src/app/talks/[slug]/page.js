// src/app/talks/[slug]/page.js

import Image from "next/image";
import Link from "next/link";
import { notFound } from "next/navigation";
import PageShell from "@/components/neo/PageShell";
import { getTalkBySlug, talks } from "@/data/talks";

export function generateStaticParams() {
  return talks.map((t) => ({ slug: t.slug }));
}

export async function generateMetadata({ params }) {
  const { slug } = await params;
  const talk = getTalkBySlug(slug);
  if (!talk) {
    return { title: "Talk not found" };
  }
  return {
    title: talk.title,
    description: talk.description,
    openGraph: {
      title: talk.title,
      description: talk.description,
      images: [{ url: talk.actionPhoto || talk.slidesThumbnail }],
      type: "article",
    },
    twitter: {
      card: "summary_large_image",
      title: talk.title,
      description: talk.description,
      images: [talk.actionPhoto || talk.slidesThumbnail],
    },
  };
}

export default async function TalkPage({ params }) {
  const { slug } = await params;
  const talk = getTalkBySlug(slug);
  if (!talk) notFound();

  const idx = talks.findIndex((t) => t.slug === slug);
  const prev = idx > 0 ? talks[idx - 1] : null;
  const next = idx < talks.length - 1 ? talks[idx + 1] : null;

  const talkDate = new Date(talk.date);
  const isUpcoming = talkDate > new Date();
  const formattedDate = talkDate.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
  const displayImage = talk.actionPhoto || talk.slidesThumbnail;
  const badgeClass =
    talk.type === "Workshop" ? "neo-badge-workshop" : "neo-badge-talk";

  return (
    <PageShell title={talk.title} current="/talks" maxWidth="52rem">
      <p className="mb-3 text-base">
        <Link href="/talks">◄ back to all talks</Link>
      </p>

      <article>
        <p className="m-0 mb-3 flex flex-wrap items-center gap-2">
          <span className={`neo-badge ${badgeClass}`}>{talk.type}</span>
          {isUpcoming && (
            <span className="neo-badge neo-badge-upcoming">Upcoming</span>
          )}
          <time
            className="text-base neo-muted"
            dateTime={talk.date}
            style={{ fontFamily: "var(--neo-ui)" }}
          >
            {formattedDate}
          </time>
        </p>

        <p className="text-base neo-muted mb-4">{talk.event}</p>

        {displayImage && (
          <Image
            src={displayImage}
            alt={`Photo from ${talk.title}`}
            width={1000}
            height={563}
            quality={90}
            sizes="(max-width: 832px) 100vw, 832px"
            className="neo-thumb-lg w-full aspect-video object-cover mb-4"
            priority
          />
        )}

        <p className="text-lg leading-relaxed mb-4">{talk.description}</p>

        {talk.moderator && (
          <p className="text-base neo-muted mb-6">
            Moderated by{" "}
            <Link
              href={talk.moderator.link}
              target="_blank"
              rel="noopener noreferrer"
            >
              {talk.moderator.name}
            </Link>
          </p>
        )}

        <p className="flex flex-wrap gap-3 m-0">
          {talk.slidesLink && (
            <Link
              href={talk.slidesLink}
              target="_blank"
              rel="noopener noreferrer"
              className="neo-link-card inline-block font-bold"
            >
              View Slides ↗
            </Link>
          )}
          {talk.replayLink && (
            <Link
              href={talk.replayLink}
              target="_blank"
              rel="noopener noreferrer"
              className="neo-link-card inline-block font-bold"
            >
              Watch Replay 📺
            </Link>
          )}
        </p>
      </article>

      <nav
        className="mt-8 pt-4 flex justify-between gap-4 text-base"
        style={{ borderTop: "2px dashed #d6008f" }}
        aria-label="Between talks"
      >
        {prev
          ? <Link href={`/talks/${prev.slug}`} className="max-w-[45%]">
              ◄ {prev.title}
            </Link>
          : <span />}
        {next
          ? <Link
              href={`/talks/${next.slug}`}
              className="max-w-[45%] text-right ml-auto"
            >
              {next.title} ►
            </Link>
          : <span />}
      </nav>
    </PageShell>
  );
}
