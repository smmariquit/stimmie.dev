import Image from "next/image";
import Link from "next/link";
import { GitHubCalendar } from "react-github-calendar";

function SnapshotCard({ label, labelClass, children, href, className = "" }) {
  const inner = (
    <div
      className={`bg-indigo-950/20 rounded-xl border border-indigo-500/10 p-4 h-full hover:border-pink-500/30 transition-colors ${className}`}
    >
      <h3
        className={`text-[10px] font-bold uppercase tracking-widest mb-3 ${labelClass}`}
      >
        {label}
      </h3>
      {children}
    </div>
  );

  if (href) {
    return (
      <Link href={href} className="block h-full group">
        {inner}
      </Link>
    );
  }

  return inner;
}

function NowChips({ film, book, anime, compact = false }) {
  const items = [
    film && { label: "Film", title: film.title, image: film.posterUrl },
    book && { label: "Book", title: book.title, image: book.coverUrl },
    anime && { label: "Anime", title: anime.title, image: anime.coverUrl },
  ].filter(Boolean);

  if (items.length === 0) {
    return <p className="text-xs text-gray-500">Nothing tracked right now.</p>;
  }

  return (
    <div className={`flex gap-2 ${compact ? "flex-col" : "flex-wrap"}`}>
      {items.map((item) => (
        <div key={item.label} className="flex items-center gap-2 min-w-0">
          {item.image && (
            <div
              className={`relative flex-shrink-0 overflow-hidden rounded border border-indigo-500/10 bg-indigo-950/30 ${compact ? "w-10 h-14" : "w-8 h-11"}`}
            >
              <img
                src={item.image}
                alt=""
                className="object-cover w-full h-full"
              />
            </div>
          )}
          <div className="min-w-0">
            <span className="text-[9px] uppercase tracking-widest text-indigo-400/70 block">
              {item.label}
            </span>
            <span className="text-xs font-medium text-pink-100 truncate block">
              {item.title}
            </span>
          </div>
        </div>
      ))}
    </div>
  );
}

function GitHubMini({ blockSize = 8 }) {
  return (
    <div className="overflow-x-auto -mx-1">
      <GitHubCalendar
        username="smmariquit"
        colorScheme="dark"
        blockSize={blockSize}
        blockMargin={3}
        fontSize={10}
      />
    </div>
  );
}

export default function SnapshotStrip({
  variant = "strip",
  featuredProject,
  featuredTalk,
  mediaData,
}) {
  const { film, book, anime } = mediaData || {};
  const talkImage = featuredTalk?.actionPhoto || featuredTalk?.slidesThumbnail;

  if (variant === "grid") {
    return (
      <div className="grid grid-cols-2 gap-3">
        {featuredProject && (
          <SnapshotCard
            label="Featured Work"
            labelClass="text-pink-400"
            href={`/projects/${featuredProject.slug}`}
          >
            <div className="flex gap-3 items-center">
              <div className="relative w-14 h-10 flex-shrink-0 overflow-hidden rounded border border-indigo-500/10">
                <Image
                  src={featuredProject.src}
                  alt=""
                  width={112}
                  height={80}
                  className="object-cover w-full h-full"
                />
              </div>
              <p className="text-xs font-bold uppercase leading-tight text-gray-200 group-hover:text-cyan-400 transition-colors line-clamp-2">
                {featuredProject.title}
              </p>
            </div>
          </SnapshotCard>
        )}
        {featuredTalk && (
          <SnapshotCard
            label="Latest Talk"
            labelClass="text-cyan-400"
            href={`/talks/${featuredTalk.slug}`}
          >
            <div className="flex gap-3 items-center">
              {talkImage && (
                <div className="relative w-14 h-14 flex-shrink-0 overflow-hidden rounded border border-indigo-500/10">
                  <Image
                    src={talkImage}
                    alt=""
                    width={56}
                    height={56}
                    className="object-cover w-full h-full"
                  />
                </div>
              )}
              <p className="text-xs font-bold uppercase leading-tight text-gray-200 group-hover:text-pink-400 transition-colors line-clamp-3">
                {featuredTalk.title}
              </p>
            </div>
          </SnapshotCard>
        )}
        <SnapshotCard label="Activity" labelClass="text-pink-400">
          <GitHubMini blockSize={7} />
        </SnapshotCard>
        <SnapshotCard label="Now" labelClass="text-cyan-400">
          <NowChips film={film} book={book} anime={anime} compact />
        </SnapshotCard>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 max-w-7xl mx-auto px-6 md:px-12 pb-8 relative z-10">
      {featuredProject && (
        <SnapshotCard
          label="Featured Work"
          labelClass="text-pink-400"
          href={`/projects/${featuredProject.slug}`}
        >
          <div className="flex gap-3 items-start">
            <div className="relative w-20 h-14 flex-shrink-0 overflow-hidden rounded border border-indigo-500/10">
              <Image
                src={featuredProject.src}
                alt=""
                width={160}
                height={112}
                className="object-cover w-full h-full"
              />
            </div>
            <p className="text-sm font-bold uppercase leading-tight text-gray-200 group-hover:text-cyan-400 transition-colors line-clamp-3">
              {featuredProject.title}
            </p>
          </div>
        </SnapshotCard>
      )}
      {featuredTalk && (
        <SnapshotCard
          label="Latest Talk"
          labelClass="text-cyan-400"
          href={`/talks/${featuredTalk.slug}`}
        >
          <div className="flex gap-3 items-start">
            {talkImage && (
              <div className="relative w-16 h-16 flex-shrink-0 overflow-hidden rounded border border-indigo-500/10">
                <Image
                  src={talkImage}
                  alt=""
                  width={64}
                  height={64}
                  className="object-cover w-full h-full"
                />
              </div>
            )}
            <p className="text-xs font-bold uppercase leading-tight text-gray-200 group-hover:text-pink-400 transition-colors line-clamp-4">
              {featuredTalk.title}
            </p>
          </div>
        </SnapshotCard>
      )}
      <SnapshotCard label="Activity" labelClass="text-pink-400">
        <GitHubMini blockSize={8} />
      </SnapshotCard>
      <SnapshotCard label="Now" labelClass="text-cyan-400">
        <NowChips film={film} book={book} anime={anime} />
      </SnapshotCard>
    </div>
  );
}
