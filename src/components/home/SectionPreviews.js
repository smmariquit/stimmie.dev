import Image from "next/image";
import Link from "next/link";
import { GitHubCalendar } from "react-github-calendar";
import { blogPosts } from "@/data/blogs";

export function ProjectsPreview({ projects, limit = 3 }) {
  const preview = projects.slice(0, limit);

  return (
    <section>
      <div className="flex justify-between items-end mb-6">
        <h2 className="text-2xl md:text-4xl font-black uppercase tracking-tighter text-white">
          Selected Works
        </h2>
        <Link
          href="/projects"
          className="text-sm uppercase tracking-widest font-bold text-pink-500 hover:text-pink-400"
        >
          View All →
        </Link>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {preview.map((p, idx) => (
          <Link
            key={p.slug}
            href={`/projects/${p.slug}`}
            className="group block overflow-hidden"
          >
            <div className="relative aspect-[16/9] bg-indigo-950/30 overflow-hidden rounded-lg border border-indigo-500/10 group-hover:border-pink-500/50 transition-colors duration-500">
              <Image
                src={p.src}
                alt={p.title}
                width={800}
                height={450}
                className="object-cover w-full h-full group-hover:scale-105 transition-transform duration-700 ease-out"
              />
            </div>
            <div className="mt-3 flex justify-between items-start">
              <h3 className="text-base md:text-lg font-bold uppercase tracking-tight group-hover:text-cyan-400 transition-colors">
                {p.title}
              </h3>
              <span className="text-xs font-bold text-pink-500">
                0{idx + 1}
              </span>
            </div>
          </Link>
        ))}
      </div>
    </section>
  );
}

export function TalksPreview({ talks, limit = 3 }) {
  const preview = talks.slice(0, limit);

  return (
    <section>
      <div className="flex justify-between items-end mb-6">
        <h2 className="text-2xl md:text-4xl font-black uppercase tracking-tighter text-white">
          Talks
        </h2>
        <Link
          href="/talks"
          className="text-sm uppercase tracking-widest font-bold text-pink-500 hover:text-pink-400"
        >
          View All →
        </Link>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {preview.map((t) => {
          const displayImage = t.actionPhoto || t.slidesThumbnail;
          return (
            <Link
              key={t.slug}
              href={`/talks/${t.slug}`}
              className="group block"
            >
              <div className="relative aspect-square overflow-hidden bg-indigo-950/30 rounded-lg border border-indigo-500/10 group-hover:border-cyan-500/50 transition-colors duration-500 mb-3">
                {displayImage && (
                  <Image
                    src={displayImage}
                    alt={t.title}
                    width={400}
                    height={400}
                    className="object-cover w-full h-full group-hover:scale-105 transition-transform duration-700 ease-out"
                  />
                )}
              </div>
              <h3 className="text-sm font-bold uppercase tracking-tight leading-tight group-hover:text-pink-400 transition-colors line-clamp-2">
                {t.title}
              </h3>
            </Link>
          );
        })}
      </div>
    </section>
  );
}

export function ProjectsGrid({ projects, compact = false }) {
  return (
    <div
      className={`grid gap-4 ${compact ? "grid-cols-2" : "grid-cols-1 md:grid-cols-2"}`}
    >
      {projects.map((p) => (
        <Link key={p.slug} href={`/projects/${p.slug}`} className="group block">
          <div
            className={`relative overflow-hidden bg-indigo-950/30 rounded-lg border border-indigo-500/10 group-hover:border-pink-500/50 transition-colors ${compact ? "aspect-[4/3]" : "aspect-[16/9]"}`}
          >
            <Image
              src={p.src}
              alt={p.title}
              width={400}
              height={300}
              className="object-cover w-full h-full group-hover:scale-105 transition-transform duration-500"
            />
          </div>
          <h3 className="mt-2 text-xs font-bold uppercase tracking-tight group-hover:text-cyan-400 transition-colors line-clamp-2">
            {p.title}
          </h3>
        </Link>
      ))}
    </div>
  );
}

export function TalksGrid({ talks, compact = false }) {
  return (
    <div
      className={`grid gap-4 ${compact ? "grid-cols-2" : "grid-cols-1 md:grid-cols-2"}`}
    >
      {talks.map((t) => {
        const displayImage = t.actionPhoto || t.slidesThumbnail;
        return (
          <Link key={t.slug} href={`/talks/${t.slug}`} className="group block">
            <div
              className={`relative overflow-hidden bg-indigo-950/30 rounded-lg border border-indigo-500/10 group-hover:border-cyan-500/50 transition-colors mb-2 ${compact ? "aspect-square" : "aspect-[4/3]"}`}
            >
              {displayImage && (
                <Image
                  src={displayImage}
                  alt={t.title}
                  width={300}
                  height={300}
                  className="object-cover w-full h-full group-hover:scale-105 transition-transform duration-500"
                />
              )}
            </div>
            <h3 className="text-xs font-bold uppercase tracking-tight leading-tight group-hover:text-pink-400 transition-colors line-clamp-3">
              {t.title}
            </h3>
          </Link>
        );
      })}
    </div>
  );
}

export function WritingList({ limit = 4 }) {
  return (
    <div className="space-y-6">
      {blogPosts.slice(0, limit).map((post) => (
        <Link
          key={post.slug}
          href={`/blog/${post.slug}`}
          className="group block"
        >
          <span className="text-xs text-pink-500/70 font-mono mb-1 block">
            {post.date}
          </span>
          <h3 className="text-lg font-bold tracking-tight group-hover:text-cyan-400 transition-colors">
            {post.title}
          </h3>
        </Link>
      ))}
    </div>
  );
}

export function ActivityBlock({ compact = false }) {
  return (
    <div
      className={`bg-indigo-950/20 rounded-xl border border-indigo-500/10 overflow-x-auto hover:border-pink-500/30 transition-colors ${compact ? "p-4" : "p-6"}`}
    >
      <GitHubCalendar
        username="smmariquit"
        colorScheme="dark"
        blockSize={compact ? 10 : 12}
        blockMargin={4}
        fontSize={compact ? 12 : 14}
      />
    </div>
  );
}

export function MediaSection({ mediaData, compact = false }) {
  const { film, book, anime } = mediaData || {};

  const items = [
    film && {
      label: "Film",
      title: film.title,
      subtitle: film.rating ? "★".repeat(Math.round(film.rating)) : null,
      image: film.posterUrl,
    },
    book && {
      label: "Book",
      title: book.title,
      subtitle: book.author,
      image: book.coverUrl,
    },
    anime && {
      label: "Anime",
      title: anime.title,
      subtitle: anime.status,
      image: anime.coverUrl,
    },
  ].filter(Boolean);

  if (items.length === 0) return null;

  return (
    <div
      className={`grid gap-6 ${compact ? "grid-cols-1" : "grid-cols-1 sm:grid-cols-3"}`}
    >
      {items.map((item) => (
        <div key={item.label} className="space-y-3">
          <h3 className="text-xs uppercase tracking-widest text-indigo-400/80 border-b border-purple-900/30 pb-2">
            {item.label}
          </h3>
          <div
            className={`relative overflow-hidden bg-indigo-950/30 rounded-lg border border-indigo-500/10 ${compact ? "aspect-[2/3] max-w-[120px]" : "aspect-[2/3]"}`}
          >
            {item.image && (
              <img
                src={item.image}
                alt=""
                className="object-cover w-full h-full"
              />
            )}
          </div>
          <div>
            <h4 className="font-bold text-sm uppercase text-pink-100">
              {item.title}
            </h4>
            {item.subtitle && (
              <p className="text-xs text-cyan-400/70 mt-1">{item.subtitle}</p>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}

export function Button88x31() {
  return (
    <div className="text-center">
      <p className="text-xs text-indigo-400/80 mb-4 uppercase tracking-widest">
        Grab my button
      </p>
      <img
        src="/stimmie_88x31.gif"
        alt="Stimmie 88x31 Button"
        className="inline-block rounded shadow-[0_0_15px_rgba(236,72,153,0.3)] hover:scale-110 transition-transform"
        style={{ imageRendering: "pixelated" }}
      />
    </div>
  );
}
