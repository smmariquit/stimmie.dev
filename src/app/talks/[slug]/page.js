import Image from "next/image";
import Link from "next/link";
import { notFound } from "next/navigation";
import { talks, getTalkBySlug } from "@/data/talks";

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
      images: [{ url: talk.src }],
      type: "article",
    },
    twitter: {
      card: "summary_large_image",
      title: talk.title,
      description: talk.description,
      images: [talk.src],
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

  return (
    <div className="min-h-screen bg-gray-950 text-white">
      <header className="sticky top-0 z-50 bg-gray-950/80 backdrop-blur-md border-b border-gray-800">
        <div className="max-w-4xl mx-auto px-4 py-4 flex items-center justify-between">
          <Link
            href="/talks"
            className="text-sm text-gray-400 hover:text-white transition-colors"
          >
            ← All talks
          </Link>
          <nav className="flex gap-4 text-sm" aria-label="Main navigation">
            <Link
              href="/projects"
              className="text-gray-400 hover:text-white transition-colors"
            >
              Projects
            </Link>
            <Link
              href="/blog"
              className="text-gray-400 hover:text-white transition-colors"
            >
              Blog
            </Link>
          </nav>
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-4 py-8 md:py-12" id="main-content">
        <article>
          <div className="flex flex-wrap items-center gap-2 mb-3">
            <span
              className={`px-2 py-0.5 text-xs rounded-full ${
                talk.type === "Workshop"
                  ? "bg-green-900 text-green-300"
                  : "bg-purple-900 text-purple-300"
              }`}
            >
              {talk.type}
            </span>
            <time className="text-gray-500 text-xs" dateTime={talk.date}>
              {talk.date}
            </time>
            {talk.upcoming && (
              <span className="px-2 py-0.5 text-xs rounded-full bg-yellow-500/20 text-yellow-300 border border-yellow-500/40">
                Upcoming
              </span>
            )}
          </div>

          <h1 className="text-3xl md:text-4xl font-black leading-tight mb-2">
            {talk.title}
          </h1>
          <p className="text-gray-400 text-sm mb-6">{talk.event}</p>

          <div className="relative aspect-video w-full overflow-hidden rounded-2xl border border-gray-800 bg-gray-900 mb-6">
            <Image
              src={talk.src}
              alt={`Slide from ${talk.title}`}
              fill
              sizes="(max-width: 896px) 100vw, 896px"
              className="object-cover"
              priority
            />
          </div>

          <p className="text-gray-300 text-base md:text-lg leading-relaxed mb-8">
            {talk.description}
          </p>

          {talk.slidesLink && (
            <Link
              href={talk.slidesLink}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 bg-blue-600 hover:bg-blue-500 text-white px-5 py-2.5 rounded-lg font-medium transition-colors"
            >
              View Slides →
            </Link>
          )}
        </article>

        <nav
          className="mt-12 pt-6 border-t border-gray-800 flex justify-between gap-4 text-sm"
          aria-label="Pagination between talks"
        >
          {prev ? (
            <Link
              href={`/talks/${prev.slug}`}
              className="text-gray-400 hover:text-white transition-colors max-w-[45%] truncate"
            >
              ← {prev.title}
            </Link>
          ) : (
            <span />
          )}
          {next ? (
            <Link
              href={`/talks/${next.slug}`}
              className="text-gray-400 hover:text-white transition-colors max-w-[45%] truncate text-right ml-auto"
            >
              {next.title} →
            </Link>
          ) : (
            <span />
          )}
        </nav>
      </main>

      <footer className="border-t border-gray-800 mt-12">
        <div className="max-w-4xl mx-auto px-4 py-8 text-center text-gray-500 text-sm">
          <p>© {new Date().getFullYear()} Stimmie. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
}
