import Image from "next/image";
import Link from "next/link";
import { notFound } from "next/navigation";
import { projects, getProjectBySlug } from "@/data/projects";
import MosaicBackground from "@/components/MosaicBackground";

export function generateStaticParams() {
  return projects.map((p) => ({ slug: p.slug }));
}

export async function generateMetadata({ params }) {
  const { slug } = await params;
  const project = getProjectBySlug(slug);
  if (!project) {
    return { title: "Project not found" };
  }
  return {
    title: project.title,
    description: project.description,
    openGraph: {
      title: project.title,
      description: project.description,
      images: [{ url: project.src }],
      type: "article",
    },
    twitter: {
      card: "summary_large_image",
      title: project.title,
      description: project.description,
      images: [project.src],
    },
  };
}

export default async function ProjectPage({ params }) {
  const { slug } = await params;
  const project = getProjectBySlug(slug);
  if (!project) notFound();

  const idx = projects.findIndex((p) => p.slug === slug);
  const prev = idx > 0 ? projects[idx - 1] : null;
  const next = idx < projects.length - 1 ? projects[idx + 1] : null;

  return (
    <div className="min-h-screen bg-black text-white">
      <MosaicBackground />

      <div className="relative" style={{ zIndex: 10 }}>
      <header className="sticky top-0 z-50 bg-gray-950/80 backdrop-blur-md border-b border-gray-800">
        <div className="max-w-4xl mx-auto px-4 py-4 flex items-center justify-between">
          <Link
            href="/projects"
            className="text-sm text-gray-400 hover:text-white transition-colors"
          >
            ← All projects
          </Link>
          <nav className="flex gap-4 text-sm" aria-label="Main navigation">
            <Link
              href="/talks"
              className="text-gray-400 hover:text-white transition-colors"
            >
              Talks
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
          <h1 className="text-3xl md:text-4xl font-black leading-tight mb-4">
            {project.title}
          </h1>

          <div className="relative aspect-video w-full overflow-hidden rounded-2xl border border-gray-800 bg-gray-900 mb-6">
            <Image
              src={project.src}
              alt={`Screenshot of ${project.title}`}
              fill
              sizes="(max-width: 896px) 100vw, 896px"
              className="object-cover"
              priority
            />
          </div>

          {project.tags && project.tags.length > 0 && (
            <div
              className="flex flex-wrap gap-2 mb-6"
              aria-label="Project tags"
            >
              {project.tags.map((tag) => (
                <span
                  key={tag}
                  className="px-2 py-1 bg-gray-800 text-gray-300 text-xs rounded-full"
                >
                  {tag}
                </span>
              ))}
            </div>
          )}

          <p className="text-gray-300 text-base md:text-lg leading-relaxed mb-8">
            {project.description}
          </p>

          {project.link && (
            <Link
              href={project.link}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 bg-blue-600 hover:bg-blue-500 text-white px-5 py-2.5 rounded-lg font-medium transition-colors"
            >
              View Project →
            </Link>
          )}
        </article>

        <nav
          className="mt-12 pt-6 border-t border-gray-800 flex justify-between gap-4 text-sm"
          aria-label="Pagination between projects"
        >
          {prev ? (
            <Link
              href={`/projects/${prev.slug}`}
              className="text-gray-400 hover:text-white transition-colors max-w-[45%] truncate"
            >
              ← {prev.title}
            </Link>
          ) : (
            <span />
          )}
          {next ? (
            <Link
              href={`/projects/${next.slug}`}
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
    </div>
  );
}
