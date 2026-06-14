// src/app/projects/[slug]/page.js

import Image from "next/image";
import Link from "next/link";
import { notFound } from "next/navigation";
import PageShell from "@/components/neo/PageShell";
import { getProjectBySlug, projects } from "@/data/projects";

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
    <PageShell title={project.title} current="/projects" maxWidth="52rem">
      <p className="mb-4 text-base">
        <Link href="/projects">◄ back to all projects</Link>
      </p>

      <article>
        <Image
          src={project.src}
          alt={`Screenshot of ${project.title}`}
          width={1000}
          height={563}
          quality={90}
          sizes="(max-width: 832px) 100vw, 832px"
          className="neo-thumb-lg w-full aspect-video object-cover mb-4"
          priority
        />

        {project.tags && project.tags.length > 0 && (
          <p className="m-0 mb-3">
            {project.tags.map((tag) => (
              <span key={tag} className="neo-tag">
                {tag}
              </span>
            ))}
          </p>
        )}

        <p className="text-lg leading-relaxed mb-6">{project.description}</p>

        {project.link && (
          <p>
            <Link
              href={project.link}
              target="_blank"
              rel="noopener noreferrer"
              className="neo-link-card inline-block font-bold"
            >
              View Project ↗
            </Link>
          </p>
        )}
      </article>

      <nav
        className="mt-8 pt-4 flex justify-between gap-4 text-base"
        style={{ borderTop: "2px dashed #d6008f" }}
        aria-label="Between projects"
      >
        {prev
          ? <Link href={`/projects/${prev.slug}`} className="max-w-[45%]">
              ◄ {prev.title}
            </Link>
          : <span />}
        {next
          ? <Link
              href={`/projects/${next.slug}`}
              className="max-w-[45%] text-right ml-auto"
            >
              {next.title} ►
            </Link>
          : <span />}
      </nav>
    </PageShell>
  );
}
