// src/app/projects/page.js

import Image from "next/image";
import Link from "next/link";
import PageShell from "@/components/neo/PageShell";
import { projects } from "@/data/projects";

export const metadata = {
  title: "Projects",
  description:
    "A collection of projects Stimmie has worked on, from Minecraft servers to data science tools.",
};

export default function ProjectsPage() {
  return (
    <PageShell
      title="~ my projects ~"
      intro="A pile of things I've built over the years, from Minecraft servers to data science tools. Click any card to read more."
      current="/projects"
      maxWidth="64rem"
    >
      <ul
        className="grid grid-cols-1 sm:grid-cols-2 gap-5 list-none p-0 m-0"
        aria-label="List of projects"
      >
        {projects.map((project) => (
          <li key={project.slug}>
            <Link
              href={`/projects/${project.slug}`}
              className="neo-media-card group block h-full"
              aria-label={`View project: ${project.title}`}
            >
              <article>
                <Image
                  src={project.src}
                  alt={`Screenshot of ${project.title}`}
                  width={800}
                  height={450}
                  quality={90}
                  sizes="(max-width: 640px) 100vw, 360px"
                  className="neo-thumb-lg w-full aspect-video object-cover"
                />
                <h2 className="font-bold text-xl mt-2 group-hover:text-[#cc0066]">
                  {project.title}
                </h2>
                {project.date && (
                  <p className="text-base neo-muted mt-0.5 font-mono">
                    {project.date}
                  </p>
                )}
                <p className="mt-1 mb-2">{project.description}</p>
                {project.tags && (
                  <p className="m-0">
                    {project.tags.map((tag) => (
                      <span key={tag} className="neo-tag">
                        {tag}
                      </span>
                    ))}
                  </p>
                )}
              </article>
            </Link>
          </li>
        ))}
      </ul>
    </PageShell>
  );
}
