"use client";
import Image from "next/image";
import Link from "next/link";
import { motion } from "framer-motion";
import { projects } from "@/data/projects";

export default function ProjectsPage() {
  return (
    <div className="min-h-screen bg-gray-950 text-white">
      {/* Header */}
      <header className="sticky top-0 z-50 bg-gray-950/80 backdrop-blur-md border-b border-gray-800">
        <div className="max-w-6xl mx-auto px-4 py-4 flex items-center justify-between">
          <Link href="/" className="text-xl font-bold hover:text-blue-400 transition-colors">
            ← Stimmie
          </Link>
          <nav className="flex gap-4" aria-label="Main navigation">
            <Link href="/talks" className="text-gray-400 hover:text-white transition-colors">
              Talks
            </Link>
            <Link href="/blog" className="text-gray-400 hover:text-white transition-colors">
              Blog
            </Link>
          </nav>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-6xl mx-auto px-4 py-12" id="main-content">
        <div className="mb-12">
          <h1 className="text-4xl md:text-5xl font-black mb-4">🚀 Projects</h1>
          <p className="text-gray-400 text-lg max-w-2xl">
            A collection of projects I&apos;ve worked on over the years — from Minecraft servers to data science tools.
          </p>
        </div>

        {/* Projects Grid */}
        <ul
          className="grid grid-cols-1 md:grid-cols-2 gap-6 list-none p-0"
          aria-label="List of projects"
        >
          {projects.map((project, idx) => (
            <li key={project.slug}>
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: idx * 0.1 }}
                className="h-full"
              >
                <Link
                  href={`/projects/${project.slug}`}
                  className="group block h-full bg-gray-900 rounded-2xl overflow-hidden border border-gray-800 hover:border-gray-700 transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
                  aria-label={`View project: ${project.title}`}
                >
                  <article>
                    <div className="aspect-video relative overflow-hidden">
                      <Image
                        src={project.src}
                        alt={`Screenshot of ${project.title}`}
                        fill
                        sizes="(max-width: 768px) 100vw, 50vw"
                        className="object-cover group-hover:scale-105 transition-transform duration-300"
                      />
                    </div>
                    <div className="p-6">
                      <h2 className="text-xl font-bold mb-2 group-hover:text-blue-300 transition-colors">
                        {project.title}
                      </h2>
                      <p className="text-gray-400 text-sm mb-4 leading-relaxed line-clamp-3">
                        {project.description}
                      </p>
                      {project.tags && (
                        <div className="flex flex-wrap gap-2">
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
                    </div>
                  </article>
                </Link>
              </motion.div>
            </li>
          ))}
        </ul>
      </main>

      {/* Footer */}
      <footer className="border-t border-gray-800 mt-12">
        <div className="max-w-6xl mx-auto px-4 py-8 text-center text-gray-500 text-sm">
          <p>© {new Date().getFullYear()} Stimmie. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
}
