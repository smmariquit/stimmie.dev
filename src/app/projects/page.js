"use client";
import Image from "next/image";
import Link from "next/link";
import { motion } from "framer-motion";

const projects = [
  {
    title: 'HearthCraft',
    src: '/projects/project1.jpg',
    description: 'At 13 years old, I decided to take my deep enjoyment of Minecraft, learn to set up a Minecraft server, worked with both managed and bare-metal servers, set up Java plugins and Docker instances, and created a multiplayer experience that served as a safe space for 10,000+ over the span of 6+ years.',
    tags: ['Minecraft', 'Docker', 'Java', 'Community'],
    link: null,
  },
  {
    title: 'Atlas Of My Skies',
    src: '/projects/project2.jpg',
    description: 'Telling the story of the skies.',
    tags: ['Photography', 'Storytelling'],
    link: null,
  },
  {
    title: 'BARLO: Bayani Alert and Response for Local Operations',
    src: '/projects/project3.jpg',
    description: "Predict a storm's economic impact from typhoon forecast data. Get insights on how to pre-emptively place logistics.",
    tags: ['Data Science', 'ML', 'Disaster Response'],
    link: null,
  },
  {
    title: 'Pharmadash',
    src: '/projects/project4.jpg',
    description: 'A hackathon project for efficient pharmaceutical inventory management and distribution.',
    tags: ['Hackathon', 'Healthcare', 'Inventory'],
    link: null,
  },
  {
    title: 'Punnett Square Visualizer',
    src: '/projects/project5.jpg',
    description: 'An interactive tool to visualize genetic crosses. Used in tutorials for science high school students.',
    tags: ['Education', 'Biology', 'Visualization'],
    link: null,
  },
];

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
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6" role="list" aria-label="List of projects">
          {projects.map((project, idx) => (
            <motion.article
              key={idx}
              className="bg-gray-900 rounded-2xl overflow-hidden border border-gray-800 hover:border-gray-700 transition-colors"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: idx * 0.1 }}
              role="listitem"
            >
              <div className="aspect-video relative overflow-hidden">
                <Image
                  src={project.src}
                  alt={`Screenshot of ${project.title}`}
                  fill
                  className="object-cover hover:scale-105 transition-transform duration-300"
                />
              </div>
              <div className="p-6">
                <h2 className="text-xl font-bold mb-2">{project.title}</h2>
                <p className="text-gray-400 text-sm mb-4 leading-relaxed">
                  {project.description}
                </p>
                {project.tags && (
                  <div className="flex flex-wrap gap-2 mb-4">
                    {project.tags.map((tag, tagIdx) => (
                      <span
                        key={tagIdx}
                        className="px-2 py-1 bg-gray-800 text-gray-300 text-xs rounded-full"
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                )}
                {project.link && (
                  <Link
                    href={project.link}
                    className="inline-flex items-center gap-2 text-blue-400 hover:text-blue-300 text-sm"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    View Project →
                  </Link>
                )}
              </div>
            </motion.article>
          ))}
        </div>
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
