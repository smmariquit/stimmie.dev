"use client";
import Image from "next/image";
import Link from "next/link";
import { motion } from "framer-motion";
import { talks } from "@/data/talks";

export default function TalksPage() {
  return (
    <div className="min-h-screen bg-gray-950 text-white">
      {/* Header */}
      <header className="sticky top-0 z-50 bg-gray-950/80 backdrop-blur-md border-b border-gray-800">
        <div className="max-w-6xl mx-auto px-4 py-4 flex items-center justify-between">
          <Link href="/" className="text-xl font-bold hover:text-blue-400 transition-colors">
            ← Stimmie
          </Link>
          <nav className="flex gap-4" aria-label="Main navigation">
            <Link href="/projects" className="text-gray-400 hover:text-white transition-colors">
              Projects
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
          <h1 className="text-4xl md:text-5xl font-black mb-4">🎤 Talks & Workshops</h1>
          <p className="text-gray-400 text-lg max-w-2xl">
            I love sharing knowledge! Here are the talks and workshops I&apos;ve given on data science, tech, and more.
          </p>
        </div>

        {/* Filter Tabs */}
        <div className="flex gap-2 mb-8 flex-wrap">
          <span className="px-4 py-2 bg-blue-600 text-white rounded-full text-sm font-medium">
            All
          </span>
          <span className="px-4 py-2 bg-gray-800 text-gray-300 rounded-full text-sm cursor-pointer hover:bg-gray-700 transition-colors">
            Workshops
          </span>
          <span className="px-4 py-2 bg-gray-800 text-gray-300 rounded-full text-sm cursor-pointer hover:bg-gray-700 transition-colors">
            Talks
          </span>
        </div>

        {/* Talks Grid */}
        <ul
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 list-none p-0"
          aria-label="List of talks and workshops"
        >
          {talks.map((talk, idx) => (
            <li key={talk.slug}>
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: idx * 0.1 }}
                className="h-full"
              >
                <Link
                  href={`/talks/${talk.slug}`}
                  className={`group block h-full bg-gray-900 rounded-2xl overflow-hidden border border-gray-800 hover:border-gray-700 transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 ${
                    talk.upcoming ? "ring-2 ring-yellow-500/50" : ""
                  }`}
                  aria-label={`Read more about ${talk.title}`}
                >
                  <article>
                    <div className="aspect-video relative overflow-hidden">
                      {talk.upcoming && (
                        <div className="absolute top-2 right-2 z-10 bg-yellow-500 text-black px-2 py-1 rounded-full text-xs font-bold">
                          Upcoming
                        </div>
                      )}
                      <Image
                        src={talk.src}
                        alt={`Slide from ${talk.title}`}
                        fill
                        sizes="(max-width: 768px) 100vw, (max-width: 1024px) 50vw, 33vw"
                        className="object-cover group-hover:scale-105 transition-transform duration-300"
                      />
                    </div>
                    <div className="p-5">
                      <div className="flex items-center gap-2 mb-2">
                        <span
                          className={`px-2 py-0.5 text-xs rounded-full ${
                            talk.type === "Workshop"
                              ? "bg-green-900 text-green-300"
                              : "bg-purple-900 text-purple-300"
                          }`}
                        >
                          {talk.type}
                        </span>
                        <span className="text-gray-500 text-xs">{talk.date}</span>
                      </div>
                      <h2 className="text-lg font-bold mb-2 leading-tight group-hover:text-blue-300 transition-colors">
                        {talk.title}
                      </h2>
                      <p className="text-gray-500 text-xs mb-2">{talk.event}</p>
                      <p className="text-gray-400 text-sm line-clamp-2">
                        {talk.description}
                      </p>
                    </div>
                  </article>
                </Link>
              </motion.div>
            </li>
          ))}
        </ul>

        {/* CTA Section */}
        <div className="mt-16 bg-gray-900 rounded-2xl p-8 text-center border border-gray-800">
          <h2 className="text-2xl font-bold mb-4">Want me to speak at your event?</h2>
          <p className="text-gray-400 mb-6 max-w-lg mx-auto">
            I&apos;m always excited to share knowledge about data science, software engineering, and tech careers.
          </p>
          <Link
            href="mailto:semariquit@gmail.com?subject=Speaking Invitation"
            className="inline-flex items-center gap-2 bg-blue-600 hover:bg-blue-500 text-white px-6 py-3 rounded-lg font-medium transition-colors"
          >
            Get in Touch
          </Link>
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
