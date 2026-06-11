// src/app/HomeClient.js

"use client";
import Image from "next/image";
import { motion } from "framer-motion";
import { useState } from 'react';
import Link from "next/link";
import { GitHubCalendar } from 'react-github-calendar';
import { talks } from "@/data/talks";
import { projects } from "@/data/projects";
import MosaicBackground from "@/components/MosaicBackground";
import { socialCategories } from "@/data/socials";

import { blogPosts } from "@/data/blogs";

const services = [
  { title: 'Software Development', description: 'Custom web apps, APIs, and automation solutions for your business.' },
  { title: 'Data Science & Analytics', description: 'Turn your data into actionable insights with modern ML/AI tools.' },
  { title: 'Workshops & Training', description: 'Hands-on sessions on data science, Python, and tech fundamentals.' },
  { title: 'Consulting', description: 'Technical guidance for startups and organizations.' },
];


// Renders a single social-link entry from the socialCategories data.
// Wraps the icon in a Link when there is an `href`, and falls back to a
// non-clickable badge for display-only entries (e.g. Discord usernames).
function SocialIcon({ link }) {
  const altText = link.alt || link.name;
  const img = (
    <Image
      src={link.icon}
      alt={altText}
      width={24}
      height={24}
      className={`rounded hover:scale-110 transition-transform w-5 h-5 md:w-6 md:h-6 ${link.name === 'GitHub' || link.name === 'Kattis' ? 'invert' : ''}`}
    />
  );

  if (!link.href) {
    return (
      <span
        title={link.name}
        aria-label={altText}
        className="cursor-default flex items-center"
      >
        {img}
      </span>
    );
  }

  return (
    <Link
      href={link.href}
      title={link.name}
      aria-label={altText}
      className="flex items-center focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 rounded"
    >
      {img}
    </Link>
  );
}

export default function HomeClient({ mediaData }) {
  const [overlayMinimized, setOverlayMinimized] = useState(false);
  const { film, book, anime } = mediaData || {};

  return (
    <div className="absolute inset-0 h-screen bg-black w-full overflow-hidden" lang="en">
      {/* Skip to main content - Accessibility */}
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 focus:z-[100] focus:bg-white focus:text-black focus:px-4 focus:py-2 focus:rounded"
      >
        Skip to main content
      </a>

      <MosaicBackground />

      {/* Bento Box Overlay */}
      <div className="fixed inset-0 flex items-center justify-center pointer-events-none" style={{ zIndex: 50 }} role="main" aria-label="Main content">
        {/* Minimize/Expand Button */}
        <button
          onClick={() => setOverlayMinimized((v) => !v)}
          className="pointer-events-auto absolute top-4 right-4 bg-gray-700/80 text-white p-2 rounded-md hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
          aria-label={overlayMinimized ? 'Expand portfolio overlay' : 'Minimize portfolio overlay'}
          aria-expanded={!overlayMinimized}
        >
          {overlayMinimized ? '+' : '-'}
        </button>

        {!overlayMinimized && (
          <div
            id="main-content"
            // On mobile: fill the screen but allow content to grow and scroll naturally.
            // On md+: switch back to a fixed bento "card" with its own scroll container.
            className="pointer-events-auto w-full h-[100dvh] overflow-y-auto custom-scrollbar p-3 md:h-[95vh] md:w-[95vw] md:p-6 lg:p-8"
            tabIndex={-1}
          >

            {/* Bento Grid:
                - mobile: single column, content sized to its own height (no auto-rows-fr stretch)
                - md+:    multi-column bento, each row stretched to fill viewport
            */}
            <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-4 gap-3 md:gap-4 lg:gap-5 md:h-full md:auto-rows-fr">

              {/* Section 1: Hero - Stimmie, Software Engineer */}
              <motion.div
                className="col-span-1 row-span-1 bg-gray-900/90 backdrop-blur-sm rounded-2xl p-4 md:p-5 lg:p-6 flex flex-col justify-between border border-gray-800"
                whileHover={{ scale: 1.02 }}
                transition={{ type: "spring", stiffness: 300, damping: 20 }}
                role="region"
                aria-label="About Stimmie"
              >
                <header>
                  <h1 className="font-sans text-2xl md:text-3xl lg:text-4xl xl:text-5xl font-black text-white">Stimmie</h1>
                  <div className="flex items-center gap-2 mt-1">
                    <p className="text-white/60 text-xs md:text-sm">Software Engineer</p>
                    <Link href="/archive" className="text-[10px] text-blue-400 hover:text-blue-300" title="View past iterations">
                      v2.0
                    </Link>
                  </div>
                </header>
                <nav
                  className="flex-1 min-h-0 overflow-y-auto custom-scrollbar pr-2 mt-3"
                  aria-label="Profiles and social links"
                >
                  <div className="flex flex-col gap-4">
                    {socialCategories.map((cat) => (
                      <div key={cat.label} className="shrink-0">
                        <h3 className="text-[8px] uppercase tracking-[0.18em] text-white/35 mb-1.5 font-medium">
                          {cat.label}
                        </h3>
                        <ul className="flex flex-row flex-wrap gap-2 list-none p-0 m-0">
                          {cat.links.map((link) => (
                            <li key={link.name}>
                              <SocialIcon link={link} />
                            </li>
                          ))}
                        </ul>
                      </div>
                    ))}
                  </div>
                </nav>
              </motion.div>

              {/* Section 2: Talks & Workshops */}
              <motion.section
                className="col-span-1 md:col-span-2 lg:col-span-2 row-span-1 bg-gray-900/90 backdrop-blur-sm rounded-2xl p-4 border border-gray-800 flex flex-col"
                whileHover={{ scale: 1.01 }}
                transition={{ type: "spring", stiffness: 300, damping: 20 }}
                aria-labelledby="talks-heading"
              >
                <div className="flex items-center justify-between mb-2">
                  <h2 id="talks-heading" className="font-bold text-base md:text-lg text-white">
                    🎤 Talks & Workshops
                  </h2>
                  <Link
                    href="/talks"
                    className="text-[11px] md:text-xs text-blue-400 hover:text-blue-300 whitespace-nowrap"
                    aria-label="View all talks"
                  >
                    View all →
                  </Link>
                </div>
                <ul
                  className="grid grid-cols-2 md:grid-cols-4 gap-2 list-none p-0 md:flex-1 md:overflow-hidden md:h-[calc(100%-2.5rem)]"
                  aria-label="List of talks and workshops"
                >
                  {talks.map((t) => (
                    <li key={t.slug}>
                      <Link
                        href={`/talks/${t.slug}`}
                        className="group relative rounded-lg overflow-hidden block aspect-video focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
                        aria-label={t.title}
                      >
                        <Image
                          src={t.src}
                          alt={`Talk: ${t.title}`}
                          width={320}
                          height={180}
                          className="object-cover w-full h-full group-hover:scale-105 transition-transform"
                        />
                        <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-end p-1">
                          <p className="text-[9px] md:text-[10px] text-white leading-tight line-clamp-2">
                            {t.title}
                          </p>
                        </div>
                      </Link>
                    </li>
                  ))}
                </ul>
              </motion.section>

              {/* Section 4: Blogs */}
              <motion.div
                className="col-span-1 row-span-1 bg-gray-900/90 backdrop-blur-sm rounded-2xl p-4 border border-gray-800"
                whileHover={{ scale: 1.02 }}
                transition={{ type: "spring", stiffness: 300, damping: 20 }}
                role="region"
                aria-label="Blog posts"
              >
                <Link href="/blog" className="group block h-full flex flex-col" aria-label="View all blog posts">
                  <div className="flex items-center justify-between mb-2">
                    <h2 className="font-bold text-base md:text-lg text-white group-hover:text-blue-400 transition-colors">📝 Blog</h2>
                    <span className="text-[11px] md:text-xs text-blue-400 opacity-0 group-hover:opacity-100 transition-opacity" aria-hidden="true">
                      Read more →
                    </span>
                  </div>
                  <div className="flex flex-col gap-2 flex-1 overflow-hidden" role="list">
                    {blogPosts.slice(0, 3).map((post, idx) => (
                      <article key={idx} className="bg-gray-800/60 rounded-lg p-2 md:p-3 group-hover:bg-gray-800/80 transition-colors" role="listitem">
                        <p className="text-xs md:text-sm font-semibold text-white line-clamp-2">{post.title}</p>
                        <time className="text-[10px] text-gray-400" dateTime={post.date}>{post.date}</time>
                      </article>
                    ))}
                  </div>
                </Link>
              </motion.div>

              {/* Section 3: Projects */}
              <motion.section
                className="col-span-1 md:col-span-2 lg:col-span-2 row-span-1 bg-gray-900/90 backdrop-blur-sm rounded-2xl p-4 border border-gray-800 flex flex-col"
                whileHover={{ scale: 1.01 }}
                transition={{ type: "spring", stiffness: 300, damping: 20 }}
                aria-labelledby="projects-heading"
              >
                <div className="flex items-center justify-between mb-2">
                  <h2 id="projects-heading" className="font-bold text-base md:text-lg text-white">
                    🚀 Projects
                  </h2>
                  <Link
                    href="/projects"
                    className="text-[11px] md:text-xs text-blue-400 hover:text-blue-300 whitespace-nowrap"
                    aria-label="View all projects"
                  >
                    View all →
                  </Link>
                </div>
                <ul
                  className="grid grid-cols-2 md:grid-cols-5 gap-2 list-none p-0 md:flex-1 md:overflow-hidden md:h-[calc(100%-2.5rem)]"
                  aria-label="List of projects"
                >
                  {projects.map((p) => (
                    <li key={p.slug}>
                      <Link
                        href={`/projects/${p.slug}`}
                        className="group relative rounded-lg overflow-hidden block aspect-video focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
                        aria-label={p.title}
                      >
                        <Image
                          src={p.src}
                          alt={`Project: ${p.title}`}
                          width={320}
                          height={180}
                          className="object-cover w-full h-full group-hover:scale-105 transition-transform"
                        />
                        <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity flex flex-col justify-end p-2">
                          <p className="text-[9px] md:text-xs text-white font-semibold line-clamp-2">
                            {p.title}
                          </p>
                        </div>
                      </Link>
                    </li>
                  ))}
                </ul>
              </motion.section>

              {/* Section 5: Services */}
              <motion.div
                className="col-span-1 row-span-1 bg-gray-900/90 backdrop-blur-sm rounded-2xl p-4 overflow-hidden border border-gray-800"
                whileHover={{ scale: 1.01 }}
                transition={{ type: "spring", stiffness: 300, damping: 20 }}
              >
                <h2 className="font-bold text-base md:text-lg mb-2 text-white">💼 Services</h2>
                <div className="grid grid-cols-1 gap-2 md:overflow-y-auto custom-scrollbar md:h-[calc(100%-2.5rem)]">
                  {services.map((s, idx) => (
                    <div key={idx} className="bg-gray-800/60 rounded-lg p-2 md:p-3">
                      <h3 className="text-xs md:text-sm font-semibold text-white">{s.title}</h3>
                      <p className="text-[9px] md:text-[10px] text-white/60 line-clamp-2">{s.description}</p>
                    </div>
                  ))}
                </div>
              </motion.div>

              {/* Section 6: GitHub Activity */}
              <motion.div
                className="col-span-1 row-span-1 bg-gray-900/90 backdrop-blur-sm rounded-2xl p-4 overflow-hidden border border-gray-800"
                whileHover={{ scale: 1.02 }}
                transition={{ type: "spring", stiffness: 300, damping: 20 }}
                role="region"
                aria-label="GitHub Activity"
              >
                <h2 className="font-bold text-base md:text-lg mb-2 text-white">🐙 GitHub</h2>
                <div className="md:overflow-hidden md:h-[calc(100%-2.5rem)] flex flex-col justify-between gap-2">
                  <div className="overflow-x-auto" aria-label="GitHub contribution calendar">
                    <GitHubCalendar
                      username="smmariquit"
                      colorScheme="dark"
                      blockSize={12}
                      blockMargin={4}
                      fontSize={14}
                    />
                  </div>
                  <Link
                    href="https://github.com/smmariquit"
                    className="text-[10px] text-blue-400 hover:text-blue-300 mt-auto"
                    aria-label="View full GitHub profile"
                  >
                    View full profile →
                  </Link>
                </div>
              </motion.div>

              {/* Section 7: Now Watching / Reading */}
              <motion.div
                className="col-span-1 row-span-1 bg-gray-900/90 backdrop-blur-sm rounded-2xl p-4 overflow-hidden border border-gray-800"
                whileHover={{ scale: 1.02 }}
                transition={{ type: "spring", stiffness: 300, damping: 20 }}
                role="region"
                aria-label="Currently watching and reading"
              >
                <h2 className="font-bold text-base md:text-lg mb-2 text-white">🎬📚 Now</h2>
                <div className="flex flex-col gap-3 md:h-[calc(100%-2.5rem)] md:overflow-y-auto custom-scrollbar">
                  {/* Latest Film from Letterboxd */}
                  {film && (
                    <Link href={film.link || 'https://letterboxd.com/stimmieuwu'} target="_blank" rel="noopener noreferrer" className="group">
                      <div className="flex gap-2 bg-gray-800/60 rounded-lg p-2 hover:bg-gray-800/80 transition-colors">
                        {film.posterUrl && (
                          <img
                            src={film.posterUrl}
                            alt={film.title}
                            className="w-10 h-14 object-cover rounded flex-shrink-0"
                          />
                        )}
                        <div className="flex flex-col justify-center min-w-0">
                          <p className="text-[10px] text-green-400 uppercase tracking-wide">Watched</p>
                          <p className="text-xs font-semibold text-white truncate group-hover:text-white/90">{film.title}</p>
                          {film.year && <p className="text-[10px] text-white/50">{film.year}</p>}
                          {film.rating && (
                            <p className="text-[10px] text-yellow-400">{'★'.repeat(Math.round(film.rating))}{'☆'.repeat(5 - Math.round(film.rating))}</p>
                          )}
                        </div>
                      </div>
                    </Link>
                  )}

                  {/* Latest Anime from MyAnimeList */}
                  {anime && (
                    <Link href={anime.link || 'https://myanimelist.net/profile/amorgosposter'} target="_blank" rel="noopener noreferrer" className="group">
                      <div className="flex gap-2 bg-gray-800/60 rounded-lg p-2 hover:bg-gray-800/80 transition-colors">
                        {anime.coverUrl && (
                          <img
                            src={anime.coverUrl}
                            alt={anime.title}
                            className="w-10 h-14 object-cover rounded flex-shrink-0"
                            onError={(e) => { e.target.style.display = 'none'; }}
                          />
                        )}
                        <div className="flex flex-col justify-center min-w-0">
                          <p className="text-[10px] text-purple-400 uppercase tracking-wide">{anime.status || 'Watching'}</p>
                          <p className="text-xs font-semibold text-white truncate group-hover:text-white/90">{anime.title}</p>
                          {anime.progress !== null && anime.total !== null && (
                            <p className="text-[10px] text-white/50">{anime.progress}/{anime.total} episodes</p>
                          )}
                          {anime.type && <p className="text-[10px] text-white/40">{anime.type}</p>}
                        </div>
                      </div>
                    </Link>
                  )}

                  {/* Latest Book from Goodreads */}
                  {book && (
                    <Link href={book.link || 'https://goodreads.com/stimmie'} target="_blank" rel="noopener noreferrer" className="group">
                      <div className="flex gap-2 bg-gray-800/60 rounded-lg p-2 hover:bg-gray-800/80 transition-colors">
                        {book.coverUrl && (
                          <img
                            src={book.coverUrl}
                            alt={book.title}
                            className="w-10 h-14 object-cover rounded flex-shrink-0"
                          />
                        )}
                        <div className="flex flex-col justify-center min-w-0">
                          <p className="text-[10px] text-blue-400 uppercase tracking-wide">
                            {book.shelf === 'currently-reading' ? 'Reading' : 'Read'}
                          </p>
                          <p className="text-xs font-semibold text-white truncate group-hover:text-white/90">{book.title}</p>
                          {book.author && <p className="text-[10px] text-white/50 truncate">{book.author}</p>}
                        </div>
                      </div>
                    </Link>
                  )}

                  {!film && !book && !anime && (
                    <p className="text-xs text-white/50">Loading media...</p>
                  )}
                </div>
              </motion.div>

              {/* Section 8: The Stimmieverse */}
              <motion.div
                className="col-span-1 md:col-span-2 lg:col-span-3 row-span-1 bg-gray-900/90 backdrop-blur-sm rounded-2xl p-4 overflow-hidden border border-gray-800"
                whileHover={{ scale: 1.01 }}
                transition={{ type: "spring", stiffness: 300, damping: 20 }}
                role="region"
                aria-label="Stimmieverse Subdomains"
              >
                <div className="flex items-center justify-between mb-3">
                  <h2 className="font-bold text-lg md:text-xl text-white">
                    🌐 The Stimmieverse
                  </h2>
                  <span className="text-xs text-gray-400">My other deployments</span>
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3 md:overflow-y-auto custom-scrollbar md:h-[calc(100%-3rem)]">
                  {[
                    { name: 'Kape', url: 'https://kape.stimmie.dev', desc: 'Support me / Buy me a coffee', icon: '☕' },
                    { name: 'Room TBA', url: 'https://room-tba.stimmie.dev', desc: 'UPLB 3D Campus Viewer', icon: '🗺️' },
                    { name: 'GradeSim', url: 'https://gradesim.stimmie.dev', desc: 'Course Planner & Simulator', icon: '🎓' },
                    { name: 'HearthCraft', url: 'https://hearthcraft.stimmie.dev', desc: 'Minecraft Server Museum', icon: '⛏️' },
                    { name: 'Atlas', url: 'https://atlas-of-my-skies.stimmie.dev', desc: 'Photography Portfolio', icon: '🌌' },
                    { name: 'Data', url: 'https://data.stimmie.dev', desc: 'Data Science & ML', icon: '📊' },
                    { name: 'The Crib', url: 'https://crib.stimmie.dev', desc: 'Personal Sandbox', icon: '🏠' },
                    { name: 'Workshops', url: 'https://workshops.stimmie.dev', desc: 'Slide Decks & Materials', icon: '🎤' },
                    { name: 'Minecraft', url: 'mc.stimmie.dev', href: 'https://crib.stimmie.dev', desc: 'Java/Bedrock Server', icon: '🎮' },
                    { name: 'Server Map', url: 'https://map.stimmie.dev', desc: 'The Crib Live Map', icon: '📍' },
                    { name: 'Web Dev', url: 'https://web.stimmie.dev', desc: 'Web Development Services', icon: '💻' },
                    { name: 'Tutoring', url: 'https://tutor.stimmie.dev', desc: 'Academic Tutoring', icon: '📚' },
                    { name: 'Repairs', url: 'https://repairs.stimmie.dev', desc: 'Tech Repair Services', icon: '🔧' },
                    { name: 'Links', url: 'https://links.stimmie.dev', desc: 'Quick Redirects', icon: '🔗' },
                  ].map((site) => (
                    <Link key={site.name} href={site.href || site.url} target="_blank" rel="noopener noreferrer" className="group block">
                      <div className="bg-gray-800/80 rounded-xl p-3 h-full hover:bg-gray-700 transition-colors border border-transparent hover:border-gray-500 shadow-sm">
                        <div className="flex items-center gap-2 mb-2">
                          <span className="text-base md:text-lg">{site.icon}</span>
                          <h3 className="text-sm md:text-base font-bold text-white group-hover:text-blue-300 transition-colors truncate">{site.name}</h3>
                        </div>
                        <p className="text-xs text-gray-300 line-clamp-2 leading-relaxed">{site.desc}</p>
                        <p className="text-xs font-medium text-blue-300 mt-2 truncate">{site.url.replace('https://', '')}</p>
                      </div>
                    </Link>
                  ))}
                </div>
              </motion.div>

            </div>
          </div>
        )}
      </div>
    </div>
  );
}
