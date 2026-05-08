"use client";
import Image from "next/image";
import { motion } from "framer-motion";
import { useState, useEffect } from 'react';
import imageData from "../../public/image_data_summary.json";
import Link from "next/link";
import { GitHubCalendar } from 'react-github-calendar';
import { talks } from "@/data/talks";
import { projects } from "@/data/projects";

// imageData is an object keyed by filename-stem; convert to an array and attach a public src.
const images = Object.values(imageData).map((it) => ({
  ...it,
  src: `/images/${it.file}`,
  thumbSrc: `/images_small/${it.file.replace(/\.[^.]+$/, ".webp")}`,
}));

// sort images by mean_hue (ascending) so neighbouring tiles flow through
// the colour wheel rather than clashing — gives the grid a gradient feel.
const imagesSortedByMeanRGB = [...images].sort((a, b) => a.mean_hue - b.mean_hue);

// Discrete rotation buckets. Using a fixed palette of angles (rather than
// random) keeps SSR/CSR markup stable and prevents hydration mismatches.
const TILE_ROTATIONS = [-2.4, 1.6, -0.8, 2.1, -1.5, 0.7, -2.0, 1.2];

const blogPosts = [
  {
    slug: "books",
    title: "Books I Read in 2025",
    date: "December 31, 2025",
    excerpt: "A reflection on all the books I devoured this year.",
  },
];

const services = [
  { title: 'Software Development', description: 'Custom web apps, APIs, and automation solutions for your business.' },
  { title: 'Data Science & Analytics', description: 'Turn your data into actionable insights with modern ML/AI tools.' },
  { title: 'Workshops & Training', description: 'Hands-on sessions on data science, Python, and tech fundamentals.' },
  { title: 'Consulting', description: 'Technical guidance for startups and organizations.' },
];

function Thumb({ image }) {
  const [src, setSrc] = useState(image.thumbSrc);
  const [triedFallback, setTriedFallback] = useState(false);

  const handleError = () => {
    if (!triedFallback && image.src) {
      setTriedFallback(true);
      setSrc(image.src);
    }
  };

  // image.mean_rgb is a 0..1 float triple from the image-summary pipeline.
  // The previous version rounded these directly (0..1 → 0 or 1), so the
  // load-state placeholder was always near-black instead of the actual
  // dominant colour. Multiply by 255 to get a real swatch.
  const r = image.mean_rgb ? Math.round(image.mean_rgb[0] * 255) : 17;
  const g = image.mean_rgb ? Math.round(image.mean_rgb[1] * 255) : 17;
  const b = image.mean_rgb ? Math.round(image.mean_rgb[2] * 255) : 17;
  const bg = `rgb(${r}, ${g}, ${b})`;

  return (
    <div
      className="relative w-full h-full overflow-hidden rounded-md"
      style={{
        background: bg,
        boxShadow:
          "0 2px 8px rgba(0,0,0,0.45), inset 0 0 0 1px rgba(255,255,255,0.04)",
      }}
    >
      <Image
        src={src}
        alt=""
        onError={handleError}
        width={160}
        height={160}
        className="absolute inset-0 w-full h-full object-cover"
        // Warm sepia-ish grade + punchy contrast — no blur. The detail-
        // obscuring is delegated to the noise overlay below so tiles
        // read as "stylised", not "redacted".
        style={{
          filter:
            "sepia(0.35) saturate(1.4) brightness(0.5) contrast(1.18)",
        }}
        loading="lazy"
      />
      {/* Per-tile fractal-noise grain. With mix-blend overlay it interleaves
          with the image's own luminance, so faces/details dissolve into
          texture instead of being smeared. The SVG data URL is shared
          across all tiles so the browser decodes it once. */}
      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-0 mix-blend-overlay"
        style={{
          backgroundImage:
            "url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='1.8' numOctaves='2' stitchTiles='stitch'/></filter><rect width='100%' height='100%' filter='url(%23n)'/></svg>\")",
          backgroundSize: "80px 80px",
          opacity: 0.55,
        }}
      />
      {/* Per-tile vignette tinted by the image's own dominant colour: keeps
          tiles from feeling disjoint and gives a gentle painterly fall-off. */}
      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-0"
        style={{
          background: `radial-gradient(ellipse at 50% 35%, transparent 0%, rgba(0,0,0,0.45) 78%), linear-gradient(180deg, rgba(${r},${g},${b},0) 30%, rgba(${r},${g},${b},0.22) 100%)`,
        }}
      />
    </div>
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
      
      {/* Background Image Grid: dense scrapbook-style mosaic.
          The list is tripled so the smaller tiles still cover tall
          viewports; mixed spans + dense flow create organic rhythm
          rather than a uniform tile sheet. Each tile is heavily blurred
          + sepia-tinted so individual faces aren't recognisable. */}
      <motion.div
        className="bg-tile-grid relative grid w-full mx-auto"
        style={{
          zIndex: 0,
          gridTemplateColumns: "repeat(auto-fill, minmax(48px, 1fr))",
          gridAutoRows: "48px",
          gridAutoFlow: "dense",
          gap: "0.35rem",
          padding: "0.35rem",
        }}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 1.1, ease: "easeOut" }}
        aria-hidden="true"
      >
        {[
          ...imagesSortedByMeanRGB,
          ...imagesSortedByMeanRGB,
          ...imagesSortedByMeanRGB,
        ].map((image, i) => (
          <div
            key={`tile-${i}-${image.src}`}
            className="bg-tile"
            style={{
              "--rot": `${TILE_ROTATIONS[i % TILE_ROTATIONS.length]}deg`,
            }}
          >
            <Thumb image={image} />
          </div>
        ))}
      </motion.div>

      {/* Atmospheric vignette: darkens edges, focuses the eye on the
          centred bento overlay without competing with it visually. */}
      <div
        aria-hidden="true"
        className="pointer-events-none fixed inset-0"
        style={{
          zIndex: 1,
          background:
            "radial-gradient(ellipse at center, transparent 25%, rgba(0,0,0,0.55) 75%, rgba(0,0,0,0.85) 100%)",
        }}
      />

      {/* Whole-viewport film grain at low opacity — analog warmth across
          the bento gutters and bento card borders. Each tile already
          carries its own (denser) noise layer; this is just glue. */}
      <div
        aria-hidden="true"
        className="pointer-events-none fixed inset-0 opacity-[0.045] mix-blend-overlay"
        style={{
          zIndex: 2,
          backgroundImage:
            "url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 220 220'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2' stitchTiles='stitch'/><feColorMatrix values='0 0 0 0 1  0 0 0 0 1  0 0 0 0 1  0 0 0 0.6 0'/></filter><rect width='100%' height='100%' filter='url(%23n)'/></svg>\")",
          backgroundSize: "220px 220px",
        }}
      />

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
                <nav className="flex flex-row flex-wrap gap-2 md:gap-3 mt-3" aria-label="Social media links">
                  <Link href="https://www.linkedin.com/in/stimmie" title="LinkedIn" aria-label="LinkedIn profile">
                    <Image src="/logos/linkedin.png" alt="LinkedIn - Professional networking profile" width={24} height={24} className="rounded hover:scale-110 transition-transform w-5 h-5 md:w-6 md:h-6" />
                  </Link>
                  <Link href="https://www.instagram.com/friedicecrm" title="Instagram">
                    <Image src="/logos/instagram.png" alt="Instagram" width={24} height={24} className="rounded hover:scale-110 transition-transform w-5 h-5 md:w-6 md:h-6" />
                  </Link>
                  <Link href="https://www.github.com/smmariquit" title="GitHub">
                    <Image src="/logos/github.png" alt="GitHub" width={24} height={24} className="rounded hover:scale-110 transition-transform w-5 h-5 md:w-6 md:h-6" />
                  </Link>
                  <Link href="https://medium.com/@semariquit" title="Medium">
                    <Image src="/logos/medium.png" alt="Medium" width={24} height={24} className="rounded hover:scale-110 transition-transform w-5 h-5 md:w-6 md:h-6" />
                  </Link>
                  <Link href="mailto:semariquit@gmail.com" title="Email">
                    <Image src="/logos/email.png" alt="Email" width={24} height={24} className="rounded hover:scale-110 transition-transform w-5 h-5 md:w-6 md:h-6" />
                  </Link>
                  <Link href="https://myanimelist.net/" title="MyAnimeList">
                    <Image src="/logos/myanimelist.png" alt="MyAnimeList" width={24} height={24} className="rounded hover:scale-110 transition-transform w-5 h-5 md:w-6 md:h-6" />
                  </Link>
                  <Link href="https://pizza-and-friends.webflow.io" title="Pizza and Friends">
                    <Image src="/logos/pizza.png" alt="Pizza and Friends" width={24} height={24} className="rounded hover:scale-110 transition-transform w-5 h-5 md:w-6 md:h-6" />
                  </Link>
                  <Link href="http://letterboxd.com/stimmieuwu" title="Letterboxd">
                    <Image src="/logos/letterboxd.png" alt="Letterboxd" width={24} height={24} className="rounded hover:scale-110 transition-transform w-5 h-5 md:w-6 md:h-6" />
                  </Link>
                  <Link href="https://www.imdb.com/name/nm12149035/" title="IMDB" aria-label="IMDB profile">
                    <Image src="/logos/imdb.png" alt="IMDB" width={24} height={24} className="rounded hover:scale-110 transition-transform w-5 h-5 md:w-6 md:h-6" />
                  </Link>
                  <Link href="https://goodreads.com/stimmie" title="Goodreads">
                    <Image src="/logos/goodreads.png" alt="Goodreads" width={24} height={24} className="rounded hover:scale-110 transition-transform w-5 h-5 md:w-6 md:h-6" />
                  </Link>
                  <Link href="https://docs.google.com/document/d/1cJya3Zb2ck9vkxIKc1LQjJomQS_LFtBOABlzHrb7Z5s/edit?usp=sharing" title="Hackathon Guide">
                    <Image src="/logos/hackathon.png" alt="Hackathon Guide" width={24} height={24} className="rounded hover:scale-110 transition-transform w-5 h-5 md:w-6 md:h-6" />
                  </Link>
                  <Link href="https://docs.google.com/document/d/1cJya3Zb2ck9vkxIKc1LQjJomQS_LFtBOABlzHrb7Z5s/" title="Freshie Resources" aria-label="Freshie Resources Document">
                    <Image src="/logos/freshie.png" alt="Freshie Resources - A guide for incoming students" width={24} height={24} className="rounded hover:scale-110 transition-transform w-5 h-5 md:w-6 md:h-6" />
                  </Link>
                  <Link href="https://osu.ppy.sh/users/14900686" title="osu! Profile" aria-label="osu! rhythm game profile">
                    <Image src="/logos/osu.png" alt="osu! - Rhythm game profile" width={24} height={24} className="rounded hover:scale-110 transition-transform w-5 h-5 md:w-6 md:h-6" />
                  </Link>
                  <Link href="https://open.kattis.com/users/simonee" title="Kattis" aria-label="Kattis competitive programming profile">
                    <Image src="/logos/kattis.png" alt="Kattis" width={24} height={24} className="rounded hover:scale-110 transition-transform w-5 h-5 md:w-6 md:h-6" />
                  </Link>
                  <Link href="https://open.spotify.com/user/opzo90f4votlfqmg9rl94qrra?si=538e1f5748424274" title="Spotify">
                    <Image src="/logos/spotify.png" alt="Spotify" width={24} height={24} className="rounded hover:scale-110 transition-transform w-5 h-5 md:w-6 md:h-6" />
                  </Link>
                  <Link href="https://www.last.fm/user/mistakenpog" title="Last.fm">
                    <Image src="/logos/lastfm.png" alt="Last.fm" width={24} height={24} className="rounded hover:scale-110 transition-transform w-5 h-5 md:w-6 md:h-6" />
                  </Link>
                  <Link href="https://leetcode.com/u/stimmers/" title="LeetCode">
                    <Image src="/logos/leetcode.png" alt="LeetCode" width={24} height={24} className="rounded hover:scale-110 transition-transform w-5 h-5 md:w-6 md:h-6" />
                  </Link>
                  <Link href="https://www.strava.com/athletes/129023200" title="Strava" aria-label="Strava fitness tracking profile">
                    <Image src="/logos/strava.png" alt="Strava - Fitness and running activities" width={24} height={24} className="rounded hover:scale-110 transition-transform w-5 h-5 md:w-6 md:h-6" />
                  </Link>
                  <Link href="https://www.kaggle.com/stimmie" title="Kaggle" aria-label="Kaggle data science profile">
                    <Image src="/logos/kaggle.png" alt="Kaggle - Data science competitions and notebooks" width={24} height={24} className="rounded hover:scale-110 transition-transform w-5 h-5 md:w-6 md:h-6" />
                  </Link>
                  <Link href="https://www.facebook.com/stimmieuwu/" title="Facebook" aria-label="Facebook profile">
                    <Image src="/logos/facebook.png" alt="Facebook" width={24} height={24} className="rounded hover:scale-110 transition-transform w-5 h-5 md:w-6 md:h-6" />
                  </Link>
                  <span title="Discord: @pataponz" aria-label="Discord username: @pataponz" className="cursor-default flex items-center">
                    <Image src="/logos/discord.png" alt="Discord - @pataponz" width={24} height={24} className="rounded hover:scale-110 transition-transform w-5 h-5 md:w-6 md:h-6" />
                  </span>
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
                <Link href="/blog" className="block h-full flex flex-col" aria-label="View all blog posts">
                  <h2 className="font-bold text-base md:text-lg mb-2 text-white">📝 Blog</h2>
                  <div className="flex flex-col gap-2 flex-1 overflow-hidden" role="list">
                    {blogPosts.slice(0, 3).map((post, idx) => (
                      <article key={idx} className="bg-gray-800/60 rounded-lg p-2 md:p-3" role="listitem">
                        <p className="text-xs md:text-sm font-semibold text-white line-clamp-2">{post.title}</p>
                        <time className="text-[10px] text-white/50" dateTime={post.date}>{post.date}</time>
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

            </div>
          </div>
        )}
      </div>
    </div>
  );
}
