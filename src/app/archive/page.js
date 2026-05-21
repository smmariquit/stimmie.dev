"use client";
import Image from "next/image";
import Link from "next/link";
import { motion } from "framer-motion";
import MosaicBackground from "@/components/MosaicBackground";

// Version history - add new versions at the top!
const versions = [
  {
    version: "2.0",
    name: "Bento Box Era",
    date: "January 2026",
    current: true,
    description: "A complete redesign featuring a modern bento box grid layout with dark theme. Added GitHub contribution calendar, improved accessibility, subdomain redirects, and dedicated pages for projects and talks.",
    features: [
      "Bento box grid layout",
      "GitHub contribution calendar integration",
      "Full accessibility support (WCAG compliant)",
      "Subdomain redirect system",
      "Dedicated Projects & Talks pages",
      "OWASP security headers",
      "Responsive full-screen design",
    ],
    screenshot: "/archive/v2-screenshot.png",
    liveUrl: null, // Current version, no archive needed
    techStack: ["Next.js 14", "Tailwind CSS", "Framer Motion", "Vercel"],
  },
  {
    version: "1.0",
    name: "Original Portfolio",
    date: "2024 - January 2026",
    current: false,
    description: "The original portfolio design featuring an image grid background with a centered overlay containing all content in a single scrollable card. Simple but functional.",
    features: [
      "Image grid background sorted by hue",
      "Centered scrollable content overlay",
      "Talks and projects sections",
      "Social media links",
      "Minimizable overlay",
    ],
    screenshot: "/archive/v1-screenshot.png",
    liveUrl: "/archive/v1", // Link to archived version if available
    techStack: ["Next.js", "Tailwind CSS", "Framer Motion"],
  },
  {
    version: "0.2",
    name: "Carrd",
    date: "~2023",
    current: false,
    description: "A simple one-page site built using Carrd's drag-and-drop builder. Quick to set up and served as a simple landing page.",
    features: [
      "Single-page layout",
      "Quick setup with Carrd builder",
      "Social links",
    ],
    screenshot: null,
    liveUrl: "https://stimmie.carrd.co/",
    techStack: ["Carrd"],
  },
  {
    version: "0.1",
    name: "Linktree",
    date: "~2022",
    current: false,
    description: "Before building custom portfolios, I used Linktree as a simple link-in-bio solution to aggregate all my social profiles.",
    features: [
      "Link-in-bio aggregator",
      "All social profiles in one place",
    ],
    screenshot: null,
    liveUrl: "https://linktr.ee/stimmeh",
    techStack: ["Linktree"],
  },
  {
    version: "HS",
    name: "High School Publications",
    date: "2022-2023",
    current: false,
    description: "Before web development, I contributed to my high school's student publications at San Miguel Corporation - Lucena. These were my earliest forays into content creation and design.",
    features: [
      "Pahayagang Miguel 2023 - Student newspaper",
      "TMH HS Newspaper 2022-2023",
      "Early experience in content creation",
    ],
    screenshot: null,
    liveUrl: "https://www.smcl.edu.ph/student_life/publications/Pahayagang%20Miguel%202023.php",
    techStack: ["Print & Digital Publications"],
    extraLinks: [
      { label: "TMH HS Newspaper", url: "https://www.smcl.edu.ph/student_life/publications/TMH%20HS%20News%20paper%202022-2023.php" },
    ],
  },
];

export default function ArchivePage() {
  return (
    <div className="min-h-screen bg-black text-white">
      <MosaicBackground />

      {/* All content above the fixed mosaic */}
      <div className="relative" style={{ zIndex: 10 }}>
        {/* Header */}
        <header className="sticky top-0 z-50 bg-gray-950/80 backdrop-blur-md border-b border-gray-800">
          <div className="max-w-4xl mx-auto px-4 py-4 flex items-center justify-between">
            <Link href="/" className="text-xl font-bold hover:text-blue-400 transition-colors">
              ← Stimmie
            </Link>
            <nav className="flex gap-4" aria-label="Main navigation">
              <Link href="/projects" className="text-gray-400 hover:text-white transition-colors">
                Projects
              </Link>
              <Link href="/talks" className="text-gray-400 hover:text-white transition-colors">
                Talks
              </Link>
            </nav>
          </div>
        </header>

        {/* Main Content */}
        <main className="max-w-4xl mx-auto px-4 py-12" id="main-content">
          <div className="mb-12">
            <h1 className="text-4xl md:text-5xl font-black mb-4">📜 Past Iterations</h1>
            <p className="text-gray-400 text-lg max-w-2xl">
              A timeline of how this portfolio has evolved over time. Each version represents a milestone in my journey as a developer.
            </p>
          </div>

          {/* Timeline */}
          <div className="relative">
            {/* Timeline line */}
            <div className="absolute left-4 md:left-8 top-0 bottom-0 w-0.5 bg-gray-800" aria-hidden="true" />

            {/* Version Cards */}
            <div className="space-y-12">
              {versions.map((v, idx) => (
                <motion.article
                  key={v.version}
                  className="relative pl-12 md:pl-20"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: idx * 0.15 }}
                >
                  {/* Timeline dot */}
                  <div 
                    className={`absolute left-2 md:left-6 w-4 h-4 rounded-full border-4 ${
                      v.current 
                        ? 'bg-blue-500 border-blue-500 ring-4 ring-blue-500/30' 
                        : 'bg-gray-800 border-gray-600'
                    }`}
                    aria-hidden="true"
                  />

                  {/* Version Card */}
                  <div className={`bg-gray-900/90 backdrop-blur-sm rounded-2xl overflow-hidden border ${v.current ? 'border-blue-500/50' : 'border-gray-800'}`}>
                    {/* Screenshot */}
                    {v.screenshot && (
                      <div className="aspect-video relative bg-gray-800 overflow-hidden">
                        <div className="absolute inset-0 flex items-center justify-center text-gray-600">
                          <span className="text-sm">Screenshot: {v.screenshot}</span>
                        </div>
                        {/* Uncomment when you have screenshots */}
                        {/* <Image
                          src={v.screenshot}
                          alt={`Screenshot of version ${v.version}`}
                          fill
                          className="object-cover"
                        /> */}
                      </div>
                    )}

                    <div className="p-6">
                      {/* Header */}
                      <div className="flex flex-wrap items-center gap-3 mb-4">
                        <span className={`px-3 py-1 rounded-full text-sm font-bold ${
                          v.current 
                            ? 'bg-blue-600 text-white' 
                            : 'bg-gray-800 text-gray-300'
                        }`}>
                          v{v.version}
                        </span>
                        <h2 className="text-xl font-bold">{v.name}</h2>
                        {v.current && (
                          <span className="px-2 py-0.5 bg-green-900 text-green-300 text-xs rounded-full">
                            Current
                          </span>
                        )}
                      </div>

                      <p className="text-gray-500 text-sm mb-4">{v.date}</p>
                      <p className="text-gray-300 mb-6">{v.description}</p>

                      {/* Features */}
                      <div className="mb-6">
                        <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wide mb-3">
                          Key Features
                        </h3>
                        <ul className="grid grid-cols-1 md:grid-cols-2 gap-2">
                          {v.features.map((feature, fIdx) => (
                            <li key={fIdx} className="flex items-center gap-2 text-sm text-gray-300">
                              <span className="text-green-500">✓</span>
                              {feature}
                            </li>
                          ))}
                        </ul>
                      </div>

                      {/* Tech Stack */}
                      <div className="mb-6">
                        <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wide mb-3">
                          Tech Stack
                        </h3>
                        <div className="flex flex-wrap gap-2">
                          {v.techStack.map((tech, tIdx) => (
                            <span
                              key={tIdx}
                              className="px-2 py-1 bg-gray-800 text-gray-300 text-xs rounded-md"
                            >
                              {tech}
                            </span>
                          ))}
                        </div>
                      </div>

                      {/* Actions */}
                      <div className="flex flex-wrap gap-4">
                        {v.liveUrl && !v.current && (
                          <Link
                            href={v.liveUrl}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="inline-flex items-center gap-2 text-blue-400 hover:text-blue-300 text-sm"
                          >
                            View Archived Version →
                          </Link>
                        )}
                        {v.extraLinks && v.extraLinks.map((link, lIdx) => (
                          <Link
                            key={lIdx}
                            href={link.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="inline-flex items-center gap-2 text-blue-400 hover:text-blue-300 text-sm"
                          >
                            {link.label} →
                          </Link>
                        ))}
                      </div>
                    </div>
                  </div>
                </motion.article>
              ))}
            </div>
          </div>

          {/* Future Section */}
          <div className="mt-16 relative pl-12 md:pl-20">
            <div 
              className="absolute left-2 md:left-6 w-4 h-4 rounded-full border-4 border-dashed border-gray-600 bg-gray-950"
              aria-hidden="true"
            />
            <div className="bg-gray-900/50 backdrop-blur-sm border border-dashed border-gray-700 rounded-2xl p-6 text-center">
              <h2 className="text-xl font-bold text-gray-400 mb-2">What&apos;s Next?</h2>
              <p className="text-gray-500">
                Stay tuned for future iterations. This portfolio is always evolving!
              </p>
            </div>
          </div>
        </main>

        {/* Footer */}
        <footer className="border-t border-gray-800 mt-12">
          <div className="max-w-4xl mx-auto px-4 py-8 text-center text-gray-500 text-sm">
            <p>© {new Date().getFullYear()} Stimmie. All rights reserved.</p>
          </div>
        </footer>
      </div>
    </div>
  );
}
