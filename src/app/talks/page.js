"use client";
import Image from "next/image";
import Link from "next/link";
import { motion } from "framer-motion";

const talks = [
  {
    title: 'UPLB DSG x UPRHS CodeIT Workshop Day 1 - What is Data Science?',
    src: '/talks/talk1.jpg',
    date: '2024',
    event: 'UPLB Data Science Guild x UPRHS CodeIT',
    type: 'Workshop',
    description: 'An introductory workshop on data science fundamentals for high school students.',
    slidesLink: null,
  },
  {
    title: 'UPLB DSG x UPRHS CodeIT Workshop Day 2 - Storytelling with Data',
    src: '/talks/talk2.jpg',
    date: '2024',
    event: 'UPLB Data Science Guild x UPRHS CodeIT',
    type: 'Workshop',
    description: 'Teaching students how to communicate insights effectively through data visualization and narrative.',
    slidesLink: null,
  },
  {
    title: 'NextStep Hacks 2025 - Winning by Talking',
    src: '/talks/talk3.jpg',
    date: '2025',
    event: 'NextStep Hacks 2025',
    type: 'Talk',
    description: 'A talk on presentation skills and how to pitch your hackathon projects effectively.',
    slidesLink: null,
  },
  {
    title: 'JPCS - QCU Logic Unlocked Day 1 - Machine Learning with Python',
    src: '/talks/talk4.jpg',
    date: '2025',
    event: 'JPCS QCU Logic Unlocked',
    type: 'Workshop',
    description: 'Hands-on workshop introducing machine learning concepts using Python and popular ML libraries.',
    slidesLink: null,
  },
  {
    title: "UPLB DSG Applicants' Workshop - Data Storytelling with Canva",
    src: '/talks/talk5.jpg',
    date: '2024',
    event: 'UPLB Data Science Guild',
    type: 'Workshop',
    description: 'Teaching applicants how to create compelling data visualizations using Canva.',
    slidesLink: null,
  },
  {
    title: 'Data Engineering Pilipinas AI Study Group - AI Use Cases That Actually Matter',
    src: '/talks/talk6.jpg',
    date: '2025',
    event: 'Data Engineering Pilipinas',
    type: 'Talk',
    description: 'Discussing practical AI applications that create real-world impact.',
    slidesLink: null,
  },
  {
    title: 'DLSU ECES - Agile Edge: Swift Project Workflows',
    src: '/talks/talk7.gif',
    date: '2025',
    event: 'DLSU ECES',
    type: 'Talk',
    description: 'Upcoming talk on agile methodologies and efficient project management.',
    slidesLink: null,
    upcoming: true,
  },
];

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
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6" role="list" aria-label="List of talks and workshops">
          {talks.map((talk, idx) => (
            <motion.article
              key={idx}
              className={`bg-gray-900 rounded-2xl overflow-hidden border border-gray-800 hover:border-gray-700 transition-colors ${talk.upcoming ? 'ring-2 ring-yellow-500/50' : ''}`}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: idx * 0.1 }}
              role="listitem"
            >
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
                  className="object-cover hover:scale-105 transition-transform duration-300"
                />
              </div>
              <div className="p-5">
                <div className="flex items-center gap-2 mb-2">
                  <span className={`px-2 py-0.5 text-xs rounded-full ${talk.type === 'Workshop' ? 'bg-green-900 text-green-300' : 'bg-purple-900 text-purple-300'}`}>
                    {talk.type}
                  </span>
                  <span className="text-gray-500 text-xs">{talk.date}</span>
                </div>
                <h2 className="text-lg font-bold mb-2 leading-tight">{talk.title}</h2>
                <p className="text-gray-500 text-xs mb-2">{talk.event}</p>
                <p className="text-gray-400 text-sm mb-4 line-clamp-2">
                  {talk.description}
                </p>
                {talk.slidesLink && (
                  <Link
                    href={talk.slidesLink}
                    className="inline-flex items-center gap-2 text-blue-400 hover:text-blue-300 text-sm"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    View Slides →
                  </Link>
                )}
              </div>
            </motion.article>
          ))}
        </div>

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
