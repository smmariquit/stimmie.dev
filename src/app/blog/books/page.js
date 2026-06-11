// src/app/blog/books/page.js

"use client";
import Link from "next/link";
import MosaicBackground from "@/components/MosaicBackground";

// Books data for 2025
const books = [
  {
    title: "Communist Manifesto",
    author: "Karl Marx and Friedrich Engels",
    // rating: 5,
    thoughts: "How oppression became more subtle and systemic over time, from early feudalism to modern capitalism.",
    category: "Non-Fiction",
  },
  {
    title: "The Design of Everyday Things",
    author: "Don Norman",
    // rating: 4,
    thoughts: "Every small feature of tools we use daily can impact our experience significantly. The design of the spaces we share impacts our behavior as a society.",
    category: "Non-Fiction",
  },
  {
    title: "The Alchemist",
    author: "Paulo Coelho",
    // rating: 5,
    thoughts: "Spiritual experiences and one's personal journey.",
    category: "Fiction",
  },
  {
    title: "The Subtle Art of Not Giving a F*ck",
    author: "Mark Manson",
    // rating: 5,
    thoughts: "You have only so much time and energy. Devote it to the things that matter.",
    category: "Fiction",
  },
  {
    title: "1984",
    author: "George Orwell",
    // rating: 5,
    thoughts: "Language can be a tool of control. ",
    category: "Fiction",
  },
  {
    title: "Animal Farm",
    author: "George Orwell",
    // rating: 5,
    thoughts: "Orwell's writing was really sharp on this one. This could be read by a teenager.",
    category: "Fiction",
  },
  {
    title: "Python One-Liners",
    author: "Christian Mayer",
    // rating: 5,
    thoughts: "Cool party tricks and some useful patterns for my workshops",
    category: "Non-Fiction",
  },
  {
    title: "AI: A Modern Approach",
    author: "Stuart Russell and Peter Norvig",
    // rating: 5,
    thoughts: "I only read this partially, but I found the definitions of AI and various techniques to be useful for my workshops : D",
    category: "Non-Fiction",
  },
  // Add more books as you read them!
];

export default function BooksPage() {
  return (
    <div className="min-h-screen bg-black w-full overflow-hidden">
      <MosaicBackground />

      {/* Blog post content overlay */}
      <div className="fixed inset-0 flex items-center justify-center pointer-events-none" style={{ zIndex: 50 }}>
        <div className="pointer-events-auto bg-gray-800/70 text-white p-6 rounded backdrop-blur-sm max-h-10/12 max-w-3xl w-full mx-4 overflow-y-auto no-scrollbar">
          <div className="flex flex-col">
            {/* Navigation */}
            <div className="flex items-center gap-4 mb-6">
              <Link href="/blog" className="text-white/70 hover:text-white transition-colors flex items-center gap-2">
                <span>←</span>
                <span>Blog</span>
              </Link>
              <span className="text-white/30">|</span>
              <Link href="/" className="text-white/70 hover:text-white transition-colors">
                Home
              </Link>
            </div>

            {/* Article header */}
            <article>
              <header className="mb-8">
                <h1 className="font-sans text-4xl sm:text-5xl font-black text-white mb-2">
                  Books I Read in 2025
                </h1>
                <p className="text-white/50">December 31, 2025</p>
              </header>

              {/* Introduction */}
              <section className="mb-8">
                <p className="text-white/90 leading-relaxed">
                  I got back to reading in 2025, in part thanks to Gen Z book club. Here are the books I read and some thoughts on each of them :DD
                </p>
              </section>

              {/* Stats */}
              <section className="bg-white/5 p-4 rounded mb-8">
                <div className="grid grid-cols-3 gap-4 text-center">
                  <div>
                    <p className="text-3xl font-bold text-white">{books.length}</p>
                    <p className="text-white/50 text-sm">Books Read</p>
                  </div>
                  <div>
                    <p className="text-3xl font-bold text-white">
                      {books.filter((b) => b.category === "Fiction").length}
                    </p>
                    <p className="text-white/50 text-sm">Fiction</p>
                  </div>
                  <div>
                    <p className="text-3xl font-bold text-white">
                      {books.filter((b) => b.category === "Non-Fiction").length}
                    </p>
                    <p className="text-white/50 text-sm">Non-Fiction</p>
                  </div>
                </div>
              </section>

              {/* Books list */}
              <section>
                <h2 className="text-2xl font-bold text-white mb-4">The Books</h2>
                <div className="flex flex-col gap-4">
                  {books.map((book, idx) => (
                    <div key={idx} className="bg-white/5 p-4 rounded">
                      <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-2">
                        <div>
                          <h3 className="text-lg font-semibold text-white">{book.title}</h3>
                          <p className="text-white/60 text-sm">by {book.author}</p>
                        </div>
                        <div className="flex items-center gap-3">
                          <span className="text-xs text-white/40 bg-white/10 px-2 py-1 rounded">
                            {book.category}
                          </span>
                        </div>
                      </div>
                      <p className="text-white/80 mt-3 text-sm leading-relaxed">{book.thoughts}</p>
                    </div>
                  ))}
                </div>
              </section>

              {/* Closing thoughts */}
              <section className="mt-8 pt-8 border-t border-white/10">
                <h2 className="text-2xl font-bold text-white mb-4">Looking Ahead</h2>
                <p className="text-white/90 leading-relaxed">
                  I'll try to read 20 books in 2026 and take advantage of the power of audiobooks.
                </p>
              </section>
            </article>
          </div>
        </div>
      </div>
    </div>
  );
}
