// src/app/blog/books/page.js

import Link from "next/link";
import PageShell from "@/components/neo/PageShell";

export const metadata = {
  title: "Books I Read in 2025",
  description: "The books Stimmie read in 2025, with thoughts on each.",
};

const books = [
  {
    title: "Communist Manifesto",
    author: "Karl Marx and Friedrich Engels",
    thoughts:
      "How oppression became more subtle and systemic over time, from early feudalism to modern capitalism.",
    category: "Non-Fiction",
  },
  {
    title: "The Design of Everyday Things",
    author: "Don Norman",
    thoughts:
      "Every small feature of tools we use daily can impact our experience significantly. The design of the spaces we share impacts our behavior as a society.",
    category: "Non-Fiction",
  },
  {
    title: "The Alchemist",
    author: "Paulo Coelho",
    thoughts: "Spiritual experiences and one's personal journey.",
    category: "Fiction",
  },
  {
    title: "The Subtle Art of Not Giving a F*ck",
    author: "Mark Manson",
    thoughts:
      "You have only so much time and energy. Devote it to the things that matter.",
    category: "Fiction",
  },
  {
    title: "1984",
    author: "George Orwell",
    thoughts: "Language can be a tool of control.",
    category: "Fiction",
  },
  {
    title: "Animal Farm",
    author: "George Orwell",
    thoughts:
      "Orwell's writing was really sharp on this one. This could be read by a teenager.",
    category: "Fiction",
  },
  {
    title: "Python One-Liners",
    author: "Christian Mayer",
    thoughts: "Cool party tricks and some useful patterns for my workshops",
    category: "Non-Fiction",
  },
  {
    title: "AI: A Modern Approach",
    author: "Stuart Russell and Peter Norvig",
    thoughts:
      "I only read this partially, but I found the definitions of AI and various techniques to be useful for my workshops : D",
    category: "Non-Fiction",
  },
];

export default function BooksPage() {
  const fiction = books.filter((b) => b.category === "Fiction").length;
  const nonFiction = books.filter((b) => b.category === "Non-Fiction").length;

  return (
    <PageShell title="Books I Read in 2025" current="/blog" maxWidth="52rem">
      <p className="mb-4 text-base">
        <Link href="/blog">◄ back to blog</Link>
      </p>

      <article className="neo-prose">
        <p className="text-base neo-muted font-mono">December 31, 2025</p>
        <p>
          I got back to reading in 2025, in part thanks to Gen Z book club. Here
          are the books I read and some thoughts on each of them :DD
        </p>

        <div className="grid grid-cols-3 gap-3 my-6">
          <div className="neo-stat">
            <p className="neo-stat-num m-0">{books.length}</p>
            <p className="text-base neo-muted m-0">Books Read</p>
          </div>
          <div className="neo-stat">
            <p className="neo-stat-num m-0">{fiction}</p>
            <p className="text-base neo-muted m-0">Fiction</p>
          </div>
          <div className="neo-stat">
            <p className="neo-stat-num m-0">{nonFiction}</p>
            <p className="text-base neo-muted m-0">Non-Fiction</p>
          </div>
        </div>

        <h2>The Books</h2>
        <ul className="list-none p-0 m-0 flex flex-col gap-3">
          {books.map((book) => (
            <li key={book.title} className="neo-link-card">
              <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-2">
                <div>
                  <h3 className="m-0">{book.title}</h3>
                  <p className="text-base neo-muted m-0">by {book.author}</p>
                </div>
                <span className="neo-tag shrink-0">{book.category}</span>
              </div>
              <p className="m-0 mt-2">{book.thoughts}</p>
            </li>
          ))}
        </ul>

        <h2>Looking Ahead</h2>
        <p>
          I&apos;ll try to read 20 books in 2026 and take advantage of the power
          of audiobooks.
        </p>
      </article>
    </PageShell>
  );
}
