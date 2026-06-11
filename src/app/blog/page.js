// src/app/blog/page.js

"use client";
import Link from "next/link";
import MosaicBackground from "@/components/MosaicBackground";

import { blogPosts } from "@/data/blogs";

export default function BlogPage() {
  return (
    <div className="min-h-screen bg-black w-full overflow-hidden">
      <MosaicBackground />

      {/* Blog content overlay */}
      <div className="fixed inset-0 flex items-center justify-center pointer-events-none" style={{ zIndex: 50 }}>
        <div className="pointer-events-auto bg-gray-800/70 text-white p-6 rounded backdrop-blur-sm max-h-10/12 max-w-3xl w-full mx-4 overflow-y-auto no-scrollbar">
          <div className="flex flex-col">
            {/* Header */}
            <div className="flex items-center justify-between mb-6">
              <Link href="/" className="text-white/70 hover:text-white transition-colors flex items-center gap-2">
                <span>←</span>
                <span>Home</span>
              </Link>
            </div>

            <h1 className="font-sans text-5xl font-black text-white mb-2">Blog</h1>
            <p className="text-white/70 mb-8">Thoughts, reflections, and stories.</p>

            {/* Blog post list */}
            <div className="flex flex-col gap-6">
              {blogPosts.map((post) => (
                <Link key={post.slug} href={`/blog/${post.slug}`} className="group">
                  <article className="bg-white/5 p-4 rounded hover:bg-white/10 transition-colors">
                    <div className="flex flex-col sm:flex-row gap-4">
                      <div className="flex-1">
                        <h2 className="text-xl font-bold text-white group-hover:text-white/90 transition-colors">
                          {post.title}
                        </h2>
                        <p className="text-white/50 text-sm mt-1">{post.date}</p>
                        <p className="text-white/80 mt-2">{post.excerpt}</p>
                      </div>
                    </div>
                  </article>
                </Link>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
