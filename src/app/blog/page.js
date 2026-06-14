// src/app/blog/page.js

import Link from "next/link";
import PageShell from "@/components/neo/PageShell";
import { blogPosts } from "@/data/blogs";

export const metadata = {
  title: "Blog",
  description: "Thoughts, reflections, and stories from Stimmie.",
};

export default function BlogPage() {
  return (
    <PageShell
      title="~ blog ~"
      intro="Thoughts, reflections, and stories. Mostly long-form rambles about building things and the lessons that came with them."
      current="/blog"
      maxWidth="52rem"
    >
      <ul
        className="flex flex-col gap-4 list-none p-0 m-0"
        aria-label="Blog posts"
      >
        {blogPosts.map((post) => (
          <li key={post.slug}>
            <Link href={`/blog/${post.slug}`} className="neo-link-card group">
              <span
                className="text-base neo-muted font-mono block"
                aria-hidden="true"
              >
                {post.date}
              </span>
              <span className="block font-bold text-xl mt-0.5 group-hover:text-[#cc0066]">
                {post.title}
              </span>
              {post.excerpt && (
                <span className="block mt-1">{post.excerpt}</span>
              )}
            </Link>
          </li>
        ))}
      </ul>
    </PageShell>
  );
}
