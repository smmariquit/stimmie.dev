"use client";
import Image from "next/image";
import Link from "next/link";
import { motion } from "framer-motion";
import { useState } from "react";
import imageData from "../../../public/image_data_summary.json";

// Reuse the same image background from the home page
const images = Object.values(imageData).map((it) => ({
  ...it,
  src: `/images/${it.file}`,
  thumbSrc: `/images_small/${it.file.replace(/\.[^.]+$/, ".webp")}`,
}));

const imagesSortedByMeanRGB = [...images].sort((a, b) => a.mean_hue - b.mean_hue);

// Blog posts data
const blogPosts = [
  {
    slug: "books",
    title: "Books I Read in 2025",
    date: "December 31, 2025",
    excerpt: "A reflection on all the books I devoured this year—fiction, non-fiction, and everything in between.",
    coverImage: "/images/1.jpg",
  },
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

  const bg = image.mean_rgb
    ? `rgb(${Math.round(image.mean_rgb[0])}, ${Math.round(image.mean_rgb[1])}, ${Math.round(image.mean_rgb[2])})`
    : "#111";

  return (
    <div style={{ position: "relative", width: "100%", paddingTop: "100%", background: bg, overflow: "hidden" }}>
      <Image
        src={src}
        alt={image.file || "image"}
        onError={handleError}
        width={320}
        height={320}
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          width: "100%",
          height: "100%",
          objectFit: "cover",
          filter: "sepia(0.5) blur(1px)",
        }}
        loading="lazy"
      />
    </div>
  );
}

export default function BlogPage() {
  return (
    <div className="absolute inset-0 min-h-screen bg-black w-full overflow-hidden">
      {/* Background image grid */}
      <motion.div
        className="fixed inset-0 grid w-full mx-auto"
        style={{ zIndex: 0, gridTemplateColumns: "repeat(auto-fit, minmax(64px, 1fr))", gap: "0.5rem" }}
      >
        {imagesSortedByMeanRGB.map((image, idx) => (
          <motion.div key={`${image.src}-${idx}`}>
            <Thumb image={image} />
          </motion.div>
        ))}
        {imagesSortedByMeanRGB.map((image, idx) => (
          <motion.div key={`${image.src}-dup-${idx}`}>
            <Thumb image={image} />
          </motion.div>
        ))}
      </motion.div>

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
