// src/app/archive/v1/page.js

"use client";
import { motion } from "framer-motion";
import Image from "next/image";
import Link from "next/link";
import { useState } from "react";
import { archiveMosaicImagesByHue } from "@/archive/mosaic";
import { v1Projects, v1SocialLinks, v1Talks } from "@/archive/v1/content";

// Banner to show this is archived
function ArchiveBanner() {
  return (
    <div className="fixed top-0 left-0 right-0 bg-yellow-600 text-black text-center py-2 z-[100] text-sm font-medium">
      📜 You&apos;re viewing an archived version (v1.0) of this portfolio.
      <Link href="/" className="underline ml-2 font-bold">
        View current version →
      </Link>
    </div>
  );
}

// Mosaic tiles from frozen archive bundle (public/archive/mosaic/).
const imagesSortedByMeanRGB = archiveMosaicImagesByHue;

const talks = v1Talks;
const projects = v1Projects;

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
    ? `rgb(${Math.round(image.mean_rgb[0] * 255)}, ${Math.round(image.mean_rgb[1] * 255)}, ${Math.round(image.mean_rgb[2] * 255)})`
    : "#111";

  return (
    <div
      style={{
        position: "relative",
        width: "100%",
        paddingTop: "100%",
        background: bg,
        overflow: "hidden",
      }}
    >
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

export default function ArchivedV1() {
  const [overlayMinimized, setOverlayMinimized] = useState(false);

  return (
    <div className="absolute inset-0 h-screen bg-black w-full overflow-hidden">
      <ArchiveBanner />

      {/* Background Image Grid */}
      <motion.div
        className="relative grid w-full mx-auto mt-10"
        style={{
          zIndex: 0,
          gridTemplateColumns: "repeat(auto-fit, minmax(64px, 1fr))",
          gap: "0.5rem",
        }}
      >
        {imagesSortedByMeanRGB.map((image) => (
          <motion.div key={image.src}>
            <Thumb key={image.src} image={image} />
          </motion.div>
        ))}
        {imagesSortedByMeanRGB.map((image) => (
          <motion.div key={`dup-${image.src}`}>
            <Thumb image={image} />
          </motion.div>
        ))}
      </motion.div>

      {/* Centered overlay */}
      <div
        className="fixed inset-0 flex items-center justify-center pointer-events-none"
        style={{ zIndex: 50 }}
      >
        {/* minimize / expand button */}
        <button
          onClick={() => setOverlayMinimized((v) => !v)}
          className="pointer-events-auto absolute top-14 right-4 bg-gray-700/80 text-white p-2 rounded-md hover:bg-gray-600"
          aria-label={overlayMinimized ? "Expand overlay" : "Minimize overlay"}
        >
          {overlayMinimized ? "+" : "-"}
        </button>

        {!overlayMinimized && (
          <div className="pointer-events-auto bg-gray-800/70 text-white p-4 rounded backdrop-blur-sm max-h-[80vh] max-w-[90vw] w-2xl overflow-y-auto no-scrollbar mt-10">
            <div className="flex flex-col items-center">
              <div className="flex flex-row gap-5 items-center">
                <div className="flex flex-col gap-2 items-center">
                  <h1 className="font-sans text-7xl font-black text-white text-center">
                    Stimmie
                  </h1>
                  <div className="flex flex-row gap-2 justify-center flex-wrap">
                    {v1SocialLinks.map((link) => (
                      <Link key={link.name} href={link.href} title={link.name}>
                        <Image
                          src={link.icon}
                          alt={`${link.name} Logo`}
                          width={24}
                          height={24}
                          className="rounded"
                        />
                      </Link>
                    ))}
                  </div>
                </div>
              </div>
              <div>
                <div className="flex-1 min-w-0 max-w-4xl mt-5">
                  <h2 className="font-bold text-2xl">Hi!</h2>
                  <p className="text-white">
                    I&apos;m a software engineer with a lifelong passion for
                    technology. What began with hosting Minecraft servers and
                    experimenting with cloud computers has evolved into building
                    modern tech solutions for startups, organizations, and
                    businesses.
                  </p>
                  <br />
                  <p className="text-white">
                    Outside of my career, I&apos;m also an avid book reader,
                    sometimes writer, volunteer, resource speaker, mentor,
                    runner, gym-goer, and someone who cares deeply about public
                    service :)
                  </p>
                </div>
                <div className="flex-1 min-w-0 max-w-4xl mt-5">
                  <h2 className="font-bold text-2xl">Talks</h2>
                  <p className="text-white">
                    Sometimes, I get invited to talks!
                  </p>
                  <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4 mt-3">
                    {talks.slice(0, 9).map((t, idx) => (
                      <div
                        key={idx}
                        className="flex flex-col items-center bg-white/5 p-2 rounded"
                      >
                        <Image
                          src={t.src}
                          alt={t.title}
                          width={320}
                          height={180}
                          className="object-cover rounded"
                        />
                        <p className="text-sm text-white mt-2 text-center">
                          {t.title}
                        </p>
                      </div>
                    ))}
                  </div>
                </div>
                {/* Projects section */}
                <div className="flex-1 min-w-0 max-w-4xl mt-8 w-full">
                  <h2 className="font-bold text-2xl">Projects</h2>
                  <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4 mt-3">
                    {projects.map((p, idx) => (
                      <div
                        key={idx}
                        className="flex flex-col bg-white/5 p-3 rounded"
                      >
                        <Image
                          src={p.src}
                          alt={p.title}
                          width={320}
                          height={180}
                          className="object-cover rounded"
                        />
                        <h3 className="text-white font-semibold mt-2">
                          {p.title}
                        </h3>
                        <p className="text-sm text-white/80 mt-1">
                          {p.description}
                        </p>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
