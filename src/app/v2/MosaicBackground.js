// src/app/v2/MosaicBackground.js

"use client";
import { motion } from "framer-motion";
import Image from "next/image";
import { useState } from "react";
import { archiveMosaicImagesByHue } from "@/archive/mosaic";

const TILE_ROTATIONS = [-2.4, 1.6, -0.8, 2.1, -1.5, 0.7, -2.0, 1.2];

function Thumb({ image }) {
  const [src, setSrc] = useState(image.thumbSrc);
  const [triedFallback, setTriedFallback] = useState(false);

  const handleError = () => {
    if (!triedFallback && image.src) {
      setTriedFallback(true);
      setSrc(image.src);
    }
  };

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
        width={96}
        height={96}
        className="object-cover w-full h-full"
        style={{ filter: "sepia(0.45) saturate(0.85)" }}
        loading="lazy"
      />
    </div>
  );
}

export default function MosaicBackground() {
  const tripled = [
    ...archiveMosaicImagesByHue,
    ...archiveMosaicImagesByHue,
    ...archiveMosaicImagesByHue,
  ];

  return (
    <div
      className="fixed inset-0 overflow-hidden pointer-events-none"
      style={{ zIndex: 0 }}
      aria-hidden="true"
    >
      <div
        className="absolute inset-0 bg-tile-grid"
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(48px, 1fr))",
          gridAutoFlow: "dense",
          gap: "2px",
          transform: "scale(1.15)",
          filter: "sepia(0.35) contrast(1.05)",
        }}
      >
        {tripled.map((image, i) => (
          <motion.div
            key={`${image.src}-${i}`}
            className="bg-tile relative"
            style={{
              "--rot": `${TILE_ROTATIONS[i % TILE_ROTATIONS.length]}deg`,
              gridColumn: "span 1",
              gridRow: "span 1",
            }}
          >
            <Thumb image={image} />
          </motion.div>
        ))}
      </div>
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          background:
            "radial-gradient(ellipse at center, transparent 30%, rgba(0,0,0,0.65) 100%)",
        }}
      />
      <div
        className="absolute inset-0 opacity-[0.04] pointer-events-none mix-blend-overlay"
        style={{
          backgroundImage:
            "url(\"data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E\")",
        }}
      />
    </div>
  );
}
