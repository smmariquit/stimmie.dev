"use client";
import Image from "next/image";
import { motion } from "framer-motion";
import { useState } from "react";
import imageData from "../../public/image_data_summary.json";

// imageData is an object keyed by filename-stem; convert to an array and attach a public src.
const images = Object.values(imageData).map((it) => ({
  ...it,
  src: `/images/${it.file}`,
  thumbSrc: `/images_small/${it.file.replace(/\.[^.]+$/, ".webp")}`,
}));

// sort images by mean_hue (ascending) so neighbouring tiles flow through
// the colour wheel rather than clashing — gives the grid a gradient feel.
const imagesSortedByMeanRGB = [...images].sort(
  (a, b) => a.mean_hue - b.mean_hue
);

// Discrete rotation buckets. Using a fixed palette of angles (rather than
// random) keeps SSR/CSR markup stable and prevents hydration mismatches.
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

  // image.mean_rgb is a 0..1 float triple from the image-summary pipeline.
  // Multiply by 255 to get a real swatch for the placeholder background.
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
        style={{
          filter:
            "sepia(0.35) saturate(1.4) brightness(0.5) contrast(1.18)",
        }}
        loading="lazy"
      />
      {/* Per-tile fractal-noise grain */}
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
      {/* Per-tile vignette tinted by the image's own dominant colour */}
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

/**
 * Full-viewport whimsical scrapbook-mosaic background.
 *
 * Renders a dense CSS-grid of hue-sorted photo tiles with sepia/noise/vignette
 * treatment, plus atmospheric vignette and film-grain overlays.
 *
 * Usage:
 *   <MosaicBackground />
 *   …your page content (should have position:relative and z-index >= 10)…
 */
export default function MosaicBackground() {
  return (
    <>
      {/* Background Image Grid: dense scrapbook-style mosaic.
          The list is tripled so the smaller tiles still cover tall
          viewports; mixed spans + dense flow create organic rhythm. */}
      <motion.div
        className="bg-tile-grid fixed inset-0 grid w-full mx-auto"
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

      {/* Atmospheric vignette: darkens edges, focuses the eye on content */}
      <div
        aria-hidden="true"
        className="pointer-events-none fixed inset-0"
        style={{
          zIndex: 1,
          background:
            "radial-gradient(ellipse at center, transparent 25%, rgba(0,0,0,0.55) 75%, rgba(0,0,0,0.85) 100%)",
        }}
      />

      {/* Whole-viewport film grain at low opacity — analog warmth */}
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
    </>
  );
}
