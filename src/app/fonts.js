import { Fredoka, Nunito } from "next/font/google";

// Rounded, friendly display font for titles, headings, and labels.
// Replaces VT323 — same retro-playful energy, but actually legible at small
// sizes (no pixelation/squinting).
export const display = Fredoka({
  subsets: ["latin"],
  weight: ["500", "600", "700"],
  variable: "--font-display",
  display: "swap",
});

// Highly readable rounded sans for body copy (replaces the old serif).
export const body = Nunito({
  subsets: ["latin"],
  weight: ["400", "600", "700"],
  variable: "--font-body",
  display: "swap",
});

// Backwards-compatible alias for older imports that referenced `pixel`.
export const pixel = display;
