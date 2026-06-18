// src/app/archive/himitsu/page.js

"use client";

import Image from "next/image";
import Link from "next/link";
import { useState } from "react";

const views = [
  {
    id: "landing",
    label: "Landing",
    src: "/archive/himitsu/landing.png",
    alt: "Himitsu portfolio landing page — split layout with Minecraft group photo and intro",
  },
  {
    id: "portfolio",
    label: "Portfolio",
    src: "/archive/himitsu/portfolio.png",
    alt: "Himitsu Minecraft Server Configuration Portfolio — Adobe Portfolio template",
  },
];

function ArchiveBanner() {
  return (
    <div className="fixed top-0 left-0 right-0 bg-[#f5c518] text-black text-center py-2 z-[100] text-sm font-medium">
      You&apos;re viewing an archived snapshot (v0.0) from the Himitsu / Adobe
      Portfolio era (~2021).
      <Link href="/archive" className="underline ml-2 font-bold">
        Back to timeline →
      </Link>
      <Link href="/" className="underline ml-2 font-bold">
        Current site →
      </Link>
    </div>
  );
}

export default function ArchivedHimitsu() {
  const [activeView, setActiveView] = useState("landing");
  const current = views.find((view) => view.id === activeView) ?? views[0];

  return (
    <div className="min-h-screen bg-white pt-10">
      <ArchiveBanner />

      <main className="mx-auto max-w-6xl px-4 py-8">
        <header className="mb-6 text-center">
          <p className="m-0 text-sm uppercase tracking-[0.2em] text-neutral-500">
            himitsulol · Adobe Portfolio
          </p>
          <h1 className="mt-2 text-3xl font-bold text-neutral-900">
            Minecraft freelancing era
          </h1>
          <p className="mx-auto mt-3 max-w-2xl text-neutral-600">
            Preserved screenshots from my first real portfolio — plugin configs,
            Discord setup, MC-Market gigs, and community work, back when I went
            by Himitsu.
          </p>
        </header>

        <div
          className="mb-4 flex flex-wrap justify-center gap-2"
          role="tablist"
          aria-label="Archived page views"
        >
          {views.map((view) => (
            <button
              key={view.id}
              type="button"
              role="tab"
              aria-selected={activeView === view.id}
              onClick={() => setActiveView(view.id)}
              className="rounded-full px-4 py-2 text-sm font-medium transition-colors"
              style={{
                background: activeView === view.id ? "#2d2d2d" : "#f3f3f3",
                color: activeView === view.id ? "#fff" : "#333",
              }}
            >
              {view.label}
            </button>
          ))}
        </div>

        <figure
          className="overflow-hidden rounded-lg border border-neutral-200 shadow-lg"
          role="tabpanel"
        >
          <Image
            src={current.src}
            alt={current.alt}
            width={1600}
            height={900}
            className="h-auto w-full"
            priority={activeView === "landing"}
          />
          <figcaption className="border-t border-neutral-200 bg-neutral-50 px-4 py-3 text-center text-sm text-neutral-600">
            Frozen screenshot — original site was built with Adobe Portfolio and
            is no longer live.
          </figcaption>
        </figure>
      </main>
    </div>
  );
}
