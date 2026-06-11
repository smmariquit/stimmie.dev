// src/app/r/[...slug]/page.js

"use client";

import { notFound, useParams } from "next/navigation";
import { useEffect } from "react";
import { lookupApexRedirect, TODO_TARGET } from "@/data/redirects";

// /r/<slug> handler for short links on the apex host.
//
// In production these requests are usually intercepted by Cloudflare's
// Bulk Redirects (see scripts/cf-migrate-redirects.mjs) so this page is
// effectively a fallback for:
//   * local dev (no CF in front),
//   * preview deployments where the BR ruleset isn't attached,
//   * any slug we add in code before it's been synced to CF.

export default function RedirectPage() {
  const params = useParams();
  const destination = lookupApexRedirect(params?.slug);

  // useEffect-driven client redirect (rather than a server redirect) gives
  // us a "Redirecting…" frame so the user sees something even on slow
  // networks. The notFound() below covers SSR for unknown slugs.
  useEffect(() => {
    if (destination && destination !== TODO_TARGET) {
      window.location.href = destination;
    }
  }, [destination]);

  if (!destination || destination === TODO_TARGET) {
    notFound();
  }

  return (
    <div className="min-h-screen bg-black flex items-center justify-center">
      <div className="text-center text-white">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-white mx-auto mb-4" />
        <p className="text-lg">Redirecting...</p>
        <p className="text-sm text-gray-400 mt-2">
          If you&apos;re not redirected,{" "}
          <a href={destination} className="text-blue-400 hover:underline">
            click here
          </a>
        </p>
      </div>
    </div>
  );
}
