// src/app/r/workshops/[[...slug]]/page.js

"use client";

import Link from "next/link";
import { notFound, useParams } from "next/navigation";
import { useEffect } from "react";

// Workshop redirects - add your workshop links here!
const workshops = {
  "dsg-codeit-day1": {
    title: "UPLB DSG x UPRHS CodeIT Workshop Day 1",
    url: "https://docs.google.com/presentation/d/YOUR_PRESENTATION_ID_1",
  },
  "dsg-codeit-day2": {
    title: "UPLB DSG x UPRHS CodeIT Workshop Day 2",
    url: "https://docs.google.com/presentation/d/YOUR_PRESENTATION_ID_2",
  },
  "nextstep-hacks": {
    title: "NextStep Hacks 2025 - Winning by Talking",
    url: "https://canva.link/x6tdjld7r16dmca",
  },
  "qcu-ml-python": {
    title: "JPCS - QCU Logic Unlocked Day 1 - Machine Learning with Python",
    url: "https://docs.google.com/presentation/d/YOUR_PRESENTATION_ID_4",
  },
  "dsg-canva": {
    title: "UPLB DSG Applicants' Workshop - Data Storytelling with Canva",
    url: "https://docs.google.com/presentation/d/YOUR_PRESENTATION_ID_5",
  },
  "dep-ai": {
    title: "Data Engineering Pilipinas AI Study Group",
    url: "https://docs.google.com/presentation/d/YOUR_PRESENTATION_ID_6",
  },
  // Add more workshops here as needed!
};

export default function WorkshopRedirect() {
  const params = useParams();
  const slug = Array.isArray(params.slug) ? params.slug[0] : params.slug;

  const workshop = slug ? workshops[slug] : null;

  useEffect(() => {
    if (workshop?.url && !workshop.url.includes("YOUR_PRESENTATION_ID")) {
      window.location.href = workshop.url;
    }
  }, [workshop]);

  // Show list of all workshops if no slug provided
  if (!slug) {
    return (
      <div className="min-h-screen bg-gray-900 p-8">
        <div className="max-w-2xl mx-auto">
          <h1 className="text-3xl font-bold text-white mb-6">
            🎤 Workshops & Talks
          </h1>
          <p className="text-gray-400 mb-8">
            Quick links to my presentation materials
          </p>
          <div className="space-y-3">
            {Object.entries(workshops).map(([key, value]) => (
              <Link
                key={key}
                href={`/r/workshops/${key}`}
                className="block bg-gray-800 hover:bg-gray-700 rounded-lg p-4 transition-colors"
              >
                <p className="text-white font-medium">{value.title}</p>
                <p className="text-gray-500 text-sm">
                  workshops.stimmie.dev/{key}
                </p>
              </Link>
            ))}
          </div>
          <Link
            href="/"
            className="inline-block mt-8 text-blue-400 hover:underline"
          >
            ← Back to main site
          </Link>
        </div>
      </div>
    );
  }

  if (!workshop) {
    notFound();
  }

  // Check if URL is placeholder
  if (workshop.url.includes("YOUR_PRESENTATION_ID")) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center p-8">
        <div className="text-center text-white max-w-md">
          <h1 className="text-2xl font-bold mb-4">{workshop.title}</h1>
          <p className="text-gray-400 mb-4">
            This link hasn&apos;t been configured yet. Please update the URL in
            the code.
          </p>
          <Link href="/r/workshops" className="text-blue-400 hover:underline">
            View all workshops
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-black flex items-center justify-center">
      <div className="text-center text-white">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-white mx-auto mb-4"></div>
        <p className="text-lg">Redirecting to {workshop.title}...</p>
        <p className="text-sm text-gray-400 mt-2">
          If you&apos;re not redirected,{" "}
          <a href={workshop.url} className="text-blue-400 hover:underline">
            click here
          </a>
        </p>
      </div>
    </div>
  );
}
