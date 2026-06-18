// src/app/archive/page.js

import Image from "next/image";
import Link from "next/link";
import PageShell from "@/components/neo/PageShell";

export const metadata = {
  title: "Past Iterations",
  description:
    "A timeline of how stimmie.dev has evolved over the years, from the Himitsu Adobe Portfolio era to the Neocities revival.",
};

// Version history - add new versions at the top!
const versions = [
  {
    version: "3.0",
    name: "Neocities Revival",
    date: "June 2026",
    current: true,
    description:
      "A full pivot to intentional retro web aesthetics, rebuilt for accessibility: a starfield background, sidebar navigation, marquee, a working visitor counter, and an 88×31 button. Every tab now shares one consistent Neocities look with WCAG-minded contrast, focus rings, and skip-to-content.",
    features: [
      "Classic personal homepage layout with sidebar + main column",
      "Neocities styling applied across every tab",
      "Readable rounded sans-serif body text, friendly display headings, ridge-bordered boxes",
      "Accessibility pass: skip link, focus states, landmarks, AA contrast",
      "Working hosted visitor counter and SemVer changelog",
    ],
    screenshot: "/archive/v3-screenshot.png",
    liveUrl: null,
    techStack: ["Next.js 16", "Tailwind CSS", "Fredoka"],
  },
  {
    version: "2.0",
    name: "Bento Box Era",
    date: "January 2026 - June 2026",
    current: false,
    description:
      "A complete redesign featuring a modern bento box grid layout with dark theme. Added GitHub contribution calendar, improved accessibility, subdomain redirects, and dedicated pages for projects and talks.",
    features: [
      "Bento box grid layout",
      "GitHub contribution calendar integration",
      "Full accessibility support (WCAG compliant)",
      "Subdomain redirect system",
      "Dedicated Projects & Talks pages",
      "OWASP security headers",
      "Responsive full-screen design",
    ],
    screenshot: "/archive/v2-screenshot.png",
    liveUrl: "/v2",
    techStack: ["Next.js 14", "Tailwind CSS", "Framer Motion", "Vercel"],
  },
  {
    version: "1.0",
    name: "Original Portfolio",
    date: "2024 - January 2026",
    current: false,
    description:
      "The original portfolio design featuring an image grid background with a centered overlay containing all content in a single scrollable card. Simple but functional.",
    features: [
      "Image grid background sorted by hue",
      "Centered scrollable content overlay",
      "Talks and projects sections",
      "Social media links",
      "Minimizable overlay",
    ],
    screenshot: "/archive/v1-screenshot.png",
    liveUrl: "/archive/v1",
    techStack: ["Next.js", "Tailwind CSS", "Framer Motion"],
  },
  {
    version: "0.2",
    name: "Carrd",
    date: "~2023",
    current: false,
    description:
      "A simple one-page site built using Carrd's drag-and-drop builder. Quick to set up and served as a simple landing page.",
    features: [
      "Single-page layout",
      "Quick setup with Carrd builder",
      "Social links",
    ],
    screenshot: "/archive/carrd-screenshot.png",
    liveUrl: "https://stimmie.carrd.co/",
    techStack: ["Carrd"],
  },
  {
    version: "0.1",
    name: "Linktree",
    date: "~2022",
    current: false,
    description:
      "Before building custom portfolios, I used Linktree as a simple link-in-bio solution to aggregate all my social profiles.",
    features: ["Link-in-bio aggregator", "All social profiles in one place"],
    screenshot: "/archive/linktree-screenshot.png",
    techStack: ["Linktree"],
  },
  {
    version: "0.0",
    name: "Himitsu (Adobe Portfolio)",
    date: "~2021",
    current: false,
    description:
      "My first proper portfolio, built on Adobe Portfolio while freelancing in the Minecraft community as Himitsu. Split-screen landing page, Behance-linked work samples, and a yellow-header portfolio for server configuration gigs.",
    features: [
      "Split layout with in-game community photo",
      "Minecraft plugin configuration portfolio",
      "Discord and MC-Market contact links",
      "Behance, Instagram, GitHub, and email links",
    ],
    screenshot: "/archive/himitsu-screenshot.png",
    liveUrl: "/archive/himitsu",
    techStack: ["Adobe Portfolio"],
  },
];

export default function ArchivePage() {
  return (
    <PageShell
      title="~ past iterations ~"
      intro="A timeline of how this portfolio has evolved over time. Each version represents a milestone in my journey as a developer."
      current="/archive"
      maxWidth="56rem"
    >
      <ol className="flex flex-col gap-6 list-none p-0 m-0">
        {versions.map((v) => (
          <li key={v.version}>
            <article
              className="neo-link-card"
              style={
                v.current
                  ? { borderColor: "#d6008f", boxShadow: "4px 4px 0 #d6008f" }
                  : undefined
              }
            >
              {v.screenshot && (
                <Image
                  src={v.screenshot}
                  alt={`Screenshot of version ${v.version}`}
                  width={900}
                  height={506}
                  className="neo-thumb-lg w-full aspect-video object-cover mb-3"
                />
              )}

              <p className="m-0 mb-2 flex flex-wrap items-center gap-2">
                <span className="neo-badge neo-badge-talk">v{v.version}</span>
                {v.current && (
                  <span className="neo-badge neo-badge-workshop">Current</span>
                )}
                <span
                  className="text-base neo-muted"
                  style={{ fontFamily: "var(--neo-ui)" }}
                >
                  {v.date}
                </span>
              </p>

              <h2 className="font-bold text-xl m-0">{v.name}</h2>
              <p className="mt-1 mb-3">{v.description}</p>

              <h3 className="font-bold text-base m-0 mb-1">Key Features</h3>
              <ul
                className="list-square pl-5 m-0 mb-3"
                style={{ listStyle: "square" }}
              >
                {v.features.map((feature, fIdx) => (
                  <li key={fIdx}>{feature}</li>
                ))}
              </ul>

              <p className="m-0 mb-3">
                {v.techStack.map((tech) => (
                  <span key={tech} className="neo-tag">
                    {tech}
                  </span>
                ))}
              </p>

              <p className="m-0 flex flex-wrap gap-4">
                {v.liveUrl && !v.current && (
                  <Link
                    href={v.liveUrl}
                    target={v.liveUrl.startsWith("http") ? "_blank" : undefined}
                    rel={
                      v.liveUrl.startsWith("http")
                        ? "noopener noreferrer"
                        : undefined
                    }
                  >
                    View archived version →
                  </Link>
                )}
                {v.extraLinks?.map((link, lIdx) => (
                  <Link
                    key={lIdx}
                    href={link.url}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    {link.label} →
                  </Link>
                ))}
              </p>
            </article>
          </li>
        ))}
      </ol>

      <div
        className="mt-8 p-5 text-center"
        style={{ border: "2px dashed #999", background: "#fafafa" }}
      >
        <h2 className="neo-section-title">what&apos;s next?</h2>
        <p className="m-0">
          Stay tuned for future iterations. This portfolio is always evolving!
        </p>
      </div>
    </PageShell>
  );
}
