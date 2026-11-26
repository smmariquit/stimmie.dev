// src/app/hire/page.js

import Image from "next/image";
import Link from "next/link";
import { grotesk } from "../fonts";

const SHARE_TITLE = "Simonee Mariquit, full-stack developer";
const DESCRIPTION =
  "I've freelanced for 20+ clients since 2022, and about 1,000 students use Elbi GradeSim in Chrome. Start with one paid week.";
const EMAIL = "mailto:semariquit@gmail.com?subject=Project%20inquiry";

export const metadata = {
  title: "Hire me",
  description: DESCRIPTION,
  alternates: { canonical: "/hire" },
  openGraph: {
    title: SHARE_TITLE,
    description: DESCRIPTION,
    url: "/hire",
    siteName: "Stimmie",
    locale: "en_US",
    type: "website",
    images: [
      {
        url: "/hire/opengraph-image",
        width: 1200,
        height: 630,
        alt: "Full-stack developer for your React project, beside a screenshot of the Room TBA campus map",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: SHARE_TITLE,
    description: DESCRIPTION,
    images: ["/hire/opengraph-image"],
  },
};

export default function HirePage() {
  return (
    <div className={`hire-page ${grotesk.variable}`}>
      <header className="hire-hero">
        <div className="hire-wrap">
          <Link href="/" className="hire-name">
            Simonee Mariquit
          </Link>
          <h1>Full-stack developer for your React project</h1>
          <p className="hire-lede">
            I've freelanced for 20+ clients since 2022. I build mostly in React
            and Next.js.
          </p>
          <div className="hire-actions">
            <a className="hire-btn" href={EMAIL}>
              Email me
            </a>
            <a
              className="hire-link"
              href="https://cv.stimmie.dev"
              target="_blank"
              rel="noopener noreferrer"
            >
              Read my CV
            </a>
          </div>
        </div>
      </header>

      <main className="hire-wrap hire-work">
        <article className="hire-project hire-project--wide">
          <a
            className="hire-shot"
            href="https://room-tba.uplb.tools"
            target="_blank"
            rel="noopener noreferrer"
          >
            <Image
              src="/projects/room-tba.png"
              alt="Room TBA showing a 3D map of the UP Los Baños campus with the rooms inside one building listed beside it"
              width={1920}
              height={964}
              sizes="(min-width: 72rem) 66rem, 92vw"
              priority
            />
          </a>
          <div className="hire-project-text">
            <h2>Room TBA</h2>
            <div>
              <p>
                Room TBA is the open-source campus map of UP Los Baños. I
                maintain it with 20 contributors, and it has 100+ releases,
                Playwright end-to-end tests and CI.
              </p>
              <p>
                UP Visayas and Bulacan State University forked it for their own
                campuses. It's a Svelte and TypeScript app on Postgres.
              </p>
            </div>
          </div>
        </article>

        <article className="hire-project hire-project--side">
          <div className="hire-project-text">
            <h2>Elbi GradeSim</h2>
            <div>
              <p>
                Elbi GradeSim lets UPLB students simulate their grades and plan
                for Latin honors. About 1,000 students use the Chrome extension.
              </p>
            </div>
          </div>
          <a
            className="hire-shot"
            href="https://gradesim.stimmie.dev"
            target="_blank"
            rel="noopener noreferrer"
          >
            <Image
              src="/projects/elbi-gradesim.png"
              alt="Elbi GradeSim showing a student's grades by semester and a running GWA"
              width={1891}
              height={951}
              sizes="(min-width: 72rem) 38rem, 92vw"
            />
          </a>
        </article>

        <section className="hire-offer">
          <h2>Start with one paid week</h2>
          <p>
            You pay for the first week as a small milestone. You see how I work
            before you commit to anything longer.
          </p>
          <a className="hire-btn" href={EMAIL}>
            Email me
          </a>
        </section>
      </main>

      <footer className="hire-wrap hire-footer">
        <a
          href="https://github.com/smmariquit"
          target="_blank"
          rel="noopener noreferrer"
        >
          GitHub
        </a>
        <a
          href="https://www.linkedin.com/in/stimmie/"
          target="_blank"
          rel="noopener noreferrer"
        >
          LinkedIn
        </a>
        <Link href="/projects">More projects</Link>
      </footer>
    </div>
  );
}
