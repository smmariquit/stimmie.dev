// src/app/get14/page.js

import Link from "next/link";
import PageShell from "@/components/neo/PageShell";
import Trainer from "./Trainer";

const DESCRIPTION =
  "Unofficial practice for GET ¼!, the pen and paper programming contest by YSES. Trace Python, JavaScript and C by hand, then type what it prints.";

export const metadata = {
  title: "GET ¼ Practice",
  description: DESCRIPTION,
  alternates: { canonical: "/get14" },
  openGraph: {
    title: "GET ¼ Practice",
    description: DESCRIPTION,
    url: "https://stimmie.dev/get14",
    siteName: "Stimmie",
    locale: "en_US",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "GET ¼ Practice",
    description: DESCRIPTION,
  },
};

export default function Get14Page() {
  return (
    <PageShell
      title="~ get ¼ practice ~"
      intro="Unofficial practice for GET ¼!, the pen and paper programming contest by YSES."
      maxWidth="56rem"
    >
      <div className="g14-page">
        <p>
          I&apos;m joining{" "}
          <Link
            href="https://www.yses.org/get14"
            target="_blank"
            rel="noopener noreferrer"
          >
            GET ¼! Reprise
          </Link>
          , so I made a small set of problems to practice with. They&apos;re my
          guess at what tracing questions could look like in Python, JavaScript
          and C. They aren&apos;t from YSES, and the real contest may be very
          different.
        </p>
        <p className="mt-2">
          For fast reps on the small steps, play the{" "}
          <Link href="/get14/drills">60 second drills</Link>.
        </p>

        <h2 className="neo-section-title">how to practice</h2>
        <ol className="neo-facts list-decimal pl-6 m-0 mt-2 space-y-1">
          <li>Grab a pen and a ¼ sheet of paper.</li>
          <li>Trace the code by hand. Don&apos;t run it.</li>
          <li>Type what it prints, then check.</li>
        </ol>

        <Trainer />
      </div>
    </PageShell>
  );
}
